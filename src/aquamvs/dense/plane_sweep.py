"""Plane sweep stereo with cost volume construction and depth extraction."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.profiler import record_function

from ..config import ReconstructionConfig
from ..projection.protocol import ProjectionModel
from .cost import aggregate_costs, compute_cost


def generate_depth_hypotheses(
    d_min: float,
    d_max: float,
    num_depths: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate uniformly-spaced depth hypotheses in ray-depth space.

    Args:
        d_min: Minimum ray depth (meters).
        d_max: Maximum ray depth (meters).
        num_depths: Number of depth hypotheses.
        device: Device for the output tensor.

    Returns:
        Depth values, shape (D,), float32, uniformly spaced from d_min to d_max
        (inclusive of both endpoints).
    """
    return torch.linspace(d_min, d_max, num_depths, device=device)


def _bgr_to_gray(image: torch.Tensor) -> torch.Tensor:
    """Convert a BGR uint8 image to grayscale float32 in [0, 1].

    Args:
        image: BGR image, shape (H, W, 3), uint8 or float32.

    Returns:
        Grayscale image, shape (H, W), float32 in [0, 1].
    """
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    # BGR to gray: 0.114*B + 0.587*G + 0.299*R
    return 0.114 * image[..., 0] + 0.587 * image[..., 1] + 0.299 * image[..., 2]


def _make_pixel_grid(
    height: int,
    width: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Create a grid of all pixel coordinates.

    Args:
        height: Image height.
        width: Image width.
        device: Device for the output tensor.

    Returns:
        Pixel coordinates (u, v), shape (H*W, 2), float32.
        u is column (0..W-1), v is row (0..H-1).
    """
    v, u = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack([u.reshape(-1), v.reshape(-1)], dim=-1)  # (H*W, 2)


def _warp_source_at_depth(
    origins: torch.Tensor,
    directions: torch.Tensor,
    src_model: ProjectionModel,
    src_image: torch.Tensor,
    depth: float,
    height: int,
    width: int,
) -> torch.Tensor:
    """Warp source image using precomputed reference rays.

    Args:
        origins: Ray origins from reference camera, shape (N, 3), float32.
        directions: Unit ray directions from reference camera, shape (N, 3), float32.
        src_model: Projection model for the source camera.
        src_image: Source image, shape (H, W), float32 grayscale in [0, 1].
        depth: Ray depth hypothesis (meters along refracted ray).
        height: Image height.
        width: Image width.

    Returns:
        Warped source image, shape (H, W), float32.
        Out-of-bounds or invalid pixels are NaN.
    """
    with record_function("grid_sample_warp"):
        H, W = height, width

        # Compute 3D points at this depth hypothesis
        points_3d = origins + depth * directions  # (N, 3)

        # Project into source camera
        src_pixels, valid = src_model.project(points_3d)  # (N, 2), (N,)

        # Convert pixel coords to normalized grid coordinates for grid_sample
        # grid_sample expects grid in [-1, 1] range
        # grid_x = 2 * u / (W - 1) - 1
        grid_x = 2.0 * src_pixels[:, 0] / (W - 1) - 1.0  # (N,)
        grid_y = 2.0 * src_pixels[:, 1] / (H - 1) - 1.0  # (N,)

        # Set invalid pixels to out-of-bounds (will get padding_mode="zeros" -> 0)
        grid_x = torch.where(valid, grid_x, torch.tensor(2.0, device=grid_x.device))
        grid_y = torch.where(valid, grid_y, torch.tensor(2.0, device=grid_y.device))

        # Reshape grid for grid_sample: (1, H, W, 2)
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, H, W, 2)

        # Source image as (1, 1, H, W) for grid_sample
        src_4d = src_image.unsqueeze(0).unsqueeze(0)

        # Bilinear sampling
        warped = F.grid_sample(
            src_4d, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )  # (1, 1, H, W)

        warped = warped.squeeze(0).squeeze(0)  # (H, W)

        # Mark invalid pixels as NaN (invalid projections + out-of-bounds)
        invalid_mask = ~valid.reshape(H, W)
        warped = torch.where(
            invalid_mask, torch.tensor(float("nan"), device=warped.device), warped
        )

        return warped


def warp_source_image(
    ref_model: ProjectionModel,
    src_model: ProjectionModel,
    src_image: torch.Tensor,
    depth: float,
    pixel_grid: torch.Tensor,
) -> torch.Tensor:
    """Warp a source image to the reference viewpoint at a given depth.

    For each reference pixel:
    1. Cast a ray through the reference projection model.
    2. Compute the 3D point at the given ray depth: point = origin + depth * direction.
    3. Project the 3D point into the source camera.
    4. Sample the source image at the projected pixel location (bilinear).

    Args:
        ref_model: Projection model for the reference camera.
        src_model: Projection model for the source camera.
        src_image: Source image, shape (H, W), float32 grayscale in [0, 1].
        depth: Ray depth hypothesis (meters along refracted ray).
        pixel_grid: Precomputed reference pixel grid, shape (H*W, 2), float32.

    Returns:
        Warped source image, shape (H, W), float32.
        Out-of-bounds or invalid pixels are NaN.
    """
    H, W = src_image.shape

    # Cast rays from reference camera for all pixels
    origins, directions = ref_model.cast_ray(pixel_grid)  # (N, 3), (N, 3)

    # Use the optimized internal function
    return _warp_source_at_depth(origins, directions, src_model, src_image, depth, H, W)


def build_cost_volume(
    ref_model: ProjectionModel,
    src_models: list[ProjectionModel],
    ref_image: torch.Tensor,
    src_images: list[torch.Tensor],
    depths: torch.Tensor,
    config: ReconstructionConfig,
) -> torch.Tensor:
    """Build a cost volume for one reference camera.

    For each depth hypothesis, warps all source images to the reference
    viewpoint, computes photometric cost between each warped source and the
    reference, and aggregates (mean) across source views.

    Supports batched depth processing for better GPU utilization via
    config.depth_batch_size.

    Args:
        ref_model: Projection model for the reference camera.
        src_models: Projection models for each source camera (len S).
        ref_image: Reference image, shape (H, W), float32 grayscale in [0, 1].
        src_images: Source images, each shape (H, W), float32 grayscale in [0, 1].
            Must be same length as src_models.
        depths: Depth hypotheses, shape (D,), float32.
        config: Dense stereo configuration.

    Returns:
        Cost volume, shape (H, W, D), float32. Lower = better match.
    """
    with record_function("build_cost_volume"), torch.no_grad():
        H, W = ref_image.shape
        D = depths.shape[0]
        S = len(src_models)
        device = ref_image.device

        # Get batch size from config (default to 1 for backward compatibility)
        batch_size = getattr(config, "depth_batch_size", 1)

        # Precompute reference pixel grid and rays (shared across all depths and sources)
        pixel_grid = _make_pixel_grid(H, W, device=device)
        origins, directions = ref_model.cast_ray(pixel_grid)  # (H*W, 3), (H*W, 3)

        # Allocate cost volume
        cost_volume = torch.zeros(H, W, D, device=device)

        # Sweep over depth hypotheses in batches
        for batch_start in range(0, D, batch_size):
            batch_end = min(batch_start + batch_size, D)

            # Process each depth in the current batch
            for d_idx in range(batch_start, batch_end):
                depth = depths[d_idx].item()

                # Compute cost for each source view at this depth
                source_costs = []
                for s_idx in range(S):
                    # Warp source image to reference viewpoint at this depth
                    warped = _warp_source_at_depth(
                        origins,
                        directions,
                        src_models[s_idx],
                        src_images[s_idx],
                        depth,
                        H,
                        W,
                    )

                    # Compute photometric cost
                    cost = compute_cost(
                        ref_image, warped, config.cost_function, config.window_size
                    )
                    source_costs.append(cost)

                # Aggregate across source views
                cost_volume[:, :, d_idx] = aggregate_costs(source_costs)

        return cost_volume


def plane_sweep_stereo(
    ref_name: str,
    ref_model: ProjectionModel,
    src_names: list[str],
    src_models: dict[str, ProjectionModel],
    ref_image: torch.Tensor,
    src_images: dict[str, torch.Tensor],
    depth_range: tuple[float, float],
    config: ReconstructionConfig,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Run plane-sweep stereo for a single reference camera.

    This is the main entry point for dense stereo. It:
    1. Generates depth hypotheses from the depth range.
    2. Converts images to grayscale float32.
    3. Builds the cost volume.
    4. Returns the cost volume and depth hypotheses for P.17 depth extraction.

    Args:
        ref_name: Reference camera name (for logging/diagnostics).
        ref_model: Projection model for the reference camera.
        src_names: List of source camera names for this reference.
        src_models: Camera name to ProjectionModel mapping (all cameras).
        ref_image: Reference image, shape (H, W, 3), uint8 or float32 BGR.
        src_images: Camera name to image mapping. Images are (H, W, 3),
            uint8 or float32 BGR.
        depth_range: (d_min, d_max) in ray-depth space (meters).
        config: Dense stereo configuration.
        device: Device for computation.

    Returns:
        Dict with keys:
            "cost_volume": shape (H, W, D), float32 -- aggregated cost volume
            "depths": shape (D,), float32 -- depth hypothesis values
            "ref_name": str -- reference camera name (passed through)
    """
    # Generate depth hypotheses
    depths = generate_depth_hypotheses(
        depth_range[0], depth_range[1], config.num_depths, device=device
    )

    # Convert reference image to grayscale float32 [0, 1]
    ref_gray = _bgr_to_gray(ref_image.to(device))

    # Collect source models and grayscale images
    src_model_list = [src_models[name] for name in src_names]
    src_gray_list = [_bgr_to_gray(src_images[name].to(device)) for name in src_names]

    # Build cost volume
    cost_volume = build_cost_volume(
        ref_model,
        src_model_list,
        ref_gray,
        src_gray_list,
        depths,
        config,
    )

    return {
        "cost_volume": cost_volume,
        "depths": depths,
        "ref_name": ref_name,
    }


def extract_depth(
    cost_volume: torch.Tensor,
    depths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract depth map and confidence map from a cost volume.

    Applies winner-take-all to find the best discrete depth, then sub-pixel
    parabola refinement for continuous depth, and estimates confidence from
    the cost profile.

    Args:
        cost_volume: Aggregated cost volume, shape (H, W, D), float32.
            Lower cost = better match.
        depths: Depth hypothesis values, shape (D,), float32.
            Uniformly spaced from d_min to d_max.

    Returns:
        depth_map: Per-pixel ray depth, shape (H, W), float32.
            Values in meters (ray-depth space). NaN for invalid pixels.
        confidence: Per-pixel confidence, shape (H, W), float32 in [0, 1].
            0 for invalid pixels, higher = more confident.
    """
    with record_function("extract_depth"):
        H, W, D = cost_volume.shape
        device = cost_volume.device
        eps = 1e-8

        # Stage 1: Winner-Take-All
        # Find the depth index with minimum cost at each pixel
        best_idx = torch.argmin(cost_volume, dim=2)  # (H, W), int64

        # Look up the discrete depth value
        depth_step = depths[1] - depths[0]  # uniform spacing
        best_depth = depths[best_idx]  # (H, W)
        best_cost = torch.gather(cost_volume, 2, best_idx.unsqueeze(-1)).squeeze(
            -1
        )  # (H, W)

        # Stage 2: Sub-Pixel Parabola Refinement
        # Get indices for neighbors (clamp to valid range)
        idx_minus = (best_idx - 1).clamp(min=0)  # (H, W)
        idx_plus = (best_idx + 1).clamp(max=D - 1)  # (H, W)

        # Gather costs at the three indices
        c_minus = torch.gather(cost_volume, 2, idx_minus.unsqueeze(-1)).squeeze(
            -1
        )  # (H, W)
        c_center = best_cost  # (H, W)
        c_plus = torch.gather(cost_volume, 2, idx_plus.unsqueeze(-1)).squeeze(
            -1
        )  # (H, W)

        # Parabola offset
        denom = c_minus - 2.0 * c_center + c_plus + eps
        offset = 0.5 * (c_minus - c_plus) / denom  # (H, W)
        offset = offset.clamp(-0.5, 0.5)

        # Refined depth
        depth_map = best_depth + offset * depth_step

        # Mark boundary pixels as invalid (cannot do parabola refinement at d_min or d_max)
        boundary_mask = (best_idx == 0) | (best_idx == D - 1)

        # Stage 3: Confidence Estimation
        # 1. Cost-based confidence: lower best_cost -> higher confidence
        # NCC cost is in [0, 2], SSIM cost is in [0, 1].
        # Normalize: confidence_cost = 1 - clamp(best_cost, 0, 1)
        confidence_cost = 1.0 - best_cost.clamp(0.0, 1.0)  # (H, W)

        # 2. Distinctness: ratio of best cost to mean cost along depth axis
        # If best is much lower than mean, the match is distinctive
        mean_cost = cost_volume.mean(dim=2)  # (H, W)
        confidence_distinct = 1.0 - (best_cost / (mean_cost + eps))  # (H, W)
        confidence_distinct = confidence_distinct.clamp(0.0, 1.0)

        # Combined confidence (geometric mean for smooth blending)
        confidence = torch.sqrt(confidence_cost * confidence_distinct)  # (H, W)

        # Apply invalid masking
        confidence = torch.where(
            boundary_mask, torch.zeros_like(confidence), confidence
        )
        depth_map = torch.where(
            boundary_mask, torch.tensor(float("nan"), device=device), depth_map
        )

        return depth_map, confidence


def save_depth_map(
    depth_map: torch.Tensor,
    confidence: torch.Tensor,
    path: str | Path,
) -> None:
    """Save depth map and confidence map to an .npz file.

    Args:
        depth_map: Per-pixel ray depth, shape (H, W), float32.
        confidence: Per-pixel confidence, shape (H, W), float32.
        path: Output file path (should end with .npz).
    """
    np.savez(
        path,
        depth=depth_map.cpu().numpy(),
        confidence=confidence.cpu().numpy(),
    )


def load_depth_map(
    path: str | Path,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load depth map and confidence map from an .npz file.

    Args:
        path: Path to .npz file.
        device: Device to place the loaded tensors on.

    Returns:
        depth_map: shape (H, W), float32.
        confidence: shape (H, W), float32.
    """
    data = np.load(path)
    depth_map = torch.from_numpy(data["depth"]).to(device)
    confidence = torch.from_numpy(data["confidence"]).to(device)
    return depth_map, confidence
