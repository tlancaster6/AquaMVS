"""Depth map fusion into unified point clouds."""

import torch
import torch.nn.functional as F

from .config import FusionConfig
from .projection.protocol import ProjectionModel


def _sample_depth_map(
    depth_map: torch.Tensor,
    pixels: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Sample a depth map at sub-pixel locations via bilinear interpolation.

    Args:
        depth_map: Depth map, shape (H, W), float32. NaN for invalid.
        pixels: Pixel coordinates (u, v), shape (K, 2), float32.
        valid: Validity mask, shape (K,), bool. Invalid pixels are skipped.

    Returns:
        Sampled depths, shape (K,), float32. NaN for invalid pixels or
        out-of-bounds locations.
    """
    H, W = depth_map.shape
    result = torch.full((pixels.shape[0],), float("nan"), device=pixels.device)

    if not valid.any():
        return result

    # Replace NaN in depth map with 0 for grid_sample (NaN poisons bilinear)
    depth_clean = torch.where(
        torch.isnan(depth_map), torch.zeros_like(depth_map), depth_map
    )
    nan_mask = torch.isnan(depth_map).float()  # 1 where NaN

    # Normalize pixel coords to [-1, 1] for grid_sample
    valid_pixels = pixels[valid]  # (V, 2)
    grid_x = 2.0 * valid_pixels[:, 0] / (W - 1) - 1.0
    grid_y = 2.0 * valid_pixels[:, 1] / (H - 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, 1, -1, 2)

    # Sample depth
    depth_4d = depth_clean.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    sampled = F.grid_sample(
        depth_4d, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    sampled = sampled.reshape(-1)  # (V,)

    # Also sample the NaN mask to detect if any neighbor was NaN
    nan_sampled = F.grid_sample(
        nan_mask.unsqueeze(0).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    nan_sampled = nan_sampled.reshape(-1)

    # Mark as NaN where any neighbor was NaN (nan_sampled > 0)
    sampled = torch.where(
        nan_sampled > 0.01,
        torch.tensor(float("nan"), device=sampled.device),
        sampled,
    )

    result[valid] = sampled
    return result


def filter_depth_map(
    ref_name: str,
    ref_model: ProjectionModel,
    ref_depth: torch.Tensor,
    ref_confidence: torch.Tensor,
    target_names: list[str],
    target_models: dict[str, ProjectionModel],
    target_depths: dict[str, torch.Tensor],
    config: FusionConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Filter a depth map by cross-view geometric consistency.

    For each valid pixel in the reference depth map, reprojects into each
    target camera and checks depth agreement. Pixels consistent with fewer
    than min_consistent_views other cameras are invalidated.

    Args:
        ref_name: Reference camera name.
        ref_model: Projection model for the reference camera.
        ref_depth: Reference depth map, shape (H, W), float32. NaN = invalid.
        ref_confidence: Reference confidence map, shape (H, W), float32.
        target_names: List of other camera names to check against.
        target_models: Camera name to ProjectionModel mapping.
        target_depths: Camera name to depth map (H, W) mapping.
        config: Fusion configuration.

    Returns:
        filtered_depth: Depth map with inconsistent pixels set to NaN,
            shape (H, W), float32.
        filtered_confidence: Confidence map with inconsistent pixels set to 0,
            shape (H, W), float32.
        consistency_count: Number of agreeing views per pixel,
            shape (H, W), int32. 0 for invalid/filtered pixels.
    """
    H, W = ref_depth.shape
    device = ref_depth.device

    # Step 1: Build valid pixel mask (not NaN, above min_confidence)
    valid_mask = ~torch.isnan(ref_depth) & (ref_confidence >= config.min_confidence)
    valid_indices = torch.where(valid_mask.reshape(-1))[0]  # (K,) flat indices
    N_valid = valid_indices.shape[0]

    if N_valid == 0:
        # No valid pixels -- return all-invalid
        return (
            torch.full((H, W), float("nan"), device=device),
            torch.zeros(H, W, device=device),
            torch.zeros(H, W, device=device, dtype=torch.int32),
        )

    # Step 2: Get pixel coordinates for valid pixels
    v_coords = valid_indices // W  # row
    u_coords = valid_indices % W  # col
    pixels = torch.stack([u_coords.float(), v_coords.float()], dim=-1)  # (K, 2)

    # Step 3: Back-project to 3D
    depths_flat = ref_depth.reshape(-1)[valid_indices]  # (K,)
    origins, directions = ref_model.cast_ray(pixels)  # (K, 3), (K, 3)
    points_3d = origins + depths_flat.unsqueeze(-1) * directions  # (K, 3)

    # Step 4: Check consistency against each target camera
    consistency_count = torch.zeros(N_valid, device=device, dtype=torch.int32)

    for target_name in target_names:
        model_t = target_models[target_name]
        depth_t = target_depths[target_name]  # (H, W)

        # Project 3D points into target camera
        pixels_t, valid_t = model_t.project(points_3d)  # (K, 2), (K,)

        # Look up target depth at projected pixel locations (bilinear)
        depth_t_lookup = _sample_depth_map(depth_t, pixels_t, valid_t)  # (K,)

        # Cast rays in target camera to get expected ray depth
        # Only for pixels that are valid in both projection and depth lookup
        lookup_valid = valid_t & ~torch.isnan(depth_t_lookup)

        if lookup_valid.any():
            origins_t, directions_t = model_t.cast_ray(pixels_t[lookup_valid])
            d_expected = (
                (points_3d[lookup_valid] - origins_t) * directions_t
            ).sum(dim=-1)

            # Compare depths
            depth_diff = (depth_t_lookup[lookup_valid] - d_expected).abs()
            consistent = depth_diff < config.depth_tolerance

            # Accumulate count
            consistent_full = torch.zeros(N_valid, device=device, dtype=torch.bool)
            consistent_full[lookup_valid] = consistent
            consistency_count += consistent_full.int()

    # Step 5: Apply min_consistent_views threshold
    passes = consistency_count >= config.min_consistent_views

    # Step 6: Build output maps
    filtered_depth = torch.full((H, W), float("nan"), device=device)
    filtered_confidence = torch.zeros(H, W, device=device)
    consistency_map = torch.zeros(H, W, device=device, dtype=torch.int32)

    passing_indices = valid_indices[passes]
    filtered_depth.reshape(-1)[passing_indices] = depths_flat[passes]
    filtered_confidence.reshape(-1)[passing_indices] = ref_confidence.reshape(-1)[
        passing_indices
    ]
    consistency_map.reshape(-1)[valid_indices] = consistency_count

    return filtered_depth, filtered_confidence, consistency_map


def filter_all_depth_maps(
    ring_cameras: list[str],
    projection_models: dict[str, ProjectionModel],
    depth_maps: dict[str, torch.Tensor],
    confidence_maps: dict[str, torch.Tensor],
    config: FusionConfig,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Run geometric consistency filtering for all ring cameras.

    Args:
        ring_cameras: List of ring camera names (reference views).
        projection_models: Camera name to ProjectionModel mapping.
        depth_maps: Camera name to depth map (H, W) mapping.
        confidence_maps: Camera name to confidence map (H, W) mapping.
        config: Fusion configuration.

    Returns:
        Dict mapping camera name to (filtered_depth, filtered_confidence,
        consistency_count) tuple.
    """
    results = {}
    for ref_name in ring_cameras:
        # Target cameras = all other ring cameras
        target_names = [name for name in ring_cameras if name != ref_name]

        filtered_depth, filtered_conf, count = filter_depth_map(
            ref_name=ref_name,
            ref_model=projection_models[ref_name],
            ref_depth=depth_maps[ref_name],
            ref_confidence=confidence_maps[ref_name],
            target_names=target_names,
            target_models=projection_models,
            target_depths=depth_maps,
            config=config,
        )
        results[ref_name] = (filtered_depth, filtered_conf, count)

    return results
