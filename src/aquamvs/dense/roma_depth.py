"""Convert RoMa v2 dense warps to depth maps via pairwise ray triangulation."""

import numpy as np
import torch
import torch.nn.functional as F
from romav2.geometry import to_pixel

from ..config import DenseMatchingConfig, FusionConfig
from ..projection.protocol import ProjectionModel

# Cross-module private import for triangulation function (stable math interface)
from ..triangulation import _triangulate_two_rays_batch


def warp_to_pairwise_depth(
    roma_result: dict,
    ref_model: ProjectionModel,
    src_model: ProjectionModel,
    certainty_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a RoMa warp to a pairwise depth map at warp resolution.

    For each pixel in the warp grid, casts rays through both cameras and
    triangulates to find the 3D point. Computes ray depth in the reference
    camera and filters by overlap certainty and triangulation validity.

    Args:
        roma_result: Raw output from _run_roma() with keys:
            "warp_AB", "overlap_AB", "H_ref", "W_ref", "H_src", "W_src".
        ref_model: Projection model for reference camera.
        src_model: Projection model for source camera.
        certainty_threshold: Minimum overlap certainty (pixels below are invalid).

    Returns:
        depth_map: (H_warp, W_warp) float32, ray depths in meters. NaN for invalid.
        certainty: (H_warp, W_warp) float32, overlap values (0 where invalid).
    """
    warp_AB = roma_result["warp_AB"]  # (H_warp, W_warp, 2), normalized [-1, 1]
    overlap_AB = roma_result["overlap_AB"]  # (H_warp, W_warp)
    H_ref = roma_result["H_ref"]
    W_ref = roma_result["W_ref"]
    H_src = roma_result["H_src"]
    W_src = roma_result["W_src"]

    H_warp, W_warp = warp_AB.shape[:2]
    device = warp_AB.device

    # Build warp grid coordinates
    v_grid, u_grid = torch.meshgrid(
        torch.arange(H_warp, device=device),
        torch.arange(W_warp, device=device),
        indexing="ij",
    )
    u_warp = u_grid.reshape(-1).float()  # (N,)
    v_warp = v_grid.reshape(-1).float()  # (N,)
    N = u_warp.shape[0]

    # Convert warp grid coords to ref image pixel coords
    # warp grid [0, W_warp) -> normalized [-1, 1] -> image [0, W_ref)
    u_ref_norm = (u_warp / W_warp) * 2 - 1
    v_ref_norm = (v_warp / H_warp) * 2 - 1
    u_ref = (u_ref_norm + 1) / 2 * W_ref
    v_ref = (v_ref_norm + 1) / 2 * H_ref
    ref_pixels = torch.stack([u_ref, v_ref], dim=-1)  # (N, 2)

    # Convert warp values to src image pixel coords
    warp_values = warp_AB.reshape(-1, 2)  # (N, 2) in normalized coords
    src_pixels = to_pixel(warp_values, H=H_src, W=W_src)  # (N, 2)

    # Cast rays
    origins_ref, dirs_ref = ref_model.cast_ray(ref_pixels)  # (N, 3), (N, 3)
    origins_src, dirs_src = src_model.cast_ray(src_pixels)  # (N, 3), (N, 3)

    # Triangulate
    points_3d, valid_tri = _triangulate_two_rays_batch(
        origins_ref, dirs_ref, origins_src, dirs_src
    )  # (N, 3), (N,)

    # Compute ray depth in reference camera
    # depth = dot(point - origin, direction)
    depth_flat = ((points_3d - origins_ref) * dirs_ref).sum(dim=-1)  # (N,)

    # Apply validity masks
    overlap_flat = overlap_AB.reshape(-1)  # (N,)
    valid_certainty = overlap_flat > certainty_threshold
    valid_depth = depth_flat > 0  # Positive depth only
    valid = valid_tri & valid_certainty & valid_depth

    # Build output maps
    depth_map = torch.full((H_warp, W_warp), float("nan"), device=device)
    certainty_map = torch.zeros(H_warp, W_warp, device=device)

    # Fill valid pixels
    valid_indices = torch.where(valid)[0]
    if valid_indices.shape[0] > 0:
        v_valid = v_warp[valid_indices].long()
        u_valid = u_warp[valid_indices].long()
        depth_map[v_valid, u_valid] = depth_flat[valid_indices]
        certainty_map[v_valid, u_valid] = overlap_flat[valid_indices]

    return depth_map, certainty_map


def aggregate_pairwise_depths(
    pairwise_depths: list[torch.Tensor],
    depth_tolerance: float,
    min_consistent_views: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate multiple pairwise depth maps via consistency-filtered median.

    For each pixel, finds the median of all non-NaN depth values, counts how
    many depths are within depth_tolerance of the median, and outputs the
    median of that inlier set if at least min_consistent_views agree.

    Args:
        pairwise_depths: List of (H, W) depth maps, NaN for invalid.
        depth_tolerance: Max depth disagreement for consistency (meters).
        min_consistent_views: Minimum agreeing sources to keep a pixel.

    Returns:
        depth_map: (H, W) float32, aggregated depths. NaN where insufficient agreement.
        confidence: (H, W) float32, n_agreeing / n_available. 0 where invalid.
    """
    if len(pairwise_depths) == 0:
        raise ValueError("pairwise_depths cannot be empty")

    # Stack into (N_sources, H, W)
    stacked = torch.stack(pairwise_depths, dim=0)  # (N, H, W)
    N_sources, H, W = stacked.shape
    device = stacked.device

    # 1. Count valid (non-NaN) depths per pixel
    nan_mask = torch.isnan(stacked)  # (N, H, W)
    n_available = (~nan_mask).sum(dim=0).float()  # (H, W)

    # 2. Initial median (torch.nanmedian ignores NaN)
    initial_median = torch.nanmedian(stacked, dim=0).values  # (H, W)

    # 3. Find inliers: within depth_tolerance of the median
    deviation = (stacked - initial_median.unsqueeze(0)).abs()  # (N, H, W)
    inlier_mask = (~nan_mask) & (deviation < depth_tolerance)  # (N, H, W)
    n_agreeing = inlier_mask.sum(dim=0).float()  # (H, W)

    # 4. Inlier median: mask out non-inliers with NaN, then nanmedian again
    inlier_stacked = torch.where(
        inlier_mask, stacked, torch.full_like(stacked, float("nan"))
    )
    inlier_median = torch.nanmedian(inlier_stacked, dim=0).values  # (H, W)

    # 5. Apply min_consistent_views threshold
    passes = n_agreeing >= min_consistent_views
    depth_map = torch.where(
        passes, inlier_median, torch.full((H, W), float("nan"), device=device)
    )

    # 6. Confidence = n_agreeing / n_available (0 where no valid sources or insufficient agreement)
    confidence_map = torch.where(
        (n_available > 0) & passes,
        n_agreeing / n_available,
        torch.zeros(H, W, device=device),
    )

    return depth_map, confidence_map


def roma_warps_to_depth_maps(
    ring_cameras: list[str],
    pairs: dict[str, list[str]],
    all_warps: dict[tuple[str, str], dict],
    projection_models: dict[str, ProjectionModel],
    dense_matching_config: DenseMatchingConfig,
    fusion_config: FusionConfig,
    image_size: tuple[int, int],
    masks: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Convert all RoMa warps to per-camera depth maps + confidence maps.

    For each ring camera (reference), collects warps from all sources,
    converts each to pairwise depth map, aggregates via consistency filtering,
    and upsamples to full image resolution. Optionally applies ROI masks.

    Args:
        ring_cameras: Reference camera names.
        pairs: ref -> [src] mapping.
        all_warps: Directed (ref, src) -> raw warp dict from run_roma_all_pairs.
        projection_models: Camera name -> ProjectionModel.
        dense_matching_config: For certainty_threshold.
        fusion_config: For depth_tolerance, min_consistent_views.
        image_size: Full image (H, W) for upsampling target.
        masks: Optional per-camera ROI masks (H, W) uint8, 255 = valid.

    Returns:
        depth_maps: {camera_name: (H, W) float32 depth map}
        confidence_maps: {camera_name: (H, W) float32 confidence}
    """
    W_full, H_full = image_size  # image_size is (width, height) per OpenCV convention
    depth_maps = {}
    confidence_maps = {}

    for ref_name in ring_cameras:
        src_names = pairs.get(ref_name, [])
        if not src_names:
            continue

        ref_model = projection_models[ref_name]

        # Collect pairwise depth maps from all sources
        pairwise_depths_list = []
        for src_name in src_names:
            warp_key = (ref_name, src_name)
            if warp_key not in all_warps:
                continue

            src_model = projection_models[src_name]
            roma_result = all_warps[warp_key]

            # Convert warp to depth
            depth_pairwise, _ = warp_to_pairwise_depth(
                roma_result,
                ref_model,
                src_model,
                dense_matching_config.certainty_threshold,
            )
            pairwise_depths_list.append(depth_pairwise)

        if len(pairwise_depths_list) == 0:
            continue

        # Aggregate pairwise depths (use roma_depth_tolerance, which is
        # relaxed vs plane-sweep depth_tolerance to account for coarser
        # warp-resolution triangulation noise -- B.16)
        depth_warp, conf_warp = aggregate_pairwise_depths(
            pairwise_depths_list,
            fusion_config.roma_depth_tolerance,
            fusion_config.min_consistent_views,
        )

        # Upsample to full resolution with NaN handling
        depth_full = _upsample_depth_map(depth_warp, (H_full, W_full))
        conf_full = _upsample_confidence_map(conf_warp, (H_full, W_full))

        # Apply mask if available
        if masks is not None and ref_name in masks:
            from ..masks import apply_mask_to_depth

            depth_full, conf_full = apply_mask_to_depth(
                depth_full, conf_full, masks[ref_name]
            )

        depth_maps[ref_name] = depth_full
        confidence_maps[ref_name] = conf_full

    return depth_maps, confidence_maps


def _upsample_depth_map(
    depth: torch.Tensor,
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Upsample a depth map to target size using valid-pixel-weighted interpolation.

    Uses normalized convolution: upsamples (depth * valid_mask) and valid_mask
    separately, then divides to get a weighted average that only uses valid
    neighbors. Only pixels with zero valid-neighbor weight become NaN. This
    avoids the boundary erosion caused by the previous approach of killing any
    pixel near a NaN source (B.16).

    Args:
        depth: (H, W) depth map, NaN for invalid.
        target_size: (H_target, W_target).

    Returns:
        Upsampled depth map (H_target, W_target).
    """
    H_target, W_target = target_size
    device = depth.device

    # Build valid mask (1 where depth is finite, 0 where NaN)
    valid_mask = (~torch.isnan(depth)).float()

    # Replace NaN with 0 so interpolation doesn't poison neighbors
    depth_clean = torch.where(torch.isnan(depth), torch.zeros_like(depth), depth)

    # Upsample depth*valid_mask (numerator) and valid_mask (denominator)
    depth_up = F.interpolate(
        depth_clean[None, None],
        size=(H_target, W_target),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    weight_up = F.interpolate(
        valid_mask[None, None],
        size=(H_target, W_target),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    # Normalize by valid-neighbor weight; NaN only where no valid neighbors
    depth_up = torch.where(
        weight_up > 1e-6,
        depth_up / weight_up,
        torch.tensor(float("nan"), device=device),
    )

    return depth_up


def _upsample_confidence_map(
    confidence: torch.Tensor,
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Upsample a confidence map to target size.

    Args:
        confidence: (H, W) confidence map.
        target_size: (H_target, W_target).

    Returns:
        Upsampled confidence map (H_target, W_target).
    """
    H_target, W_target = target_size

    conf_up = F.interpolate(
        confidence[None, None],
        size=(H_target, W_target),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    return conf_up
