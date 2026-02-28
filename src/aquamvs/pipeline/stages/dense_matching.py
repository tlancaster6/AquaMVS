"""Dense matching stage (RoMa v2 for full and sparse modes)."""

import gc
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from ...dense import (
    aggregate_pairwise_depths,
    save_depth_map,
    upsample_confidence_map,
    upsample_depth_map,
    warp_to_pairwise_depth,
)
from ...features import match_all_pairs_roma
from ...features.roma import _run_roma, create_roma_matcher  # noqa: PLC2701
from ..context import PipelineContext

logger = logging.getLogger(__name__)


def run_roma_full_path(
    undistorted_tensors: dict[str, torch.Tensor],
    ctx: PipelineContext,
    frame_dir: Path,
    frame_idx: int,
) -> tuple[dict, dict, dict]:
    """Run RoMa v2 dense matching in full mode (warps to depth maps).

    Processes warps incrementally: each pair is matched and immediately
    converted to a pairwise depth map, avoiding accumulation of all warp
    tensors in memory. The RoMa model is deleted after all pairs are matched
    and before depth aggregation begins.

    Args:
        undistorted_tensors: Dict of undistorted BGR images as torch.Tensor.
        ctx: Pipeline context.
        frame_dir: Frame output directory.
        frame_idx: Frame index (for logging).

    Returns:
        Tuple of (depth_maps, confidence_maps, consistency_maps):
            - depth_maps: Dict mapping camera name to depth map tensor.
            - confidence_maps: Dict mapping camera name to confidence map tensor.
            - consistency_maps: Dict mapping camera name to consistency count tensor.
    """
    from ...profiling import timed_stage

    with timed_stage("dense_matching", logger):
        config = ctx.config
        device = ctx.device

        # --- RoMa + full: incremental warp -> depth conversion ---
        logger.info("Frame %d: running RoMa v2 dense matching (full mode)", frame_idx)

        # Step 1: Create matcher once
        matcher = create_roma_matcher(device)

        # Count total pairs for progress logging
        total_pairs = sum(len(srcs) for srcs in ctx.pairs.values())
        pair_idx = 0

        # Step 2: Match each pair and immediately convert warp to depth
        # Accumulate only lightweight pairwise depth tensors per reference camera
        pairwise_depths: dict[str, list[torch.Tensor]] = {
            ref: [] for ref in ctx.ring_cameras
        }

        for ref_name in ctx.ring_cameras:
            src_names = ctx.pairs.get(ref_name, [])
            ref_model = ctx.projection_models[ref_name]

            for src_name in src_names:
                pair_idx += 1
                logger.info(
                    "Matching pair %d/%d: %s -> %s",
                    pair_idx,
                    total_pairs,
                    ref_name,
                    src_name,
                )

                img_ref = undistorted_tensors[ref_name]
                img_src = undistorted_tensors[src_name]

                # Match pair
                warp_result = _run_roma(img_ref, img_src, matcher)

                # Immediately convert to pairwise depth (warp goes out of scope)
                src_model = ctx.projection_models[src_name]
                depth_pairwise, _ = warp_to_pairwise_depth(
                    warp_result,
                    ref_model,
                    src_model,
                    config.dense_matching.certainty_threshold,
                )
                pairwise_depths[ref_name].append(depth_pairwise)

        # Step 3: Delete matcher and free GPU memory before aggregation
        del matcher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 4: Aggregate pairwise depths per reference camera
        logger.info("Frame %d: aggregating pairwise depths", frame_idx)
        W_full, H_full = list(ctx.calibration.cameras.values())[0].image_size
        depth_maps = {}
        confidence_maps = {}
        consistency_maps = {}

        for ref_name in ctx.ring_cameras:
            if not pairwise_depths[ref_name]:
                continue

            depth_warp, conf_warp, consist_warp = aggregate_pairwise_depths(
                pairwise_depths[ref_name],
                config.reconstruction.roma_depth_tolerance,
                config.reconstruction.min_consistent_views,
            )

            # Free per-reference pairwise list after aggregation
            pairwise_depths[ref_name].clear()

            # Upsample to full resolution with NaN handling
            depth_full = upsample_depth_map(depth_warp, (H_full, W_full))
            conf_full = upsample_confidence_map(conf_warp, (H_full, W_full))

            # Upsample consistency counts (nearest-neighbor to preserve integers)
            consist_full = F.interpolate(
                consist_warp.float()[None, None],
                size=(H_full, W_full),
                mode="nearest",
            )[0, 0].to(torch.int32)

            # Apply mask if available
            if ctx.masks is not None and ref_name in ctx.masks:
                from ...masks import apply_mask_to_depth

                depth_full, conf_full = apply_mask_to_depth(
                    depth_full, conf_full, ctx.masks[ref_name]
                )
                # Zero out consistency where mask kills the depth
                consist_full = torch.where(
                    torch.isnan(depth_full),
                    torch.zeros_like(consist_full),
                    consist_full,
                )

            depth_maps[ref_name] = depth_full
            confidence_maps[ref_name] = conf_full
            consistency_maps[ref_name] = consist_full

        # Save depth maps (always -- viz pass reloads from disk)
        depth_dir = frame_dir / "depth_maps"
        depth_dir.mkdir(exist_ok=True)
        for cam_name in depth_maps:
            save_depth_map(
                depth_maps[cam_name],
                confidence_maps[cam_name],
                depth_dir / f"{cam_name}.npz",
            )

        return depth_maps, confidence_maps, consistency_maps


def run_roma_sparse_path(
    undistorted_tensors: dict[str, torch.Tensor],
    ctx: PipelineContext,
    frame_dir: Path,
    frame_idx: int,
) -> dict:
    """Run RoMa v2 in sparse mode (correspondences only).

    Args:
        undistorted_tensors: Dict of undistorted BGR images as torch.Tensor.
        ctx: Pipeline context.
        frame_dir: Frame output directory.
        frame_idx: Frame index (for logging).

    Returns:
        Dict with "all_matches" key containing match results.
    """
    from ...profiling import timed_stage

    with timed_stage("dense_matching", logger):
        config = ctx.config
        device = ctx.device

        # --- RoMa + sparse: correspondences -> triangulation -> sparse surface ---
        logger.info("Frame %d: matching with RoMa v2 (sparse mode)", frame_idx)
        all_matches = match_all_pairs_roma(
            undistorted_images=undistorted_tensors,
            pairs=ctx.pairs,
            config=config.dense_matching,
            device=device,
            masks=ctx.masks,
        )

        # Explicit cleanup after matching (matcher created internally by
        # match_all_pairs_roma is now eligible for GC)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save matches if requested
        if config.runtime.save_features:
            from ...features import save_matches

            features_dir = frame_dir / "features"
            features_dir.mkdir(exist_ok=True)
            for (ref, src), match in all_matches.items():
                save_matches(match, features_dir / f"{ref}_{src}.pt")

        # Feature viz is skipped (no per-camera keypoints)

        return {"all_matches": all_matches}
