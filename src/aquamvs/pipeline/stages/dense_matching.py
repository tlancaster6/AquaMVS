"""Dense matching stage (RoMa v2 for full and sparse modes)."""

import logging
from pathlib import Path

import torch

from ...dense import roma_warps_to_depth_maps, save_depth_map
from ...features import match_all_pairs_roma, run_roma_all_pairs
from ..context import PipelineContext

logger = logging.getLogger(__name__)


def run_roma_full_path(
    undistorted_tensors: dict[str, torch.Tensor],
    ctx: PipelineContext,
    frame_dir: Path,
    frame_idx: int,
) -> tuple[dict, dict, dict]:
    """Run RoMa v2 dense matching in full mode (warps to depth maps).

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

        # --- RoMa + full: warps -> depth maps -> fusion -> surface ---
        logger.info("Frame %d: running RoMa v2 dense matching (full mode)", frame_idx)
        all_warps = run_roma_all_pairs(
            undistorted_images=undistorted_tensors,
            pairs=ctx.pairs,
            config=config.dense_matching,
            device=device,
            masks=None,  # Masks applied after upsampling
        )

        logger.info("Frame %d: converting RoMa warps to depth maps", frame_idx)
        depth_maps, confidence_maps, consistency_maps = roma_warps_to_depth_maps(
            ring_cameras=ctx.ring_cameras,
            pairs=ctx.pairs,
            all_warps=all_warps,
            projection_models=ctx.projection_models,
            dense_matching_config=config.dense_matching,
            reconstruction_config=config.reconstruction,
            image_size=list(ctx.calibration.cameras.values())[0].image_size,
            masks=ctx.masks,
        )

        # Save depth maps (always — viz pass reloads from disk)
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

        # Save matches if requested
        if config.runtime.save_features:
            from ...features import save_matches

            features_dir = frame_dir / "features"
            features_dir.mkdir(exist_ok=True)
            for (ref, src), match in all_matches.items():
                save_matches(match, features_dir / f"{ref}_{src}.pt")

        # Feature viz is skipped (no per-camera keypoints)

        return {"all_matches": all_matches}
