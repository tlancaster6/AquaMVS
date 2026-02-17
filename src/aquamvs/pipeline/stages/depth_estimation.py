"""Depth estimation stage (plane sweep stereo)."""

import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from ...dense import extract_depth, plane_sweep_stereo, save_depth_map
from ..context import PipelineContext

logger = logging.getLogger(__name__)


def run_depth_estimation(
    undistorted_tensors: dict[str, torch.Tensor],
    depth_ranges: dict[str, tuple[float, float]],
    ctx: PipelineContext,
    frame_dir: Path,
    frame_idx: int,
) -> tuple[dict, dict]:
    """Run plane sweep stereo to generate depth maps for all ring cameras.

    Args:
        undistorted_tensors: Dict of undistorted BGR images as torch.Tensor.
        depth_ranges: Dict mapping camera name to (min_depth, max_depth) tuple.
        ctx: Pipeline context.
        frame_dir: Frame output directory.
        frame_idx: Frame index (for logging).

    Returns:
        Tuple of (depth_maps, confidence_maps):
            - depth_maps: Dict mapping camera name to depth map tensor.
            - confidence_maps: Dict mapping camera name to confidence map tensor.
    """
    from ...profiling import timed_stage

    with timed_stage("depth_estimation", logger):
        config = ctx.config
        device = ctx.device

        # --- Stage 6: Dense Stereo (per ring camera) ---
        logger.info("Frame %d: running dense stereo", frame_idx)
        depth_maps = {}
        confidence_maps = {}

        for ref_name in tqdm(
            ctx.ring_cameras,
            desc="Plane sweep stereo",
            disable=config.runtime.quiet or not sys.stderr.isatty(),
            unit="camera",
            leave=False,
        ):
            if ref_name not in depth_ranges:
                logger.warning(
                    "Frame %d: no depth range for %s, skipping", frame_idx, ref_name
                )
                continue

            src_names = ctx.pairs.get(ref_name, [])
            if not src_names:
                logger.warning(
                    "Frame %d: no source cameras for %s, skipping", frame_idx, ref_name
                )
                continue

            # Plane sweep
            sweep_result = plane_sweep_stereo(
                ref_name=ref_name,
                ref_model=ctx.projection_models[ref_name],
                src_names=src_names,
                src_models=ctx.projection_models,
                ref_image=undistorted_tensors[ref_name],
                src_images=undistorted_tensors,
                depth_range=depth_ranges[ref_name],
                config=config.reconstruction,
                device=device,
            )

            # Depth extraction
            depth_map, confidence = extract_depth(
                sweep_result["cost_volume"],
                sweep_result["depths"],
            )

            # Apply mask to depth map
            if ctx.masks and ref_name in ctx.masks:
                from ...masks import apply_mask_to_depth

                depth_map, confidence = apply_mask_to_depth(
                    depth_map, confidence, ctx.masks[ref_name]
                )

            depth_maps[ref_name] = depth_map
            confidence_maps[ref_name] = confidence

            logger.debug("Frame %d: %s depth extracted", frame_idx, ref_name)

        # --- Save depth maps (always — viz pass reloads from disk) ---
        depth_dir = frame_dir / "depth_maps"
        depth_dir.mkdir(exist_ok=True)
        for cam_name in depth_maps:
            save_depth_map(
                depth_maps[cam_name],
                confidence_maps[cam_name],
                depth_dir / f"{cam_name}.npz",
            )

        return depth_maps, confidence_maps
