"""Sparse matching stage (LightGlue feature extraction, matching, triangulation)."""

import logging
from pathlib import Path

import torch

from ...features import extract_features_batch, match_all_pairs
from ...triangulation import (
    compute_depth_ranges,
    filter_sparse_cloud,
    save_sparse_cloud,
    triangulate_all_pairs,
)
from ..context import PipelineContext
from ..helpers import _should_viz

logger = logging.getLogger(__name__)


def run_lightglue_path(
    undistorted_tensors: dict[str, torch.Tensor],
    ctx: PipelineContext,
    frame_dir: Path,
    frame_idx: int,
) -> dict:
    """Run LightGlue feature extraction and matching.

    Extracts features, applies masks, matches pairs, saves features/matches,
    and renders visualizations if enabled.

    Args:
        undistorted_tensors: Dict of undistorted BGR images as torch.Tensor.
        ctx: Pipeline context.
        frame_dir: Frame output directory.
        frame_idx: Frame index (for logging).

    Returns:
        Dict with "all_matches" key containing match results.
    """
    config = ctx.config
    device = ctx.device

    # --- Stage 2: Feature Extraction ---
    logger.info("Frame %d: extracting features", frame_idx)
    all_features = extract_features_batch(
        undistorted_tensors,
        config.sparse_matching,
        device=device,
    )

    # --- Apply masks to features ---
    if ctx.masks:
        from ...masks import apply_mask_to_features

        for cam_name in list(all_features.keys()):
            if cam_name in ctx.masks:
                n_before = all_features[cam_name]["keypoints"].shape[0]
                all_features[cam_name] = apply_mask_to_features(
                    all_features[cam_name], ctx.masks[cam_name]
                )
                n_after = all_features[cam_name]["keypoints"].shape[0]
                logger.debug(
                    "Frame %d: %s mask filtered %d -> %d keypoints",
                    frame_idx,
                    cam_name,
                    n_before,
                    n_after,
                )

    # --- Stage 3: Feature Matching ---
    logger.info("Frame %d: matching features", frame_idx)
    all_matches = match_all_pairs(
        all_features,
        ctx.pairs,
        image_size=list(ctx.calibration.cameras.values())[0].image_size,
        config=config.sparse_matching,
        device=device,
        extractor_type=config.sparse_matching.extractor_type,
    )

    # --- [viz] Feature overlays ---
    if _should_viz(config, "features"):
        try:
            from ...visualization.features import render_all_features

            logger.info("Frame %d: rendering feature visualizations", frame_idx)
            viz_dir = frame_dir / "viz"
            viz_dir.mkdir(exist_ok=True)

            # Convert tensors to numpy for viz
            # Get numpy images from undistorted (passed separately if needed)
            # For now, convert tensors back to numpy
            np_images = {
                name: img.cpu().numpy() for name, img in undistorted_tensors.items()
            }
            np_features = {
                name: {k: v.cpu().numpy() for k, v in feats.items()}
                for name, feats in all_features.items()
            }
            np_matches = {
                pair: {k: v.cpu().numpy() for k, v in match.items()}
                for pair, match in all_matches.items()
            }

            render_all_features(
                images=np_images,
                all_features=np_features,
                all_matches=np_matches,
                sparse_cloud=None,  # Not available yet at this pipeline point
                projection_models=None,
                output_dir=viz_dir,
            )
        except Exception:
            logger.exception("Frame %d: feature visualization failed", frame_idx)

    # --- Save features (opt-in) ---
    if config.runtime.save_features:
        from ...features import save_features, save_matches

        features_dir = frame_dir / "features"
        features_dir.mkdir(exist_ok=True)
        for name, feats in all_features.items():
            save_features(feats, features_dir / f"{name}.pt")
        for (ref, src), match in all_matches.items():
            save_matches(match, features_dir / f"{ref}_{src}.pt")

    return {"all_matches": all_matches}


def run_triangulation(
    all_matches: dict,
    ctx: PipelineContext,
    frame_dir: Path,
    frame_idx: int,
) -> tuple[dict, dict]:
    """Triangulate sparse point cloud and compute depth ranges.

    Args:
        all_matches: Dict of match results from run_lightglue_path.
        ctx: Pipeline context.
        frame_dir: Frame output directory.
        frame_idx: Frame index (for logging).

    Returns:
        Tuple of (sparse_cloud, depth_ranges):
            - sparse_cloud: Dict with "points_3d" and "scores" tensors.
            - depth_ranges: Dict mapping camera name to (min_depth, max_depth) tuple.
    """
    config = ctx.config

    # --- Stage 4: Sparse Triangulation ---
    logger.info("Frame %d: triangulating sparse points", frame_idx)
    sparse_cloud = triangulate_all_pairs(ctx.projection_models, all_matches)

    # Save sparse cloud (always saved)
    sparse_dir = frame_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    save_sparse_cloud(sparse_cloud, sparse_dir / "sparse_cloud.pt")

    # --- Stage 4b: Filter sparse cloud ---
    n_before = sparse_cloud["points_3d"].shape[0]
    sparse_cloud = filter_sparse_cloud(
        sparse_cloud,
        water_z=ctx.calibration.water_z,
    )
    n_after = sparse_cloud["points_3d"].shape[0]
    logger.info(
        "Frame %d: sparse cloud filtered %d -> %d points (%d removed)",
        frame_idx,
        n_before,
        n_after,
        n_before - n_after,
    )

    # --- Stage 5: Depth Range Estimation ---
    logger.info("Frame %d: estimating depth ranges", frame_idx)
    depth_ranges = compute_depth_ranges(
        ctx.projection_models,
        sparse_cloud,
        margin=config.reconstruction.depth_margin,
    )

    return sparse_cloud, depth_ranges
