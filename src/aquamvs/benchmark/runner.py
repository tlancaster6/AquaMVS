"""Benchmark sweep execution for comparing feature extraction configurations."""

import copy
import logging
import time
from pathlib import Path

import torch
from aquacal.io.video import VideoSet

from ..calibration import undistort_image
from ..config import FeatureExtractionConfig, PipelineConfig
from ..features.extraction import extract_features_batch
from ..features.matching import match_all_pairs
from ..masks import apply_mask_to_features
from ..pipeline import setup_pipeline
from ..triangulation import filter_sparse_cloud, triangulate_all_pairs
from .metrics import BenchmarkResults, ConfigResult, config_name
from .visualization import render_config_outputs

logger = logging.getLogger(__name__)


def run_benchmark(
    config: PipelineConfig,
    frame: int = 0,
) -> BenchmarkResults:
    """Run benchmark sweep across feature extraction configurations.

    Performs one-time setup (calibration, projection models, pairs, masks),
    reads and undistorts a single frame, then sweeps over all combinations
    of extractors and CLAHE settings. Collects per-stage timing and metrics.

    Args:
        config: Full pipeline configuration.
        frame: Frame index to benchmark (default: 0).

    Returns:
        BenchmarkResults containing all configuration results.
    """
    device = config.device.device

    # --- One-time setup ---
    logger.info("Setting up pipeline context")
    ctx = setup_pipeline(config)

    # --- Read and undistort one frame ---
    logger.info("Reading frame %d", frame)
    with VideoSet(config.camera_video_map) as videos:
        # Read the specified frame
        frame_found = False
        for frame_idx, raw_images in videos.iterate_frames(
            start=frame, stop=frame + 1, step=1
        ):
            if frame_idx == frame:
                frame_found = True
                break

        if not frame_found:
            raise ValueError(f"Frame {frame} not found in videos")

        # Filter out None images
        images = {name: img for name, img in raw_images.items() if img is not None}
        if not images:
            raise ValueError(f"Frame {frame} has no valid images")

    # Undistort all cameras
    logger.info("Undistorting images")
    undistorted = {}
    for name, img in images.items():
        if name in ctx.undistortion_maps:
            undistorted[name] = undistort_image(img, ctx.undistortion_maps[name])

    # Convert to tensors (shared across all configs)
    undistorted_tensors = {
        name: torch.from_numpy(img) for name, img in undistorted.items()
    }

    # Compute camera centers once (shared across all configs)
    camera_centers = {
        name: pos.cpu().numpy()
        for name, pos in ctx.calibration.camera_positions().items()
    }

    # Get image size for matching
    image_size = list(ctx.calibration.cameras.values())[0].image_size

    # Get camera names
    camera_names = sorted(undistorted.keys())

    # --- Sweep over configurations ---
    sweep_configs = []
    for extractor in config.benchmark.extractors:
        for clahe in config.benchmark.clahe:
            sweep_configs.append((extractor, clahe))

    logger.info("Running benchmark sweep: %d configurations", len(sweep_configs))

    results = []
    for extractor, clahe in sweep_configs:
        cfg_name = config_name(extractor, clahe)
        logger.info("Benchmarking: %s", cfg_name)

        # Build per-config FeatureExtractionConfig
        feat_config = copy.copy(config.feature_extraction)
        feat_config.extractor_type = extractor
        feat_config.clahe_enabled = clahe

        # --- Time extraction ---
        t0 = time.perf_counter()
        all_features = extract_features_batch(
            undistorted_tensors,
            feat_config,
            device=device,
        )
        t1 = time.perf_counter()
        extraction_time = t1 - t0

        # --- Apply masks (included in extraction time) ---
        if ctx.masks:
            for cam_name in list(all_features.keys()):
                if cam_name in ctx.masks:
                    all_features[cam_name] = apply_mask_to_features(
                        all_features[cam_name], ctx.masks[cam_name]
                    )
        t2 = time.perf_counter()
        extraction_time = t2 - t0  # Update to include mask time

        # --- Time matching ---
        t_match_start = time.perf_counter()
        all_matches = match_all_pairs(
            all_features,
            ctx.pairs,
            image_size=image_size,
            config=config.matching,
            device=device,
            extractor_type=extractor,
        )
        t_match_end = time.perf_counter()
        matching_time = t_match_end - t_match_start

        # --- Time triangulation ---
        t_tri_start = time.perf_counter()
        sparse_cloud = triangulate_all_pairs(ctx.projection_models, all_matches)
        sparse_cloud = filter_sparse_cloud(
            sparse_cloud, water_z=ctx.calibration.water_z
        )
        t_tri_end = time.perf_counter()
        triangulation_time = t_tri_end - t_tri_start

        total_time = extraction_time + matching_time + triangulation_time

        # --- Collect metrics ---
        keypoint_counts = {}
        keypoint_mean_scores = {}
        for cam_name, feats in all_features.items():
            n_kpts = feats["keypoints"].shape[0]
            keypoint_counts[cam_name] = n_kpts
            if n_kpts > 0:
                keypoint_mean_scores[cam_name] = float(feats["scores"].mean())
            else:
                keypoint_mean_scores[cam_name] = 0.0

        match_counts = {}
        for pair_key, matches in all_matches.items():
            n_matches = matches["ref_keypoints"].shape[0]
            match_counts[pair_key] = n_matches

        sparse_point_count = sparse_cloud["points_3d"].shape[0]

        # --- Render per-config visual artifacts ---
        render_config_outputs(
            config_name=cfg_name,
            undistorted_images=undistorted,
            all_features=all_features,
            all_matches=all_matches,
            sparse_cloud=sparse_cloud,
            projection_models=ctx.projection_models,
            undistorted_tensors=undistorted_tensors,
            voxel_size=config.fusion.voxel_size,
            surface_config=config.surface,
            output_dir=Path(config.output_dir) / "benchmark",
            camera_centers=camera_centers,
        )

        # --- Build ConfigResult ---
        result = ConfigResult(
            config_name=cfg_name,
            extractor_type=extractor,
            clahe_enabled=clahe,
            keypoint_counts=keypoint_counts,
            keypoint_mean_scores=keypoint_mean_scores,
            match_counts=match_counts,
            sparse_point_count=sparse_point_count,
            extraction_time=extraction_time,
            matching_time=matching_time,
            triangulation_time=triangulation_time,
            total_time=total_time,
        )
        results.append(result)

        logger.info(
            "  %s: %d keypoints, %d matches, %d sparse points, %.2fs total",
            cfg_name,
            sum(keypoint_counts.values()),
            sum(match_counts.values()),
            sparse_point_count,
            total_time,
        )

    # Get canonical pair keys
    pair_keys = sorted(ctx.pairs_flat if hasattr(ctx, "pairs_flat") else [])
    if not pair_keys:
        # Build from all_matches keys
        pair_keys = sorted(all_matches.keys())

    return BenchmarkResults(
        results=results,
        frame_idx=frame,
        camera_names=camera_names,
        pair_keys=pair_keys,
    )


__all__ = ["run_benchmark"]
