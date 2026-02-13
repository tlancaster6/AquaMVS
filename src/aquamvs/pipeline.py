"""Pipeline orchestration for multi-frame reconstruction."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from aquacal.io.video import VideoSet

from .calibration import (
    CalibrationData,
    UndistortionData,
    compute_undistortion_maps,
    load_calibration_data,
    undistort_image,
)
from .config import PipelineConfig
from .dense import extract_depth, plane_sweep_stereo, save_depth_map
from .features import extract_features_batch, match_all_pairs, select_pairs
from .fusion import filter_all_depth_maps, fuse_depth_maps, save_point_cloud
from .projection.protocol import ProjectionModel
from .projection.refractive import RefractiveProjectionModel
from .surface import reconstruct_surface, save_mesh
from .triangulation import (
    compute_depth_ranges,
    save_sparse_cloud,
    triangulate_all_pairs,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Precomputed data that is constant across all frames.

    Created once by setup_pipeline() and reused for every frame.
    """

    config: PipelineConfig
    calibration: CalibrationData
    undistortion_maps: dict[str, UndistortionData]
    projection_models: dict[str, ProjectionModel]
    pairs: dict[str, list[str]]
    ring_cameras: list[str]
    auxiliary_cameras: list[str]
    device: str


def setup_pipeline(config: PipelineConfig) -> PipelineContext:
    """Perform one-time pipeline initialization.

    Loads calibration, creates projection models, computes undistortion
    maps, and selects camera pairs. All returned data is constant for
    the entire video session.

    Args:
        config: Full pipeline configuration.

    Returns:
        PipelineContext with all precomputed data.
    """
    device = config.device.device

    # 1. Load calibration
    logger.info("Loading calibration from %s", config.calibration_path)
    calibration = load_calibration_data(config.calibration_path)

    ring_cameras = calibration.ring_cameras
    auxiliary_cameras = calibration.auxiliary_cameras
    logger.info(
        "Found %d ring cameras, %d auxiliary cameras",
        len(ring_cameras),
        len(auxiliary_cameras),
    )

    # 2. Compute undistortion maps
    logger.info("Computing undistortion maps")
    undistortion_maps = {}
    for name, cam in calibration.cameras.items():
        undistortion_maps[name] = compute_undistortion_maps(cam)

    # 3. Create projection models (using undistorted K)
    logger.info("Creating projection models")
    projection_models = {}
    for name, cam in calibration.cameras.items():
        K_new = undistortion_maps[name].K_new
        projection_models[name] = RefractiveProjectionModel(
            K=K_new,
            R=cam.R,
            t=cam.t,
            water_z=calibration.water_z,
            normal=calibration.interface_normal,
            n_air=calibration.n_air,
            n_water=calibration.n_water,
        )

    # 4. Select pairs
    logger.info("Selecting camera pairs")
    camera_positions = calibration.camera_positions()
    pairs = select_pairs(
        camera_positions,
        ring_cameras,
        auxiliary_cameras,
        config.pair_selection,
    )

    # 5. Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 6. Save config copy
    config.to_yaml(output_dir / "config.yaml")
    logger.info("Config saved to %s", output_dir / "config.yaml")

    return PipelineContext(
        config=config,
        calibration=calibration,
        undistortion_maps=undistortion_maps,
        projection_models=projection_models,
        pairs=pairs,
        ring_cameras=ring_cameras,
        auxiliary_cameras=auxiliary_cameras,
        device=device,
    )


def process_frame(
    frame_idx: int,
    raw_images: dict[str, np.ndarray],
    ctx: PipelineContext,
) -> None:
    """Process a single frame through the full reconstruction pipeline.

    Runs all stages sequentially, saving outputs to the frame's output
    directory. Each stage logs its completion.

    Args:
        frame_idx: Frame index (for output directory naming).
        raw_images: Camera name to raw BGR image (H, W, 3) uint8 mapping.
            May contain None values for cameras that failed to read.
        ctx: Precomputed pipeline context from setup_pipeline().
    """
    config = ctx.config
    device = ctx.device

    # Create frame output directory
    frame_dir = Path(config.output_dir) / f"frame_{frame_idx:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Filter out None images (cameras that failed to read)
    images = {name: img for name, img in raw_images.items() if img is not None}
    if not images:
        logger.warning("Frame %d: no valid images, skipping", frame_idx)
        return

    # --- Stage 1: Undistort ---
    logger.info("Frame %d: undistorting images", frame_idx)
    undistorted = {}
    for name, img in images.items():
        if name in ctx.undistortion_maps:
            undistorted[name] = undistort_image(img, ctx.undistortion_maps[name])

    # Convert to tensors for feature extraction
    undistorted_tensors = {
        name: torch.from_numpy(img) for name, img in undistorted.items()
    }

    # --- Stage 2: Feature Extraction ---
    logger.info("Frame %d: extracting features", frame_idx)
    all_features = extract_features_batch(
        undistorted_tensors,
        config.feature_extraction,
        device=device,
    )

    # --- Stage 3: Feature Matching ---
    logger.info("Frame %d: matching features", frame_idx)
    all_matches = match_all_pairs(
        all_features,
        ctx.pairs,
        image_size=list(ctx.calibration.cameras.values())[0].image_size,
        config=config.matching,
        device=device,
    )

    # --- Stage 4: Sparse Triangulation ---
    logger.info("Frame %d: triangulating sparse points", frame_idx)
    sparse_cloud = triangulate_all_pairs(ctx.projection_models, all_matches)

    # Save sparse cloud (always saved)
    sparse_dir = frame_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    save_sparse_cloud(sparse_cloud, sparse_dir / "sparse_cloud.pt")

    # --- Stage 5: Depth Range Estimation ---
    logger.info("Frame %d: estimating depth ranges", frame_idx)
    depth_ranges = compute_depth_ranges(
        ctx.projection_models,
        sparse_cloud,
        margin=config.dense_stereo.depth_margin,
    )

    # --- Stage 6: Dense Stereo (per ring camera) ---
    logger.info("Frame %d: running dense stereo", frame_idx)
    depth_maps = {}
    confidence_maps = {}

    for ref_name in ctx.ring_cameras:
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
            config=config.dense_stereo,
            device=device,
        )

        # Depth extraction
        depth_map, confidence = extract_depth(
            sweep_result["cost_volume"],
            sweep_result["depths"],
        )

        depth_maps[ref_name] = depth_map
        confidence_maps[ref_name] = confidence

        logger.debug("Frame %d: %s depth extracted", frame_idx, ref_name)

    # Save depth maps
    depth_dir = frame_dir / "depth_maps"
    depth_dir.mkdir(exist_ok=True)
    for cam_name in depth_maps:
        save_depth_map(
            depth_maps[cam_name],
            confidence_maps[cam_name],
            depth_dir / f"{cam_name}.npz",
        )

    # --- Stage 7: Geometric Consistency Filtering ---
    logger.info("Frame %d: filtering depth maps", frame_idx)
    filtered = filter_all_depth_maps(
        ctx.ring_cameras,
        ctx.projection_models,
        depth_maps,
        confidence_maps,
        config.fusion,
    )

    filtered_depths = {name: f[0] for name, f in filtered.items()}
    filtered_confs = {name: f[1] for name, f in filtered.items()}

    # --- Stage 8: Depth Map Fusion ---
    logger.info("Frame %d: fusing depth maps", frame_idx)
    # Convert undistorted images to tensors for color sampling
    undistorted_for_fusion = {
        name: torch.from_numpy(img) for name, img in undistorted.items()
    }
    fused_pcd = fuse_depth_maps(
        ctx.ring_cameras,
        ctx.projection_models,
        filtered_depths,
        filtered_confs,
        undistorted_for_fusion,
        config.fusion,
    )

    # Save fused point cloud
    pcd_dir = frame_dir / "point_cloud"
    pcd_dir.mkdir(exist_ok=True)
    save_point_cloud(fused_pcd, pcd_dir / "fused.ply")

    # --- Stage 9: Surface Reconstruction ---
    logger.info("Frame %d: reconstructing surface", frame_idx)
    mesh = reconstruct_surface(fused_pcd, config.surface)

    # Save mesh
    mesh_dir = frame_dir / "mesh"
    mesh_dir.mkdir(exist_ok=True)
    save_mesh(mesh, mesh_dir / "surface.ply")

    logger.info("Frame %d: complete", frame_idx)


def run_pipeline(config: PipelineConfig) -> None:
    """Run the full reconstruction pipeline over video frames.

    Performs one-time setup, then iterates over frames from the video
    files according to FrameSamplingConfig, processing each frame.

    Args:
        config: Full pipeline configuration.
    """
    # One-time setup
    ctx = setup_pipeline(config)

    # Open video files
    logger.info("Opening video files")
    with VideoSet(config.camera_video_map) as videos:
        frame_sampling = config.frame_sampling

        logger.info(
            "Processing frames %d to %s (step %d)",
            frame_sampling.start,
            frame_sampling.stop or "end",
            frame_sampling.step,
        )

        for frame_idx, raw_images in videos.iterate_frames(
            start=frame_sampling.start,
            stop=frame_sampling.stop,
            step=frame_sampling.step,
        ):
            try:
                process_frame(frame_idx, raw_images, ctx)
            except Exception:
                logger.exception("Frame %d: processing failed, skipping", frame_idx)
                continue

    logger.info("Pipeline complete")
