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
    filter_sparse_cloud,
    save_sparse_cloud,
    triangulate_all_pairs,
)

logger = logging.getLogger(__name__)


def _should_viz(config: PipelineConfig, stage: str) -> bool:
    """Check whether a visualization stage should run.

    Args:
        config: Pipeline configuration.
        stage: Viz stage name (one of VALID_VIZ_STAGES).

    Returns:
        True if viz is enabled and this stage should run.
    """
    viz = config.visualization
    if not viz.enabled:
        return False
    # Empty stages list = all stages
    if not viz.stages:
        return True
    return stage in viz.stages


def _collect_height_maps(
    config: PipelineConfig,
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Collect height maps from completed frame outputs for the gallery.

    Loads fused point clouds from each frame's output directory and
    grids them into height maps using scipy.

    Args:
        config: Pipeline config (for output_dir and surface.grid_resolution).

    Returns:
        List of (frame_idx, height_map, grid_x, grid_y) tuples.
    """
    from scipy.interpolate import griddata

    output_dir = Path(config.output_dir)
    height_maps = []

    for frame_dir in sorted(output_dir.glob("frame_*")):
        pcd_path = frame_dir / "point_cloud" / "fused.ply"
        if not pcd_path.exists():
            continue

        try:
            import open3d as o3d

            pcd = o3d.io.read_point_cloud(str(pcd_path))
            if not pcd.has_points():
                continue

            pts = np.asarray(pcd.points)
            frame_idx = int(frame_dir.name.split("_")[1])

            # Grid the points
            resolution = config.surface.grid_resolution
            x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
            x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
            grid_x = np.arange(x_min, x_max + resolution, resolution)
            grid_y = np.arange(y_min, y_max + resolution, resolution)
            gx, gy = np.meshgrid(grid_x, grid_y)

            height_map = griddata(
                pts[:, :2],
                pts[:, 2],
                (gx, gy),
                method="linear",
                fill_value=np.nan,
            )
            height_maps.append((frame_idx, height_map, grid_x, grid_y))
        except Exception:
            logger.warning("Could not load height map from %s", frame_dir.name)
            continue

    return height_maps


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

    # Filter calibration cameras to those with video files
    video_cameras = set(config.camera_video_map.keys())

    all_ring = calibration.ring_cameras
    all_auxiliary = calibration.auxiliary_cameras

    ring_cameras = [c for c in all_ring if c in video_cameras]
    auxiliary_cameras = [c for c in all_auxiliary if c in video_cameras]

    skipped_ring = [c for c in all_ring if c not in video_cameras]
    skipped_aux = [c for c in all_auxiliary if c not in video_cameras]
    if skipped_ring:
        logger.warning(
            "Ring cameras in calibration but missing video (skipped): %s",
            skipped_ring,
        )
    if skipped_aux:
        logger.warning(
            "Auxiliary cameras in calibration but missing video (skipped): %s",
            skipped_aux,
        )

    active_cameras = set(ring_cameras) | set(auxiliary_cameras)
    logger.info(
        "Found %d ring cameras, %d auxiliary cameras (of %d/%d in calibration)",
        len(ring_cameras),
        len(auxiliary_cameras),
        len(all_ring),
        len(all_auxiliary),
    )

    # 2. Compute undistortion maps (only for cameras with video)
    logger.info("Computing undistortion maps")
    undistortion_maps = {}
    for name in active_cameras:
        cam = calibration.cameras[name]
        undistortion_maps[name] = compute_undistortion_maps(cam)

    # 3. Create projection models (only for cameras with video)
    logger.info("Creating projection models")
    projection_models = {}
    for name in active_cameras:
        cam = calibration.cameras[name]
        K_new = undistortion_maps[name].K_new
        projection_models[name] = RefractiveProjectionModel(
            K=K_new,
            R=cam.R,
            t=cam.t,
            water_z=calibration.water_z,
            normal=calibration.interface_normal,
            n_air=calibration.n_air,
            n_water=calibration.n_water,
        ).to(device)

    # 4. Select pairs (using only active cameras)
    logger.info("Selecting camera pairs")
    camera_positions = {
        name: pos
        for name, pos in calibration.camera_positions().items()
        if name in active_cameras
    }
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
    directory. Each stage logs its completion. Visualization calls are
    gated by VizConfig and output persistence by OutputConfig.

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

    # --- [viz] Feature overlays ---
    if _should_viz(config, "features"):
        try:
            from .visualization.features import render_all_features

            logger.info("Frame %d: rendering feature visualizations", frame_idx)
            viz_dir = frame_dir / "viz"
            viz_dir.mkdir(exist_ok=True)

            # Convert tensors to numpy for viz
            np_images = {
                name: img for name, img in undistorted.items()
            }  # already numpy
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
    if config.output.save_features:
        from .features import save_features, save_matches

        features_dir = frame_dir / "features"
        features_dir.mkdir(exist_ok=True)
        for name, feats in all_features.items():
            save_features(feats, features_dir / f"{name}.pt")
        for (ref, src), match in all_matches.items():
            save_matches(match, features_dir / f"{ref}_{src}.pt")

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

    # --- Save depth maps (opt-out) ---
    if config.output.save_depth_maps:
        depth_dir = frame_dir / "depth_maps"
        depth_dir.mkdir(exist_ok=True)
        for cam_name in depth_maps:
            save_depth_map(
                depth_maps[cam_name],
                confidence_maps[cam_name],
                depth_dir / f"{cam_name}.npz",
            )

    # --- [viz] Depth map renders ---
    if _should_viz(config, "depth"):
        try:
            from .visualization.depth import render_all_depth_maps

            logger.info("Frame %d: rendering depth map visualizations", frame_idx)
            viz_dir = frame_dir / "viz"
            viz_dir.mkdir(exist_ok=True)

            # Convert depth/confidence tensors to numpy
            np_depths = {name: dm.cpu().numpy() for name, dm in depth_maps.items()}
            np_confs = {name: cm.cpu().numpy() for name, cm in confidence_maps.items()}

            render_all_depth_maps(np_depths, np_confs, viz_dir)
        except Exception:
            logger.exception("Frame %d: depth visualization failed", frame_idx)

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

    # --- Clean up intermediates after successful fusion ---
    if not config.output.keep_intermediates:
        depth_dir = frame_dir / "depth_maps"
        if depth_dir.exists():
            import shutil

            shutil.rmtree(depth_dir)
            logger.debug("Frame %d: removed intermediate depth maps", frame_idx)

    # --- Save fused point cloud (opt-out) ---
    if config.output.save_point_cloud:
        if fused_pcd.has_points():
            pcd_dir = frame_dir / "point_cloud"
            pcd_dir.mkdir(exist_ok=True)
            save_point_cloud(fused_pcd, pcd_dir / "fused.ply")
        else:
            logger.warning(
                "Frame %d: fused point cloud is empty, skipping point cloud save",
                frame_idx,
            )

    # --- Stage 9: Surface Reconstruction ---
    if fused_pcd.has_points():
        logger.info("Frame %d: reconstructing surface", frame_idx)
        mesh = reconstruct_surface(fused_pcd, config.surface)

        # --- Save mesh (opt-out) ---
        if config.output.save_mesh:
            mesh_dir = frame_dir / "mesh"
            mesh_dir.mkdir(exist_ok=True)
            save_mesh(mesh, mesh_dir / "surface.ply")
    else:
        logger.warning(
            "Frame %d: fused point cloud is empty, skipping surface reconstruction",
            frame_idx,
        )
        mesh = None

    # --- [viz] 3D scene renders ---
    if _should_viz(config, "scene"):
        try:
            from .visualization.scene import render_all_scenes

            logger.info("Frame %d: rendering 3D scene visualizations", frame_idx)
            viz_dir = frame_dir / "viz"
            viz_dir.mkdir(exist_ok=True)

            render_all_scenes(
                point_cloud=fused_pcd,
                mesh=mesh,
                output_dir=viz_dir,
            )
        except Exception:
            logger.exception("Frame %d: scene visualization failed", frame_idx)

    # --- [viz] Camera rig diagram ---
    if _should_viz(config, "rig"):
        try:
            from .visualization.rig import render_rig_diagram
            import numpy as _np

            logger.info("Frame %d: rendering rig diagram", frame_idx)
            viz_dir = frame_dir / "viz"
            viz_dir.mkdir(exist_ok=True)

            # Convert camera data to numpy
            cam_positions = {
                name: pos.cpu().numpy()
                for name, pos in ctx.calibration.camera_positions().items()
            }
            cam_rotations = {
                name: cam.R.cpu().numpy()
                for name, cam in ctx.calibration.cameras.items()
            }

            # Optional point cloud overlay
            pcd_points = None
            if fused_pcd is not None and fused_pcd.has_points():
                pcd_points = _np.asarray(fused_pcd.points)

            # Get K and image_size from first camera for frustum aspect ratio
            first_cam = next(iter(ctx.calibration.cameras.values()))
            K_np = first_cam.K.cpu().numpy()

            render_rig_diagram(
                camera_positions=cam_positions,
                camera_rotations=cam_rotations,
                water_z=ctx.calibration.water_z,
                output_path=viz_dir / "rig.png",
                K=K_np,
                image_size=first_cam.image_size,
                point_cloud_points=pcd_points,
            )
        except Exception:
            logger.exception("Frame %d: rig visualization failed", frame_idx)

    logger.info("Frame %d: complete", frame_idx)


def run_pipeline(config: PipelineConfig) -> None:
    """Run the full reconstruction pipeline over video frames.

    Performs one-time setup, then iterates over frames from the video
    files according to FrameSamplingConfig, processing each frame.
    After all frames, generates summary visualizations if enabled.

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

    # --- [viz] Summary plots ---
    if _should_viz(config, "summary"):
        try:
            from .visualization.summary import render_timeseries_gallery

            logger.info("Rendering summary visualizations")
            summary_dir = Path(config.output_dir) / "summary"
            summary_dir.mkdir(parents=True, exist_ok=True)

            # Build height maps from frame outputs for gallery
            height_maps = _collect_height_maps(config)
            if height_maps:
                render_timeseries_gallery(
                    height_maps, summary_dir / "timeseries_gallery.png"
                )
        except Exception:
            logger.exception("Summary visualization failed")

    logger.info("Pipeline complete")
