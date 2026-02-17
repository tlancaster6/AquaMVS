"""Visualization pass: reload saved artifacts from disk and render all viz stages.

This module runs after the compute pass completes and GPU memory has been
freed, avoiding Open3D OpenGL / CUDA GPU memory conflicts on Windows.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from ..config import PipelineConfig
from .context import PipelineContext
from .helpers import _collect_height_maps, _should_viz

logger = logging.getLogger(__name__)


def run_visualization_pass(config: PipelineConfig, ctx: PipelineContext) -> None:
    """Iterate frame directories and render all enabled visualization stages.

    Args:
        config: Pipeline configuration.
        ctx: Pipeline context (for calibration data needed by rig viz).
    """
    output_dir = Path(config.output_dir)

    for frame_dir in sorted(output_dir.glob("frame_*")):
        if not frame_dir.is_dir():
            continue
        try:
            _run_frame_viz(frame_dir, config, ctx)
        except Exception:
            logger.exception("Visualization failed for %s", frame_dir.name)

    # Summary plots (multi-frame)
    if _should_viz(config, "summary"):
        try:
            from ..visualization.summary import render_timeseries_gallery

            logger.info("Rendering summary visualizations")
            summary_dir = output_dir / "summary"
            summary_dir.mkdir(parents=True, exist_ok=True)

            height_maps = _collect_height_maps(config)
            if height_maps:
                render_timeseries_gallery(
                    height_maps, summary_dir / "timeseries_gallery.png"
                )
        except Exception:
            logger.exception("Summary visualization failed")


def _run_frame_viz(
    frame_dir: Path,
    config: PipelineConfig,
    ctx: PipelineContext,
) -> None:
    """Orchestrate visualization for one frame.

    Args:
        frame_dir: Frame output directory (e.g. output/frame_000000).
        config: Pipeline configuration.
        ctx: Pipeline context.
    """
    viz_dir = frame_dir / "viz"
    viz_dir.mkdir(exist_ok=True)

    frame_name = frame_dir.name
    logger.info("%s: running visualization pass", frame_name)

    if _should_viz(config, "depth"):
        try:
            _viz_depth(frame_dir, viz_dir)
        except Exception:
            logger.exception("%s: depth visualization failed", frame_name)

    if _should_viz(config, "features"):
        try:
            _viz_features(frame_dir, viz_dir, ctx)
        except Exception:
            logger.exception("%s: features visualization failed", frame_name)

    # Load point cloud and mesh for scene + rig viz
    pcd = None
    mesh = None
    if _should_viz(config, "scene") or _should_viz(config, "rig"):
        pcd, mesh = _load_scene_artifacts(frame_dir)

    if _should_viz(config, "scene"):
        try:
            _viz_scene(pcd, mesh, viz_dir, frame_name)
        except Exception:
            logger.exception("%s: scene visualization failed", frame_name)

    if _should_viz(config, "rig"):
        try:
            _viz_rig(ctx, pcd, viz_dir, frame_name)
        except Exception:
            logger.exception("%s: rig visualization failed", frame_name)


def _viz_depth(frame_dir: Path, viz_dir: Path) -> None:
    """Load depth maps from disk and render colormapped visualizations.

    Args:
        frame_dir: Frame output directory.
        viz_dir: Visualization output directory.
    """
    from ..dense.plane_sweep import load_depth_map

    depth_dir = frame_dir / "depth_maps"
    if not depth_dir.exists():
        return

    np_depths = {}
    np_confs = {}
    for npz_path in sorted(depth_dir.glob("*.npz")):
        cam_name = npz_path.stem
        depth, confidence = load_depth_map(npz_path, device="cpu")
        np_depths[cam_name] = depth.numpy()
        np_confs[cam_name] = confidence.numpy()

    if not np_depths:
        return

    from ..visualization.depth import render_all_depth_maps

    logger.info("%s: rendering depth map visualizations", frame_dir.name)
    render_all_depth_maps(np_depths, np_confs, viz_dir)


def _viz_features(frame_dir: Path, viz_dir: Path, ctx: PipelineContext) -> None:
    """Load undistorted images, features, and matches from disk and render overlays.

    Args:
        frame_dir: Frame output directory.
        viz_dir: Visualization output directory.
        ctx: Pipeline context.
    """
    from ..features.extraction import load_features
    from ..features.matching import load_matches

    undist_dir = frame_dir / "undistorted"
    features_dir = frame_dir / "features"

    if not undist_dir.exists() or not features_dir.exists():
        logger.debug(
            "%s: skipping features viz (missing undistorted/ or features/)",
            frame_dir.name,
        )
        return

    # Load undistorted images
    np_images = {}
    for img_path in sorted(undist_dir.glob("*.png")):
        cam_name = img_path.stem
        img = cv2.imread(str(img_path))
        if img is not None:
            np_images[cam_name] = img

    # Load per-camera features
    np_features = {}
    for feat_path in sorted(features_dir.glob("*.pt")):
        name = feat_path.stem
        # Skip match files (contain underscore separating ref_src)
        if "_" in name:
            continue
        feats = load_features(feat_path)
        np_features[name] = {k: v.cpu().numpy() for k, v in feats.items()}

    # Load per-pair matches
    np_matches = {}
    for match_path in sorted(features_dir.glob("*.pt")):
        name = match_path.stem
        if "_" not in name:
            continue
        parts = name.split("_", 1)
        if len(parts) == 2:
            match = load_matches(match_path)
            pair = (parts[0], parts[1])
            np_matches[pair] = {k: v.cpu().numpy() for k, v in match.items()}

    if not np_images or not np_features:
        return

    from ..visualization.features import render_all_features

    logger.info("%s: rendering feature visualizations", frame_dir.name)
    render_all_features(
        images=np_images,
        all_features=np_features,
        all_matches=np_matches,
        sparse_cloud=None,
        projection_models=None,
        output_dir=viz_dir,
    )


def _load_scene_artifacts(
    frame_dir: Path,
) -> tuple[o3d.geometry.PointCloud | None, o3d.geometry.TriangleMesh | None]:
    """Load point cloud and mesh from disk.

    Args:
        frame_dir: Frame output directory.

    Returns:
        Tuple of (point_cloud, mesh), either may be None if not found.
    """
    pcd = None
    mesh = None

    # Try fused first, then sparse
    for pcd_name in ("fused.ply", "sparse.ply"):
        pcd_path = frame_dir / "point_cloud" / pcd_name
        if pcd_path.exists():
            pcd = o3d.io.read_point_cloud(str(pcd_path))
            break

    mesh_path = frame_dir / "mesh" / "surface.ply"
    if mesh_path.exists():
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    return pcd, mesh


def _viz_scene(
    pcd: o3d.geometry.PointCloud | None,
    mesh: o3d.geometry.TriangleMesh | None,
    viz_dir: Path,
    frame_name: str,
) -> None:
    """Render 3D scene visualizations from loaded artifacts.

    Args:
        pcd: Point cloud (may be None).
        mesh: Triangle mesh (may be None).
        viz_dir: Visualization output directory.
        frame_name: Frame directory name (for logging).
    """
    from ..visualization.scene import render_all_scenes

    logger.info("%s: rendering 3D scene visualizations", frame_name)
    render_all_scenes(
        point_cloud=pcd,
        mesh=mesh,
        output_dir=viz_dir,
    )


def _viz_rig(
    ctx: PipelineContext,
    pcd: o3d.geometry.PointCloud | None,
    viz_dir: Path,
    frame_name: str,
) -> None:
    """Render camera rig diagram with optional point cloud overlay.

    Args:
        ctx: Pipeline context (for calibration data).
        pcd: Optional point cloud for overlay.
        viz_dir: Visualization output directory.
        frame_name: Frame directory name (for logging).
    """
    from ..visualization.rig import render_rig_diagram

    logger.info("%s: rendering rig diagram", frame_name)

    cam_positions = {
        name: pos.cpu().numpy()
        for name, pos in ctx.calibration.camera_positions().items()
    }
    cam_rotations = {
        name: cam.R.cpu().numpy() for name, cam in ctx.calibration.cameras.items()
    }

    pcd_points = None
    if pcd is not None and pcd.has_points():
        pcd_points = np.asarray(pcd.points)

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
