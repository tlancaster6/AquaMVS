"""Helper functions for pipeline operations."""

import logging
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from matplotlib import cm
from scipy.interpolate import griddata

from ..config import PipelineConfig
from ..projection.protocol import ProjectionModel

logger = logging.getLogger(__name__)


def _should_viz(config: PipelineConfig, stage: str) -> bool:
    """Check whether a visualization stage should run.

    Args:
        config: Pipeline configuration.
        stage: Viz stage name (one of VALID_VIZ_STAGES).

    Returns:
        True if viz is enabled and this stage should run.
    """
    if not config.runtime.viz_enabled:
        return False
    # Empty stages list = all stages
    if not config.runtime.viz_stages:
        return True
    return stage in config.runtime.viz_stages


def _save_consistency_map(
    consistency: torch.Tensor,
    output_stem: Path,
    max_value: int,
) -> None:
    """Save consistency map as NPZ and colormapped PNG.

    Args:
        consistency: Consistency count map (H, W), int32.
        output_stem: Output path stem (without extension).
        max_value: Maximum consistency value for normalization
            (number of source cameras for this reference view).
    """
    consistency_np = consistency.cpu().numpy()

    # NPZ for programmatic analysis
    np.savez_compressed(
        str(output_stem.with_suffix(".npz")),
        consistency=consistency_np,
    )

    # Colormapped PNG for visual inspection
    # Normalize by number of source cameras, not per-frame max
    cmap = cm.get_cmap("viridis")
    normalized = consistency_np.astype(float) / max(max_value, 1)
    colored = cmap(normalized)  # (H, W, 4) RGBA float [0, 1]
    colored_bgr = (colored[:, :, :3][:, :, ::-1] * 255).astype(np.uint8)
    cv2.imwrite(str(output_stem.with_suffix(".png")), colored_bgr)


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
    output_dir = Path(config.output_dir)
    height_maps = []

    for frame_dir in sorted(output_dir.glob("frame_*")):
        pcd_path = frame_dir / "point_cloud" / "fused.ply"
        if not pcd_path.exists():
            continue

        try:
            pcd = o3d.io.read_point_cloud(str(pcd_path))
            if not pcd.has_points():
                continue

            pts = np.asarray(pcd.points)
            frame_idx = int(frame_dir.name.split("_")[1])

            # Grid the points
            resolution = config.reconstruction.grid_resolution
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


def _sparse_cloud_to_open3d(
    sparse_cloud: dict[str, torch.Tensor],
    projection_models: dict[str, ProjectionModel],
    images: dict[str, torch.Tensor],
    voxel_size: float,
    camera_centers: dict[str, np.ndarray],
    ring_cameras: list[str],
) -> o3d.geometry.PointCloud:
    """Convert sparse triangulated cloud to Open3D PointCloud with colors and normals.

    Builds point cloud, downsamples, estimates normals, then assigns colors using
    the best viewing angle per point (most perpendicular to local surface normal).

    Args:
        sparse_cloud: Dict with "points_3d" (N, 3) and "scores" (N,) tensors.
        projection_models: Dict of ProjectionModel by camera name.
        images: Dict of undistorted BGR images (H, W, 3) uint8 tensors by camera name.
        voxel_size: Voxel size for downsampling (meters).
        camera_centers: Dict of camera centers in world frame, shape (3,) float64 per camera.
        ring_cameras: List of ring (non-auxiliary) camera names for color selection.

    Returns:
        Open3D PointCloud with points, colors, and normals.
    """
    from ..coloring import best_view_colors

    points_3d = sparse_cloud["points_3d"]  # (N, 3)
    N = points_3d.shape[0]

    if N == 0:
        return o3d.geometry.PointCloud()

    # 1. Build uncolored point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d.cpu().numpy())

    # 2. Voxel downsample (reduces N significantly)
    if N > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    # 3. Estimate normals (needed for best-view selection)
    if pcd.has_points():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 5, max_nn=30
            )
        )
        # Orient normals toward camera origin (0, 0, 0)
        pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))

    # 4. Assign colors using best-view selection on DOWNSAMPLED points
    if pcd.has_points():
        # Filter to ring cameras only (exclude auxiliary cameras)
        ring_models = {n: m for n, m in projection_models.items() if n in ring_cameras}
        ring_images = {n: img for n, img in images.items() if n in ring_cameras}
        ring_centers = {n: c for n, c in camera_centers.items() if n in ring_cameras}

        colors = best_view_colors(
            np.asarray(pcd.points),
            np.asarray(pcd.normals),
            ring_models,
            ring_images,
            ring_centers,
        )
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd
