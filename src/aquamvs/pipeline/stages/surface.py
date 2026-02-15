"""Surface reconstruction and visualization stage."""

import logging
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from torch.profiler import record_function

from ...fusion import save_point_cloud
from ...surface import reconstruct_surface, save_mesh
from ..context import PipelineContext
from ..helpers import _should_viz, _sparse_cloud_to_open3d

logger = logging.getLogger(__name__)


def run_surface_stage(
    fused_pcd: o3d.geometry.PointCloud,
    undistorted_tensors: dict[str, torch.Tensor],
    camera_centers: dict[str, np.ndarray],
    ctx: PipelineContext,
    frame_dir: Path,
    frame_idx: int,
) -> o3d.geometry.TriangleMesh | None:
    """Run surface reconstruction, mesh coloring, save, and visualizations.

    Args:
        fused_pcd: Fused point cloud (after outlier removal).
        undistorted_tensors: Dict of undistorted BGR images as torch.Tensor.
        camera_centers: Dict of camera positions in world frame, shape (3,) float64.
        ctx: Pipeline context.
        frame_dir: Frame output directory.
        frame_idx: Frame index (for logging).

    Returns:
        Reconstructed mesh or None if point cloud is empty.
    """
    with record_function("surface_reconstruction"):
        config = ctx.config

        # --- Stage 9: Surface Reconstruction ---
        if fused_pcd.has_points():
            logger.info("Frame %d: reconstructing surface", frame_idx)
            mesh = reconstruct_surface(fused_pcd, config.reconstruction)

            # Re-color mesh vertices via best-view camera projection
            if mesh is not None and len(mesh.vertices) > 0:
                from ...coloring import best_view_colors

                # Filter to ring cameras only (exclude auxiliary cameras)
                ring_models = {
                    n: m
                    for n, m in ctx.projection_models.items()
                    if n in ctx.ring_cameras
                }
                ring_images = {
                    n: img
                    for n, img in undistorted_tensors.items()
                    if n in ctx.ring_cameras
                }
                ring_centers = {
                    n: c for n, c in camera_centers.items() if n in ctx.ring_cameras
                }

                mesh.compute_vertex_normals()
                vertex_colors = best_view_colors(
                    np.asarray(mesh.vertices),
                    np.asarray(mesh.vertex_normals),
                    ring_models,
                    ring_images,
                    ring_centers,
                )
                mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

            # --- Save mesh (opt-out) ---
            if config.runtime.save_mesh:
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
                from ...visualization.scene import render_all_scenes

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
                from ...visualization.rig import render_rig_diagram

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
                    pcd_points = np.asarray(fused_pcd.points)

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
        return mesh


def run_sparse_surface_stage(
    sparse_cloud: dict[str, torch.Tensor],
    undistorted_tensors: dict[str, torch.Tensor],
    camera_centers: dict[str, np.ndarray],
    ctx: PipelineContext,
    frame_dir: Path,
    frame_idx: int,
) -> None:
    """Run sparse mode surface reconstruction (convert sparse cloud to Open3D, surface, viz).

    This is the early-return path for sparse mode after triangulation.

    Args:
        sparse_cloud: Dict with "points_3d" and "scores" tensors.
        undistorted_tensors: Dict of undistorted BGR images as torch.Tensor.
        camera_centers: Dict of camera positions in world frame, shape (3,) float64.
        ctx: Pipeline context.
        frame_dir: Frame output directory.
        frame_idx: Frame index (for logging).
    """
    with record_function("surface_reconstruction"):
        config = ctx.config

        # Convert sparse cloud to Open3D PointCloud
        pcd = None
        if sparse_cloud["points_3d"].shape[0] > 0:
            pcd = _sparse_cloud_to_open3d(
                sparse_cloud,
                ctx.projection_models,
                undistorted_tensors,
                config.reconstruction.voxel_size,
                camera_centers,
                ctx.ring_cameras,
            )
        else:
            logger.warning(
                "Frame %d: sparse cloud is empty, skipping surface reconstruction",
                frame_idx,
            )

        # Save point cloud (if non-empty and save_point_cloud is enabled)
        if pcd is not None and pcd.has_points() and config.runtime.save_point_cloud:
            pcd_dir = frame_dir / "point_cloud"
            pcd_dir.mkdir(exist_ok=True)
            save_point_cloud(pcd, pcd_dir / "sparse.ply")

        # Statistical outlier removal (sparse path)
        # Skip if too few points for meaningful neighbor statistics
        if (
            config.reconstruction.outlier_removal_enabled
            and pcd is not None
            and pcd.has_points()
            and len(pcd.points) > config.reconstruction.outlier_nb_neighbors
        ):
            original_count = len(pcd.points)
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=config.reconstruction.outlier_nb_neighbors,
                std_ratio=config.reconstruction.outlier_std_ratio,
            )
            removed = original_count - len(pcd.points)
            logger.info(
                "Frame %d: removed %d outliers (%.1f%%) from sparse cloud",
                frame_idx,
                removed,
                removed / original_count * 100 if original_count > 0 else 0.0,
            )

        # Surface reconstruction (if non-empty)
        mesh = None
        if pcd is not None and pcd.has_points():
            logger.info("Frame %d: reconstructing surface", frame_idx)
            mesh = reconstruct_surface(pcd, config.reconstruction)

            # Re-color mesh vertices via best-view camera projection
            if mesh is not None and len(mesh.vertices) > 0:
                from ...coloring import best_view_colors

                # Filter to ring cameras only (exclude auxiliary cameras)
                ring_models = {
                    n: m
                    for n, m in ctx.projection_models.items()
                    if n in ctx.ring_cameras
                }
                ring_images = {
                    n: img
                    for n, img in undistorted_tensors.items()
                    if n in ctx.ring_cameras
                }
                ring_centers = {
                    n: c for n, c in camera_centers.items() if n in ctx.ring_cameras
                }

                mesh.compute_vertex_normals()
                vertex_colors = best_view_colors(
                    np.asarray(mesh.vertices),
                    np.asarray(mesh.vertex_normals),
                    ring_models,
                    ring_images,
                    ring_centers,
                )
                mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

            # Save mesh (opt-out)
            if config.runtime.save_mesh:
                mesh_dir = frame_dir / "mesh"
                mesh_dir.mkdir(exist_ok=True)
                save_mesh(mesh, mesh_dir / "surface.ply")

        # [viz] 3D scene renders
        if _should_viz(config, "scene") and pcd is not None:
            try:
                from ...visualization.scene import render_all_scenes

                logger.info("Frame %d: rendering 3D scene visualizations", frame_idx)
                viz_dir = frame_dir / "viz"
                viz_dir.mkdir(exist_ok=True)

                render_all_scenes(
                    point_cloud=pcd,
                    mesh=mesh,
                    output_dir=viz_dir,
                )
            except Exception:
                logger.exception("Frame %d: scene visualization failed", frame_idx)

        # [viz] Camera rig diagram
        if _should_viz(config, "rig"):
            try:
                from ...visualization.rig import render_rig_diagram

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
                if pcd is not None and pcd.has_points():
                    pcd_points = np.asarray(pcd.points)

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

        logger.info("Frame %d: complete (sparse mode)", frame_idx)
