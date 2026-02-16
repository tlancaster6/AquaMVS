"""Surface reconstruction stage."""

import logging
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from torch.profiler import record_function

from ...fusion import save_point_cloud
from ...surface import reconstruct_surface, save_mesh
from ..context import PipelineContext
from ..helpers import _sparse_cloud_to_open3d

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

            # --- Save mesh (always — viz pass reloads from disk) ---
            mesh_dir = frame_dir / "mesh"
            mesh_dir.mkdir(exist_ok=True)
            save_mesh(mesh, mesh_dir / "surface.ply")
        else:
            logger.warning(
                "Frame %d: fused point cloud is empty, skipping surface reconstruction",
                frame_idx,
            )
            mesh = None

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

        # Save point cloud (always — viz pass reloads from disk)
        if pcd is not None and pcd.has_points():
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

            # Save mesh (always — viz pass reloads from disk)
            mesh_dir = frame_dir / "mesh"
            mesh_dir.mkdir(exist_ok=True)
            save_mesh(mesh, mesh_dir / "surface.ply")

        logger.info("Frame %d: complete (sparse mode)", frame_idx)
