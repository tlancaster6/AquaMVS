"""Per-config visualization artifacts and cross-config comparison grids."""

import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch

from ..config import SurfaceConfig
from ..pipeline import _sparse_cloud_to_open3d  # noqa: used for benchmark colorized PLY
from ..surface import reconstruct_surface
from ..visualization.features import render_keypoints, render_matches
from ..visualization.scene import render_scene

logger = logging.getLogger(__name__)


def render_config_outputs(
    config_name: str,
    undistorted_images: dict[str, np.ndarray],
    all_features: dict[str, dict[str, torch.Tensor]],
    all_matches: dict[tuple[str, str], dict[str, torch.Tensor]],
    sparse_cloud: dict[str, torch.Tensor],
    projection_models: dict,
    undistorted_tensors: dict[str, torch.Tensor],
    voxel_size: float,
    surface_config: SurfaceConfig,
    output_dir: Path,
    camera_centers: dict[str, np.ndarray],
) -> None:
    """Render per-configuration visual artifacts.

    Creates:
        - keypoints_{cam}.png for each camera
        - matches_{ref}_{src}.png for each matched pair
        - sparse_cloud.ply with colors and normals
        - sparse_top.png, sparse_oblique.png, sparse_side.png (3D renders)
        - mesh.ply with vertex colors and faces
        - mesh_top.png, mesh_oblique.png, mesh_side.png (3D renders)

    Args:
        config_name: Configuration name (e.g., "superpoint_clahe_off").
        undistorted_images: Camera name to BGR uint8 image mapping.
        all_features: Camera name to features dict (torch tensors).
        all_matches: Pair key to matches dict (torch tensors).
        sparse_cloud: Sparse triangulation result with "points_3d" and "scores".
        projection_models: Camera name to ProjectionModel mapping.
        undistorted_tensors: Camera name to image tensor mapping (for coloring).
        voxel_size: Voxel size for downsampling (meters).
        surface_config: Surface reconstruction configuration.
        output_dir: Benchmark root directory (e.g., "output/benchmark").
        camera_centers: Camera centers in world frame, shape (3,) float64 per camera.
    """
    # Create config directory
    config_dir = Path(output_dir) / config_name
    config_dir.mkdir(parents=True, exist_ok=True)

    # Render keypoints for each camera
    logger.info("  Rendering keypoints for config: %s", config_name)
    for cam_name, image in undistorted_images.items():
        if cam_name not in all_features:
            continue

        features = all_features[cam_name]
        keypoints = features["keypoints"].cpu().numpy()  # (N, 2)
        scores = features.get("scores")
        if scores is not None:
            scores = scores.cpu().numpy()  # (N,)

        output_path = config_dir / f"keypoints_{cam_name}.png"
        render_keypoints(image, keypoints, scores, output_path)

    # Render matches for each pair
    logger.info("  Rendering matches for config: %s", config_name)
    for pair_key, match_dict in all_matches.items():
        ref_name, src_name = pair_key
        if ref_name not in undistorted_images or src_name not in undistorted_images:
            continue

        ref_image = undistorted_images[ref_name]
        src_image = undistorted_images[src_name]
        ref_kpts = match_dict["ref_keypoints"].cpu().numpy()  # (M, 2)
        src_kpts = match_dict["src_keypoints"].cpu().numpy()  # (M, 2)
        scores = match_dict.get("scores")
        if scores is not None:
            scores = scores.cpu().numpy()  # (M,)

        output_path = config_dir / f"matches_{ref_name}_{src_name}.png"
        render_matches(ref_image, src_image, ref_kpts, src_kpts, scores, output_path)

    # Handle empty sparse cloud
    n_points = sparse_cloud["points_3d"].shape[0]
    if n_points == 0:
        logger.warning(
            "  Empty sparse cloud for config %s - skipping PLY and renders", config_name
        )
        return

    # Convert sparse cloud to Open3D with colors and normals
    logger.info("  Converting sparse cloud to Open3D for config: %s", config_name)
    pcd = _sparse_cloud_to_open3d(
        sparse_cloud,
        projection_models,
        undistorted_tensors,
        voxel_size,
        camera_centers,
    )

    # Save colored+normal'd PLY
    ply_path = config_dir / "sparse_cloud.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd)
    logger.info("  Saved sparse PLY: %s", ply_path)

    # Render 3D views from canonical viewpoints
    logger.info("  Rendering 3D views for config: %s", config_name)
    render_scene(pcd, config_dir, prefix="sparse")

    # Reconstruct surface mesh
    logger.info("  Reconstructing surface mesh for config: %s", config_name)
    try:
        mesh = reconstruct_surface(pcd, surface_config)

        # Save mesh PLY
        mesh_path = config_dir / "mesh.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        logger.info("  Saved mesh PLY: %s", mesh_path)

        # Render mesh from canonical viewpoints
        logger.info("  Rendering mesh views for config: %s", config_name)
        render_scene(mesh, config_dir, prefix="mesh")
    except Exception as e:
        logger.warning("  Mesh reconstruction failed for config %s: %s", config_name, e)


def render_comparison_grids(
    config_names: list[str],
    camera_names: list[str],
    output_dir: Path,
) -> None:
    """Render cross-config comparison grids.

    Creates:
        - comparison/keypoints_grid.png (rows=configs, cols=cameras)
        - comparison/sparse_renders_grid.png (rows=configs, cols=viewpoints)
        - comparison/mesh_grid.png (rows=configs, cols=viewpoints)

    Missing images are handled gracefully (blank/gray cells).

    Args:
        config_names: List of configuration names.
        camera_names: List of camera names.
        output_dir: Benchmark root directory (e.g., "output/benchmark").
    """
    comparison_dir = Path(output_dir) / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # --- Keypoint grid: rows=configs, cols=cameras ---
    logger.info("Rendering keypoint comparison grid")
    n_rows = len(config_names)
    n_cols = len(camera_names)

    if n_rows == 0 or n_cols == 0:
        logger.warning("No configs or cameras to render in keypoint grid")
    else:
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i, config_name in enumerate(config_names):
            for j, cam_name in enumerate(camera_names):
                ax = axes[i, j]
                image_path = output_dir / config_name / f"keypoints_{cam_name}.png"

                if image_path.exists():
                    img = cv2.imread(str(image_path))
                    if img is not None:
                        # Convert BGR to RGB for matplotlib
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        ax.imshow(img)
                else:
                    # Missing image - show gray
                    ax.set_facecolor("lightgray")

                ax.axis("off")

                # Add row and column labels
                if j == 0:
                    ax.set_ylabel(config_name, fontsize=10, rotation=90, labelpad=10)
                if i == 0:
                    ax.set_title(cam_name, fontsize=10)

        plt.tight_layout()
        grid_path = comparison_dir / "keypoints_grid.png"
        plt.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved keypoint grid: %s", grid_path)

    # --- Sparse renders grid: rows=configs, cols=viewpoints ---
    logger.info("Rendering sparse 3D comparison grid")
    viewpoints = ["top", "oblique", "side"]
    n_rows = len(config_names)
    n_cols = len(viewpoints)

    if n_rows == 0 or n_cols == 0:
        logger.warning("No configs or viewpoints to render in sparse grid")
    else:
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i, config_name in enumerate(config_names):
            for j, viewpoint in enumerate(viewpoints):
                ax = axes[i, j]
                image_path = output_dir / config_name / f"sparse_{viewpoint}.png"

                if image_path.exists():
                    img = plt.imread(str(image_path))  # PNG, already RGB
                    ax.imshow(img)
                else:
                    # Missing image - show gray
                    ax.set_facecolor("lightgray")

                ax.axis("off")

                # Add row and column labels
                if j == 0:
                    ax.set_ylabel(config_name, fontsize=10, rotation=90, labelpad=10)
                if i == 0:
                    ax.set_title(viewpoint.capitalize(), fontsize=10)

        plt.tight_layout()
        grid_path = comparison_dir / "sparse_renders_grid.png"
        plt.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved sparse renders grid: %s", grid_path)

    # --- Mesh renders grid: rows=configs, cols=viewpoints ---
    logger.info("Rendering mesh comparison grid")
    n_rows = len(config_names)
    n_cols = len(viewpoints)

    if n_rows == 0 or n_cols == 0:
        logger.warning("No configs or viewpoints to render in mesh grid")
    else:
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i, config_name in enumerate(config_names):
            for j, viewpoint in enumerate(viewpoints):
                ax = axes[i, j]
                image_path = output_dir / config_name / f"mesh_{viewpoint}.png"

                if image_path.exists():
                    img = plt.imread(str(image_path))  # PNG, already RGB
                    ax.imshow(img)
                else:
                    # Missing image - show gray
                    ax.set_facecolor("lightgray")

                ax.axis("off")

                # Add row and column labels
                if j == 0:
                    ax.set_ylabel(config_name, fontsize=10, rotation=90, labelpad=10)
                if i == 0:
                    ax.set_title(viewpoint.capitalize(), fontsize=10)

        plt.tight_layout()
        grid_path = comparison_dir / "mesh_grid.png"
        plt.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved mesh renders grid: %s", grid_path)


__all__ = ["render_config_outputs", "render_comparison_grids"]
