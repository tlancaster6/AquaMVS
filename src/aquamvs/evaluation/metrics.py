"""Metrics for surface reconstruction quality."""

from typing import Any

import numpy as np
import open3d as o3d
import torch
from scipy.interpolate import griddata

from aquamvs.projection.protocol import ProjectionModel


def cloud_to_cloud_distance(
    cloud_a: o3d.geometry.PointCloud,
    cloud_b: o3d.geometry.PointCloud,
) -> dict[str, float]:
    """Compute distance metrics between two point clouds.

    For each point in cloud_a, finds the nearest neighbor in cloud_b
    and vice versa. Reports symmetric metrics.

    Args:
        cloud_a: First point cloud.
        cloud_b: Second point cloud.

    Returns:
        Dict with keys:
            "mean_a_to_b": float -- mean distance from A to nearest in B (meters)
            "mean_b_to_a": float -- mean distance from B to nearest in A (meters)
            "hausdorff_a_to_b": float -- max distance from A to nearest in B (meters)
            "hausdorff_b_to_a": float -- max distance from B to nearest in A (meters)
            "hausdorff": float -- symmetric Hausdorff = max of the two directed values
            "mean": float -- symmetric mean = average of the two directed means
            "median_a_to_b": float -- median distance from A to B
            "median_b_to_a": float -- median distance from B to A
    """
    # A -> B distances
    dists_a = np.asarray(cloud_a.compute_point_cloud_distance(cloud_b))
    # B -> A distances
    dists_b = np.asarray(cloud_b.compute_point_cloud_distance(cloud_a))

    return {
        "mean_a_to_b": float(dists_a.mean()),
        "mean_b_to_a": float(dists_b.mean()),
        "hausdorff_a_to_b": float(dists_a.max()),
        "hausdorff_b_to_a": float(dists_b.max()),
        "hausdorff": float(max(dists_a.max(), dists_b.max())),
        "mean": float((dists_a.mean() + dists_b.mean()) / 2),
        "median_a_to_b": float(np.median(dists_a)),
        "median_b_to_a": float(np.median(dists_b)),
    }


def height_map_difference(
    cloud_a: o3d.geometry.PointCloud,
    cloud_b: o3d.geometry.PointCloud,
    grid_resolution: float = 0.005,
) -> dict[str, Any]:
    """Compute height-map difference between two point clouds.

    Projects both clouds onto a shared XY grid, interpolates Z values,
    and computes per-cell differences. Useful for comparing approximately
    planar surfaces (e.g., sand bed reconstructions).

    Args:
        cloud_a: First point cloud.
        cloud_b: Second point cloud.
        grid_resolution: XY grid cell size in meters.

    Returns:
        Dict with keys:
            "mean_diff": float -- mean signed Z difference (A - B), meters
            "std_diff": float -- std of Z differences, meters
            "abs_mean_diff": float -- mean absolute Z difference, meters
            "max_abs_diff": float -- max absolute Z difference, meters
            "rmse": float -- root mean square of Z differences, meters
            "diff_map": ndarray (Ny, Nx) -- per-cell Z difference (A - B),
                NaN where either cloud has no data
            "grid_x": ndarray (Nx,) -- X coordinates of grid columns
            "grid_y": ndarray (Ny,) -- Y coordinates of grid rows
    """
    pts_a = np.asarray(cloud_a.points)
    pts_b = np.asarray(cloud_b.points)

    # Shared grid covering both clouds
    all_pts = np.vstack([pts_a, pts_b])
    x_min, y_min = all_pts[:, 0].min(), all_pts[:, 1].min()
    x_max, y_max = all_pts[:, 0].max(), all_pts[:, 1].max()

    grid_x = np.arange(x_min, x_max + grid_resolution, grid_resolution)
    grid_y = np.arange(y_min, y_max + grid_resolution, grid_resolution)
    gx, gy = np.meshgrid(grid_x, grid_y)

    # Interpolate Z for each cloud
    z_a = griddata(pts_a[:, :2], pts_a[:, 2], (gx, gy), method="linear", fill_value=np.nan)
    z_b = griddata(pts_b[:, :2], pts_b[:, 2], (gx, gy), method="linear", fill_value=np.nan)

    # Difference (only where both clouds have data)
    diff = z_a - z_b
    valid = ~np.isnan(diff)

    if not valid.any():
        return {
            "mean_diff": float("nan"),
            "std_diff": float("nan"),
            "abs_mean_diff": float("nan"),
            "max_abs_diff": float("nan"),
            "rmse": float("nan"),
            "diff_map": diff,
            "grid_x": grid_x,
            "grid_y": grid_y,
        }

    valid_diffs = diff[valid]

    return {
        "mean_diff": float(valid_diffs.mean()),
        "std_diff": float(valid_diffs.std()),
        "abs_mean_diff": float(np.abs(valid_diffs).mean()),
        "max_abs_diff": float(np.abs(valid_diffs).max()),
        "rmse": float(np.sqrt((valid_diffs**2).mean())),
        "diff_map": diff,
        "grid_x": grid_x,
        "grid_y": grid_y,
    }


def reprojection_error(
    points_3d: torch.Tensor,
    observations: dict[str, torch.Tensor],
    projection_models: dict[str, ProjectionModel],
) -> dict[str, Any]:
    """Compute reprojection error of sparse 3D points.

    For each 3D point, projects it through each camera's projection model
    and compares to the observed 2D pixel coordinate.

    Args:
        points_3d: Triangulated 3D points, shape (N, 3), float32.
        observations: Camera name to observed pixel coords mapping.
            Each value is shape (N, 2), float32. NaN for unobserved points.
        projection_models: Camera name to ProjectionModel mapping.

    Returns:
        Dict with keys:
            "mean_error": float -- mean reprojection error in pixels
            "median_error": float -- median reprojection error in pixels
            "max_error": float -- max reprojection error in pixels
            "per_camera": dict[str, float] -- mean error per camera
            "errors": ndarray -- all individual errors (K,) in pixels
    """
    all_errors = []
    per_camera = {}

    for cam_name, obs_pixels in observations.items():
        if cam_name not in projection_models:
            continue

        model = projection_models[cam_name]

        # Find observed (non-NaN) points for this camera
        valid_obs = ~torch.isnan(obs_pixels[:, 0])
        if not valid_obs.any():
            continue

        valid_points = points_3d[valid_obs]
        valid_pixels = obs_pixels[valid_obs]

        # Project 3D points
        projected, proj_valid = model.project(valid_points)

        # Compute pixel error only where projection is valid
        both_valid = proj_valid
        if not both_valid.any():
            continue

        errors = torch.linalg.norm(
            projected[both_valid] - valid_pixels[both_valid], dim=-1
        )  # (K,) in pixels

        per_camera[cam_name] = float(errors.mean())
        all_errors.append(errors)

    if not all_errors:
        return {
            "mean_error": float("nan"),
            "median_error": float("nan"),
            "max_error": float("nan"),
            "per_camera": {},
            "errors": np.array([]),
        }

    all_errors_cat = torch.cat(all_errors)
    errors_np = all_errors_cat.cpu().numpy()

    return {
        "mean_error": float(errors_np.mean()),
        "median_error": float(np.median(errors_np)),
        "max_error": float(errors_np.max()),
        "per_camera": per_camera,
        "errors": errors_np,
    }
