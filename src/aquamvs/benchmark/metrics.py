"""Relative accuracy metrics for benchmark comparison."""

from pathlib import Path

import numpy as np

# Optional Open3D import — used for point cloud loading.
try:
    import open3d as o3d

    _OPEN3D_AVAILABLE = True
except ImportError:
    _OPEN3D_AVAILABLE = False


def compute_relative_metrics(pathway_output_dir: Path) -> dict[str, float]:
    """Compute relative metrics from a pathway's output.

    Looks for ``fused_points.ply`` inside the pathway output directory tree.
    If no cloud is found, all metrics default to 0.

    Args:
        pathway_output_dir: Root output directory for a single pathway run.
            Point clouds are searched recursively inside it.

    Returns:
        Dictionary with keys:

        - ``point_count``: number of points in fused cloud (0 if no cloud)
        - ``cloud_density``: points per m² based on bounding-box XY area
          (0 if no cloud)
        - ``outlier_removal_pct``: percentage of points removed as outliers
          (0 — not computed from file, set by caller if available)
    """
    if not _OPEN3D_AVAILABLE:
        return {"point_count": 0, "cloud_density": 0.0, "outlier_removal_pct": 0.0}

    # Search for fused point cloud in frame subdirectories
    cloud_files = list(pathway_output_dir.rglob("fused_points.ply"))

    if not cloud_files:
        return {"point_count": 0, "cloud_density": 0.0, "outlier_removal_pct": 0.0}

    # Use the first found cloud (there's one per frame in benchmark mode)
    cloud_path = cloud_files[0]
    pcd = o3d.io.read_point_cloud(str(cloud_path))
    points = np.asarray(pcd.points)
    point_count = len(points)

    if point_count == 0:
        return {"point_count": 0, "cloud_density": 0.0, "outlier_removal_pct": 0.0}

    # Compute bounding-box XY area as a scan-area proxy
    bbox = pcd.get_axis_aligned_bounding_box()
    min_bound = np.asarray(bbox.min_bound)
    max_bound = np.asarray(bbox.max_bound)
    xy_extent = max_bound[:2] - min_bound[:2]  # (width, height) in metres
    scan_area = float(xy_extent[0]) * float(xy_extent[1])

    if scan_area > 0:
        cloud_density = point_count / scan_area
    else:
        cloud_density = 0.0

    return {
        "point_count": float(point_count),
        "cloud_density": cloud_density,
        "outlier_removal_pct": 0.0,  # Populated externally if available
    }


__all__ = ["compute_relative_metrics"]
