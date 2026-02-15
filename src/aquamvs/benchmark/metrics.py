"""Benchmark metrics and result data structures."""

import logging
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ============================================================================
# Legacy feature extraction benchmark structures (backward compatibility)
# ============================================================================


@dataclass
class ConfigResult:
    """Results from one configuration in the benchmark sweep."""

    config_name: str  # e.g., "superpoint_clahe_on"
    extractor_type: str
    clahe_enabled: bool

    # Per-camera keypoint counts and mean scores
    keypoint_counts: dict[str, int]
    keypoint_mean_scores: dict[str, float]

    # Per-pair match counts
    match_counts: dict[tuple[str, str], int]

    # Sparse cloud stats
    sparse_point_count: int

    # Timing (seconds)
    extraction_time: float
    matching_time: float
    triangulation_time: float
    total_time: float


@dataclass
class BenchmarkResults:
    """Aggregated results from the full benchmark sweep."""

    results: list[ConfigResult]
    frame_idx: int
    camera_names: list[str]  # ring + auxiliary
    pair_keys: list[tuple[str, str]]


def config_name(extractor_type: str, clahe_enabled: bool) -> str:
    """Generate a human-readable config name like 'superpoint_clahe_on'.

    Args:
        extractor_type: Feature extractor backend name.
        clahe_enabled: Whether CLAHE preprocessing is enabled.

    Returns:
        Config name string.
    """
    clahe_suffix = "clahe_on" if clahe_enabled else "clahe_off"
    return f"{extractor_type}_{clahe_suffix}"


def total_keypoints(result: ConfigResult) -> int:
    """Sum keypoints across all cameras.

    Args:
        result: Configuration result to aggregate.

    Returns:
        Total keypoint count.
    """
    return sum(result.keypoint_counts.values())


def total_matches(result: ConfigResult) -> int:
    """Sum matches across all pairs.

    Args:
        result: Configuration result to aggregate.

    Returns:
        Total match count.
    """
    return sum(result.match_counts.values())


# ============================================================================
# New accuracy metrics for reconstruction quality evaluation
# ============================================================================


def compute_accuracy_metrics(
    reconstructed_points: NDArray[np.float64],
    ground_truth_mesh: o3d.geometry.TriangleMesh,
    tolerance_mm: float | None = None,
) -> dict[str, float]:
    """Compute accuracy metrics comparing reconstructed points to ground truth mesh.

    Args:
        reconstructed_points: N × 3 array of reconstructed 3D points in world coordinates.
        ground_truth_mesh: Ground truth surface as triangle mesh.
        tolerance_mm: Tolerance for accurate completeness metric (None = skip).

    Returns:
        Dictionary with metrics:
            - mean_error_mm: Mean distance to ground truth surface
            - median_error_mm: Median distance to ground truth surface
            - std_error_mm: Standard deviation of distance
            - raw_completeness_pct: Percentage of reconstructed points (vs expected)
            - accurate_completeness_pct: Percentage within tolerance (if tolerance provided)
    """
    if len(reconstructed_points) == 0:
        logger.warning("No reconstructed points for accuracy metrics")
        return {
            "mean_error_mm": float("nan"),
            "median_error_mm": float("nan"),
            "std_error_mm": float("nan"),
            "raw_completeness_pct": 0.0,
        }

    # Create point cloud from reconstructed points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(reconstructed_points)

    # Compute point-to-surface distances
    distances = np.asarray(
        pcd.compute_point_cloud_distance(ground_truth_mesh)
    )  # distances in meters

    # Convert to millimeters
    distances_mm = distances * 1000.0

    # Compute geometric error metrics
    mean_error = float(np.mean(distances_mm))
    median_error = float(np.median(distances_mm))
    std_error = float(np.std(distances_mm))

    # Raw completeness: fraction of expected points reconstructed
    # (Using mesh surface area as proxy for expected point count)
    mesh_area = ground_truth_mesh.get_surface_area()
    expected_points = mesh_area * 1e6  # ~1 point per mm^2 as baseline
    raw_completeness = (len(reconstructed_points) / expected_points) * 100.0

    metrics = {
        "mean_error_mm": mean_error,
        "median_error_mm": median_error,
        "std_error_mm": std_error,
        "raw_completeness_pct": raw_completeness,
    }

    # Tolerance-based accurate completeness (optional)
    if tolerance_mm is not None:
        within_tolerance = np.sum(distances_mm <= tolerance_mm)
        accurate_completeness = (within_tolerance / len(reconstructed_points)) * 100.0
        metrics["accurate_completeness_pct"] = accurate_completeness

    return metrics


def compute_charuco_metrics(
    detected_corners: NDArray[np.float64],
    projected_corners: NDArray[np.float64],
) -> dict[str, float]:
    """Compute point-level error metrics at ChArUco corner positions.

    Args:
        detected_corners: N × 3 array of detected corner 3D positions.
        projected_corners: N × 3 array of projected corner positions from reconstruction.

    Returns:
        Dictionary with metrics:
            - mean_error_mm: Mean corner reprojection error
            - median_error_mm: Median corner reprojection error
            - max_error_mm: Maximum corner reprojection error
            - rmse_mm: Root mean squared error
    """
    if len(detected_corners) == 0 or len(projected_corners) == 0:
        logger.warning("No corners for ChArUco metrics")
        return {
            "mean_error_mm": float("nan"),
            "median_error_mm": float("nan"),
            "max_error_mm": float("nan"),
            "rmse_mm": float("nan"),
        }

    # Compute per-corner Euclidean distance
    errors = np.linalg.norm(detected_corners - projected_corners, axis=1)
    errors_mm = errors * 1000.0

    return {
        "mean_error_mm": float(np.mean(errors_mm)),
        "median_error_mm": float(np.median(errors_mm)),
        "max_error_mm": float(np.max(errors_mm)),
        "rmse_mm": float(np.sqrt(np.mean(errors_mm**2))),
    }


def compute_plane_fit_metrics(points: NDArray[np.float64]) -> dict[str, float]:
    """Compute plane fitting metrics for overall shape/scale accuracy.

    Fits a plane to the point cloud and measures deviation from planarity.
    Useful for ChArUco board evaluation and synthetic flat plane validation.

    Args:
        points: N × 3 array of 3D points.

    Returns:
        Dictionary with metrics:
            - plane_fit_rmse_mm: RMS deviation from fitted plane
            - plane_fit_max_error_mm: Maximum deviation from fitted plane
            - plane_normal: Fitted plane normal vector (as tuple)
    """
    if len(points) < 3:
        logger.warning("Insufficient points for plane fitting")
        return {
            "plane_fit_rmse_mm": float("nan"),
            "plane_fit_max_error_mm": float("nan"),
            "plane_normal": (float("nan"), float("nan"), float("nan")),
        }

    # Center the points
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # SVD to find best-fit plane normal (smallest singular value direction)
    _, _, vh = np.linalg.svd(centered)
    normal = vh[2, :]  # Last row = normal direction

    # Compute signed distances to plane
    distances = np.dot(centered, normal)  # meters
    distances_mm = distances * 1000.0

    rmse = float(np.sqrt(np.mean(distances_mm**2)))
    max_error = float(np.max(np.abs(distances_mm)))

    return {
        "plane_fit_rmse_mm": rmse,
        "plane_fit_max_error_mm": max_error,
        "plane_normal": tuple(float(x) for x in normal),
    }


__all__ = [
    # Legacy feature extraction benchmark
    "ConfigResult",
    "BenchmarkResults",
    "config_name",
    "total_keypoints",
    "total_matches",
    # New accuracy metrics
    "compute_accuracy_metrics",
    "compute_charuco_metrics",
    "compute_plane_fit_metrics",
]
