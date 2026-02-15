"""Tests for benchmark metrics and helper functions."""

import numpy as np
import open3d as o3d
import pytest

from aquamvs.benchmark.metrics import (
    ConfigResult,
    compute_accuracy_metrics,
    compute_charuco_metrics,
    compute_plane_fit_metrics,
    config_name,
    total_keypoints,
    total_matches,
)


def test_config_name():
    """Test config_name generates expected strings."""
    assert config_name("superpoint", True) == "superpoint_clahe_on"
    assert config_name("superpoint", False) == "superpoint_clahe_off"
    assert config_name("aliked", True) == "aliked_clahe_on"
    assert config_name("aliked", False) == "aliked_clahe_off"
    assert config_name("disk", True) == "disk_clahe_on"
    assert config_name("disk", False) == "disk_clahe_off"


def test_total_keypoints():
    """Test total_keypoints sums across cameras."""
    result = ConfigResult(
        config_name="test",
        extractor_type="superpoint",
        clahe_enabled=False,
        keypoint_counts={"cam0": 100, "cam1": 200, "cam2": 50},
        keypoint_mean_scores={"cam0": 0.5, "cam1": 0.6, "cam2": 0.4},
        match_counts={},
        sparse_point_count=0,
        extraction_time=0.0,
        matching_time=0.0,
        triangulation_time=0.0,
        total_time=0.0,
    )

    assert total_keypoints(result) == 350


def test_total_matches():
    """Test total_matches sums across pairs."""
    result = ConfigResult(
        config_name="test",
        extractor_type="superpoint",
        clahe_enabled=False,
        keypoint_counts={},
        keypoint_mean_scores={},
        match_counts={
            ("cam0", "cam1"): 50,
            ("cam0", "cam2"): 30,
            ("cam1", "cam2"): 40,
        },
        sparse_point_count=0,
        extraction_time=0.0,
        matching_time=0.0,
        triangulation_time=0.0,
        total_time=0.0,
    )

    assert total_matches(result) == 120


# ============================================================================
# Tests for new accuracy metrics
# ============================================================================


def test_compute_accuracy_metrics_perfect_match():
    """Test accuracy metrics with perfect reconstruction (zero error)."""
    # Create a simple flat plane mesh
    mesh = o3d.geometry.TriangleMesh()
    vertices = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Sample points exactly on the mesh surface
    points = np.array(
        [
            [0.25, 0.25, 1.0],
            [0.5, 0.5, 1.0],
            [0.75, 0.75, 1.0],
        ]
    )

    metrics = compute_accuracy_metrics(points, mesh, tolerance_mm=1.0)

    # Perfect match should have near-zero error
    assert metrics["mean_error_mm"] < 1e-6
    assert metrics["median_error_mm"] < 1e-6
    assert metrics["std_error_mm"] < 1e-6
    assert metrics["accurate_completeness_pct"] == pytest.approx(100.0, abs=1e-6)


def test_compute_accuracy_metrics_offset():
    """Test accuracy metrics with known offset from ground truth."""
    # Create a flat plane mesh at z=1.0
    mesh = o3d.geometry.TriangleMesh()
    vertices = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Points offset by 5mm (0.005m) from surface
    offset = 0.005  # 5mm
    points = np.array(
        [
            [0.25, 0.25, 1.0 + offset],
            [0.5, 0.5, 1.0 + offset],
            [0.75, 0.75, 1.0 + offset],
        ]
    )

    metrics = compute_accuracy_metrics(points, mesh, tolerance_mm=10.0)

    # Should measure 5mm error
    assert metrics["mean_error_mm"] == pytest.approx(5.0, abs=0.1)
    assert metrics["median_error_mm"] == pytest.approx(5.0, abs=0.1)
    assert metrics["std_error_mm"] < 0.1  # uniform offset = low std
    # All points within 10mm tolerance
    assert metrics["accurate_completeness_pct"] == pytest.approx(100.0, abs=1e-6)


def test_compute_accuracy_metrics_tolerance():
    """Test tolerance-based accurate completeness metric."""
    # Create a flat plane mesh at z=1.0
    mesh = o3d.geometry.TriangleMesh()
    vertices = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Mix of points within and outside tolerance
    points = np.array(
        [
            [0.25, 0.25, 1.001],  # 1mm error - within 5mm tolerance
            [0.5, 0.5, 1.003],  # 3mm error - within 5mm tolerance
            [0.75, 0.75, 1.010],  # 10mm error - outside 5mm tolerance
        ]
    )

    metrics = compute_accuracy_metrics(points, mesh, tolerance_mm=5.0)

    # 2 out of 3 points within tolerance
    assert metrics["accurate_completeness_pct"] == pytest.approx(66.67, abs=0.1)


def test_compute_accuracy_metrics_no_tolerance():
    """Test that accurate completeness is skipped when tolerance not provided."""
    # Create a simple mesh
    mesh = o3d.geometry.TriangleMesh()
    vertices = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.5, 1.0, 1.0],
        ]
    )
    triangles = np.array([[0, 1, 2]])
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    points = np.array([[0.5, 0.5, 1.0]])

    metrics = compute_accuracy_metrics(points, mesh, tolerance_mm=None)

    # Should have basic metrics but not accurate_completeness_pct
    assert "mean_error_mm" in metrics
    assert "median_error_mm" in metrics
    assert "std_error_mm" in metrics
    assert "raw_completeness_pct" in metrics
    assert "accurate_completeness_pct" not in metrics


def test_compute_accuracy_metrics_empty_points():
    """Test graceful handling of empty point cloud."""
    mesh = o3d.geometry.TriangleMesh()
    vertices = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.5, 1.0, 1.0]])
    triangles = np.array([[0, 1, 2]])
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    points = np.array([]).reshape(0, 3)

    metrics = compute_accuracy_metrics(points, mesh)

    # Should return NaN/zero for empty input
    assert np.isnan(metrics["mean_error_mm"])
    assert np.isnan(metrics["median_error_mm"])
    assert np.isnan(metrics["std_error_mm"])
    assert metrics["raw_completeness_pct"] == 0.0


def test_compute_charuco_metrics():
    """Test ChArUco corner reprojection error metrics."""
    # Ground truth corners
    detected = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.0, 0.1, 1.0],
        ]
    )

    # Projected corners with known errors
    projected = np.array(
        [
            [0.001, 0.0, 1.0],  # 1mm error in X
            [0.1, 0.002, 1.0],  # 2mm error in Y
            [0.0, 0.1, 1.003],  # 3mm error in Z
        ]
    )

    metrics = compute_charuco_metrics(detected, projected)

    # Check that errors are computed correctly
    assert "mean_error_mm" in metrics
    assert "median_error_mm" in metrics
    assert "max_error_mm" in metrics
    assert "rmse_mm" in metrics

    # Mean of [1mm, 2mm, 3mm] = 2mm
    assert metrics["mean_error_mm"] == pytest.approx(2.0, abs=0.1)
    # Median of [1mm, 2mm, 3mm] = 2mm
    assert metrics["median_error_mm"] == pytest.approx(2.0, abs=0.1)
    # Max of [1mm, 2mm, 3mm] = 3mm
    assert metrics["max_error_mm"] == pytest.approx(3.0, abs=0.1)


def test_compute_charuco_metrics_empty():
    """Test ChArUco metrics with empty input."""
    detected = np.array([]).reshape(0, 3)
    projected = np.array([]).reshape(0, 3)

    metrics = compute_charuco_metrics(detected, projected)

    # Should return NaN for empty input
    assert np.isnan(metrics["mean_error_mm"])
    assert np.isnan(metrics["median_error_mm"])
    assert np.isnan(metrics["max_error_mm"])
    assert np.isnan(metrics["rmse_mm"])


def test_compute_plane_fit_metrics():
    """Test plane fitting with synthetic planar points."""
    # Create planar points at z=1.0 with small noise
    np.random.seed(42)
    n_points = 100
    x = np.random.uniform(-0.5, 0.5, n_points)
    y = np.random.uniform(-0.5, 0.5, n_points)
    z = np.ones(n_points) * 1.0 + np.random.normal(0, 0.001, n_points)  # 1mm std noise

    points = np.column_stack([x, y, z])

    metrics = compute_plane_fit_metrics(points)

    # Normal should point in Z direction
    normal = metrics["plane_normal"]
    assert normal[0] == pytest.approx(0.0, abs=0.1)
    assert normal[1] == pytest.approx(0.0, abs=0.1)
    assert abs(normal[2]) == pytest.approx(1.0, abs=0.1)

    # RMSE should be close to 1mm (noise level)
    assert metrics["plane_fit_rmse_mm"] < 2.0  # should be ~1mm


def test_compute_plane_fit_metrics_perfect_plane():
    """Test plane fitting with perfect planar points (no noise)."""
    # Create perfect planar points at z=1.0
    x = np.array([0.0, 1.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    z = np.ones(4)

    points = np.column_stack([x, y, z])

    metrics = compute_plane_fit_metrics(points)

    # Normal should point in Z direction
    normal = metrics["plane_normal"]
    assert normal[0] == pytest.approx(0.0, abs=1e-6)
    assert normal[1] == pytest.approx(0.0, abs=1e-6)
    assert abs(normal[2]) == pytest.approx(1.0, abs=1e-6)

    # Perfect plane should have near-zero error
    assert metrics["plane_fit_rmse_mm"] < 1e-6
    assert metrics["plane_fit_max_error_mm"] < 1e-6


def test_compute_plane_fit_metrics_insufficient_points():
    """Test plane fitting with insufficient points."""
    points = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])  # Only 2 points

    metrics = compute_plane_fit_metrics(points)

    # Should return NaN for insufficient input
    assert np.isnan(metrics["plane_fit_rmse_mm"])
    assert np.isnan(metrics["plane_fit_max_error_mm"])
    assert all(np.isnan(metrics["plane_normal"]))
