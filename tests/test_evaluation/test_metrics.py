"""Tests for evaluation metrics."""

import numpy as np
import open3d as o3d
import pytest
import torch

from aquamvs.evaluation import (
    cloud_to_cloud_distance,
    height_map_difference,
    reprojection_error,
)
from aquamvs.projection import RefractiveProjectionModel


def create_flat_cloud(z=1.5, n_points=200, x_range=(-0.1, 0.1), y_range=(-0.1, 0.1)):
    """Create a synthetic point cloud of a flat horizontal plane."""
    xy = np.random.RandomState(42).uniform(
        [x_range[0], y_range[0]], [x_range[1], y_range[1]], size=(n_points, 2)
    )
    z_vals = np.full(n_points, z)
    points = np.column_stack([xy, z_vals])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def create_sphere_cloud(center, radius=0.05, n_points=200):
    """Create a synthetic point cloud in the shape of a sphere."""
    rng = np.random.RandomState(42)
    theta = rng.uniform(0, 2 * np.pi, n_points)
    phi = rng.uniform(0, np.pi, n_points)
    x = radius * np.sin(phi) * np.cos(theta) + center[0]
    y = radius * np.sin(phi) * np.sin(theta) + center[1]
    z = radius * np.cos(phi) + center[2]
    points = np.column_stack([x, y, z])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


class TestCloudToCloudDistance:
    """Tests for cloud_to_cloud_distance."""

    def test_identical_clouds(self):
        """Distance between identical clouds should be zero."""
        cloud = create_flat_cloud(z=1.5, n_points=200)

        result = cloud_to_cloud_distance(cloud, cloud)

        assert result["mean_a_to_b"] < 1e-10
        assert result["mean_b_to_a"] < 1e-10
        assert result["hausdorff_a_to_b"] < 1e-10
        assert result["hausdorff_b_to_a"] < 1e-10
        assert result["hausdorff"] < 1e-10
        assert result["mean"] < 1e-10
        assert result["median_a_to_b"] < 1e-10
        assert result["median_b_to_a"] < 1e-10

    def test_known_offset(self):
        """Cloud offset by a known distance."""
        cloud_a = create_flat_cloud(z=1.5, n_points=200)

        # Shift cloud_a by 10mm in Z to create cloud_b
        points_b = np.asarray(cloud_a.points).copy()
        points_b[:, 2] += 0.01  # 10mm offset
        cloud_b = o3d.geometry.PointCloud()
        cloud_b.points = o3d.utility.Vector3dVector(points_b)

        result = cloud_to_cloud_distance(cloud_a, cloud_b)

        # All distances should be approximately 10mm (0.01m)
        assert np.isclose(result["mean_a_to_b"], 0.01, atol=1e-4)
        assert np.isclose(result["mean_b_to_a"], 0.01, atol=1e-4)
        assert np.isclose(result["hausdorff_a_to_b"], 0.01, atol=1e-4)
        assert np.isclose(result["hausdorff_b_to_a"], 0.01, atol=1e-4)
        assert np.isclose(result["hausdorff"], 0.01, atol=1e-4)
        assert np.isclose(result["mean"], 0.01, atol=1e-4)

    def test_asymmetric_clouds(self):
        """Asymmetric clouds (A is subset of B)."""
        # Create cloud B as a full flat plane
        cloud_b = create_flat_cloud(z=1.5, n_points=500)

        # Create cloud A as a subset (same Z, but fewer points in a smaller region)
        cloud_a = create_flat_cloud(z=1.5, n_points=100, x_range=(-0.05, 0.05))

        result = cloud_to_cloud_distance(cloud_a, cloud_b)

        # A's points are in B's region, so A->B distance should be small
        assert result["mean_a_to_b"] < 0.02  # Should be small

        # B has points not in A's region, so B->A distance may be larger
        # But since they're at the same Z, it will still be relatively small
        # This test just verifies the computation runs correctly
        assert result["mean_b_to_a"] >= 0.0


class TestHeightMapDifference:
    """Tests for height_map_difference."""

    def test_identical_clouds(self):
        """Height map difference for identical clouds should be zero."""
        cloud = create_flat_cloud(z=1.5, n_points=500)

        result = height_map_difference(cloud, cloud, grid_resolution=0.005)

        # All differences should be zero (or very close due to interpolation)
        assert np.abs(result["mean_diff"]) < 1e-4
        assert result["std_diff"] < 1e-4
        assert result["abs_mean_diff"] < 1e-4
        assert result["max_abs_diff"] < 1e-3
        assert result["rmse"] < 1e-4

        # Grid and diff_map should be present
        assert "diff_map" in result
        assert "grid_x" in result
        assert "grid_y" in result

    def test_flat_planes_with_offset(self):
        """Two flat planes with known Z offset."""
        cloud_a = create_flat_cloud(z=1.50, n_points=500)
        cloud_b = create_flat_cloud(z=1.51, n_points=500)

        result = height_map_difference(cloud_a, cloud_b, grid_resolution=0.005)

        # Mean diff should be approximately -0.01 (A is below B)
        assert np.isclose(result["mean_diff"], -0.01, atol=0.002)
        # RMSE should be approximately 0.01
        assert np.isclose(result["rmse"], 0.01, atol=0.002)
        # Abs mean diff should be approximately 0.01
        assert np.isclose(result["abs_mean_diff"], 0.01, atol=0.002)
        # Max abs diff should be around 0.01
        assert result["max_abs_diff"] < 0.015

    def test_non_overlapping_clouds(self):
        """Clouds in different XY regions should produce NaN metrics."""
        # Cloud A at X ~ 0
        cloud_a = create_flat_cloud(
            z=1.5, n_points=200, x_range=(-0.05, 0.0), y_range=(-0.05, 0.0)
        )
        # Cloud B at X ~ 1 (far away in X)
        cloud_b = create_flat_cloud(
            z=1.5, n_points=200, x_range=(0.9, 1.0), y_range=(-0.05, 0.0)
        )

        result = height_map_difference(cloud_a, cloud_b, grid_resolution=0.005)

        # All scalar metrics should be NaN
        assert np.isnan(result["mean_diff"])
        assert np.isnan(result["std_diff"])
        assert np.isnan(result["abs_mean_diff"])
        assert np.isnan(result["max_abs_diff"])
        assert np.isnan(result["rmse"])

        # diff_map should be all NaN
        assert np.all(np.isnan(result["diff_map"]))

    def test_grid_resolution(self):
        """Verify grid resolution affects grid size."""
        cloud_a = create_flat_cloud(z=1.5, n_points=200)
        cloud_b = create_flat_cloud(z=1.51, n_points=200)

        result_fine = height_map_difference(cloud_a, cloud_b, grid_resolution=0.002)
        result_coarse = height_map_difference(cloud_a, cloud_b, grid_resolution=0.01)

        # Fine grid should have more cells than coarse grid
        assert len(result_fine["grid_x"]) > len(result_coarse["grid_x"])
        assert len(result_fine["grid_y"]) > len(result_coarse["grid_y"])


class TestReprojectionError:
    """Tests for reprojection_error."""

    @pytest.fixture
    def simple_projection_model(self):
        """Create a simple refractive projection model for testing."""
        device = "cpu"
        # Identity rotation (camera frame = world frame)
        R = torch.eye(3, dtype=torch.float32, device=device)
        # Zero translation (camera at world origin)
        t = torch.zeros(3, dtype=torch.float32, device=device)
        # Simple intrinsics
        K = torch.tensor(
            [[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        water_z = 1.0
        normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
        n_air = 1.0
        n_water = 1.333

        return RefractiveProjectionModel(K, R, t, water_z, normal, n_air, n_water)

    def test_zero_error(self, simple_projection_model):
        """Perfect reprojection should give zero error."""
        device = "cpu"

        # Create some 3D points in the water (Z > water_z)
        points_3d = torch.tensor(
            [
                [0.0, 0.0, 1.5],
                [0.1, 0.0, 1.5],
                [0.0, 0.1, 1.5],
                [-0.1, 0.0, 1.5],
                [0.0, -0.1, 1.5],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Project them to get perfect observations
        pixels, valid = simple_projection_model.project(points_3d)
        assert valid.all()

        # Create observations dict
        observations = {"cam1": pixels}
        projection_models = {"cam1": simple_projection_model}

        result = reprojection_error(points_3d, observations, projection_models)

        # Error should be zero (within numerical precision)
        assert result["mean_error"] < 1e-4
        assert result["median_error"] < 1e-4
        assert result["max_error"] < 1e-4
        assert "cam1" in result["per_camera"]
        assert result["per_camera"]["cam1"] < 1e-4
        assert len(result["errors"]) == 5

    def test_known_error(self, simple_projection_model):
        """Add a known pixel offset to observations."""
        device = "cpu"

        points_3d = torch.tensor(
            [
                [0.0, 0.0, 1.5],
                [0.1, 0.0, 1.5],
                [0.0, 0.1, 1.5],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Project to get base pixels
        pixels, valid = simple_projection_model.project(points_3d)
        assert valid.all()

        # Add a 2-pixel offset in u direction
        pixels_with_error = pixels + torch.tensor([2.0, 0.0], dtype=torch.float32)

        observations = {"cam1": pixels_with_error}
        projection_models = {"cam1": simple_projection_model}

        result = reprojection_error(points_3d, observations, projection_models)

        # Mean error should be approximately 2.0 pixels
        assert np.isclose(result["mean_error"], 2.0, atol=0.1)
        assert np.isclose(result["median_error"], 2.0, atol=0.1)

    def test_nan_observations(self, simple_projection_model):
        """NaN observations should be excluded."""
        device = "cpu"

        points_3d = torch.tensor(
            [
                [0.0, 0.0, 1.5],
                [0.1, 0.0, 1.5],
                [0.0, 0.1, 1.5],
                [-0.1, 0.0, 1.5],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Project to get base pixels
        pixels, valid = simple_projection_model.project(points_3d)
        assert valid.all()

        # Mark the second and third observations as NaN (unobserved)
        pixels_with_nan = pixels.clone()
        pixels_with_nan[1, :] = float("nan")
        pixels_with_nan[2, :] = float("nan")

        observations = {"cam1": pixels_with_nan}
        projection_models = {"cam1": simple_projection_model}

        result = reprojection_error(points_3d, observations, projection_models)

        # Should only process 2 valid observations (indices 0 and 3)
        assert len(result["errors"]) == 2
        assert result["mean_error"] < 1e-4  # Those two should have zero error

    def test_multiple_cameras(self, simple_projection_model):
        """Test with multiple cameras."""
        device = "cpu"

        points_3d = torch.tensor(
            [
                [0.0, 0.0, 1.5],
                [0.1, 0.0, 1.5],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Perfect observations for cam1
        pixels1, valid = simple_projection_model.project(points_3d)
        # Observations with 1-pixel error for cam2
        pixels2 = pixels1 + torch.tensor([1.0, 0.0], dtype=torch.float32)

        observations = {"cam1": pixels1, "cam2": pixels2}
        projection_models = {
            "cam1": simple_projection_model,
            "cam2": simple_projection_model,
        }

        result = reprojection_error(points_3d, observations, projection_models)

        # cam1 should have ~0 error
        assert result["per_camera"]["cam1"] < 1e-4
        # cam2 should have ~1 pixel error
        assert np.isclose(result["per_camera"]["cam2"], 1.0, atol=0.1)
        # Overall mean should be ~0.5 pixels (average of 0 and 1)
        assert np.isclose(result["mean_error"], 0.5, atol=0.1)
        # Total of 4 observations (2 points * 2 cameras)
        assert len(result["errors"]) == 4

    def test_no_valid_observations(self, simple_projection_model):
        """All observations NaN should return NaN metrics."""
        device = "cpu"

        points_3d = torch.tensor(
            [[0.0, 0.0, 1.5], [0.1, 0.0, 1.5]], dtype=torch.float32, device=device
        )

        # All observations are NaN
        pixels_nan = torch.full((2, 2), float("nan"), dtype=torch.float32)

        observations = {"cam1": pixels_nan}
        projection_models = {"cam1": simple_projection_model}

        result = reprojection_error(points_3d, observations, projection_models)

        assert np.isnan(result["mean_error"])
        assert np.isnan(result["median_error"])
        assert np.isnan(result["max_error"])
        assert result["per_camera"] == {}
        assert len(result["errors"]) == 0
