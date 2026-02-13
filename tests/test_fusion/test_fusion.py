"""Tests for depth map fusion."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from aquamvs.config import FusionConfig
from aquamvs.fusion import (
    backproject_depth_map,
    fuse_depth_maps,
    load_point_cloud,
    save_point_cloud,
)
from aquamvs.projection import RefractiveProjectionModel


@pytest.fixture
def reference_camera(device):
    """Create a simple camera at world origin looking down."""
    R = torch.eye(3, dtype=torch.float32, device=device)
    t = torch.zeros(3, dtype=torch.float32, device=device)
    K = torch.tensor(
        [[400.0, 0.0, 8.0], [0.0, 400.0, 8.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    water_z = 1.0
    normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
    return RefractiveProjectionModel(K, R, t, water_z, normal, 1.0, 1.333)


@pytest.fixture
def offset_camera(device):
    """Create a camera offset to the right."""
    R = torch.eye(3, dtype=torch.float32, device=device)
    t = torch.tensor([-0.2, 0.0, 0.0], dtype=torch.float32, device=device)
    K = torch.tensor(
        [[400.0, 0.0, 8.0], [0.0, 400.0, 8.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    water_z = 1.0
    normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
    return RefractiveProjectionModel(K, R, t, water_z, normal, 1.0, 1.333)


@pytest.fixture
def third_camera(device):
    """Create a third camera offset in Y direction."""
    R = torch.eye(3, dtype=torch.float32, device=device)
    t = torch.tensor([0.0, -0.2, 0.0], dtype=torch.float32, device=device)
    K = torch.tensor(
        [[400.0, 0.0, 8.0], [0.0, 400.0, 8.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    water_z = 1.0
    normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
    return RefractiveProjectionModel(K, R, t, water_z, normal, 1.0, 1.333)


class TestBackprojectDepthMap:
    """Tests for backproject_depth_map."""

    def test_known_geometry(self, reference_camera, device):
        """Test back-projection with known geometry."""
        # Create a small depth map with a few known valid pixels
        H, W = 16, 16
        depth_map = torch.full((H, W), float("nan"), device=device)

        # Set specific pixels with known depths
        # Use pixels that will be extracted in a predictable order
        depth_map[6, 10] = 0.8  # row=6, col=10 -> u=10, v=6
        depth_map[8, 8] = 0.5   # row=8, col=8 -> u=8, v=8

        # Create a dummy image
        image = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)

        # Back-project
        result = backproject_depth_map(reference_camera, depth_map, image)

        # Verify we get 2 points
        assert result["points"].shape == (2, 3)
        assert result["colors"].shape == (2, 3)
        assert result["confidence"].shape == (2,)

        # Verify each point individually by casting ray at the actual pixel coordinates
        # The function extracts pixels in row-major order, so we get (6,10) then (8,8)
        actual_pixels = torch.tensor([[10.0, 6.0], [8.0, 8.0]], device=device)
        actual_depths = torch.tensor([0.8, 0.5], device=device)

        origins, directions = reference_camera.cast_ray(actual_pixels)
        expected_points = origins + actual_depths.unsqueeze(-1) * directions

        torch.testing.assert_close(
            result["points"],
            expected_points,
            atol=1e-5,
            rtol=0,
        )

    def test_color_sampling(self, reference_camera, device):
        """Test that colors are correctly sampled from the BGR image."""
        H, W = 16, 16
        depth_map = torch.full((H, W), float("nan"), device=device)

        # Set a few valid pixels
        depth_map[4, 4] = 0.5
        depth_map[8, 8] = 0.6
        depth_map[12, 12] = 0.7

        # Create an image with distinct BGR colors at those pixels
        image = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
        image[4, 4] = torch.tensor([255, 0, 0], dtype=torch.uint8, device=device)  # Blue
        image[8, 8] = torch.tensor([0, 255, 0], dtype=torch.uint8, device=device)  # Green
        image[12, 12] = torch.tensor([0, 0, 255], dtype=torch.uint8, device=device)  # Red

        # Back-project
        result = backproject_depth_map(reference_camera, depth_map, image)

        # Expected RGB colors (BGR -> RGB, uint8 -> float [0, 1])
        expected_colors = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )

        torch.testing.assert_close(
            result["colors"],
            expected_colors,
            atol=1e-5,
            rtol=0,
        )

    def test_all_nan_depth_map(self, reference_camera, device):
        """Test that all-NaN depth map returns empty tensors."""
        H, W = 16, 16
        depth_map = torch.full((H, W), float("nan"), device=device)
        image = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)

        result = backproject_depth_map(reference_camera, depth_map, image)

        assert result["points"].shape == (0, 3)
        assert result["colors"].shape == (0, 3)
        assert result["confidence"].shape == (0,)

    def test_confidence_handling(self, reference_camera, device):
        """Test that confidence is correctly passed through."""
        H, W = 16, 16
        depth_map = torch.full((H, W), float("nan"), device=device)
        depth_map[8, 8] = 0.5
        depth_map[10, 10] = 0.6

        confidence = torch.zeros((H, W), device=device)
        confidence[8, 8] = 0.9
        confidence[10, 10] = 0.7

        image = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)

        result = backproject_depth_map(reference_camera, depth_map, image, confidence)

        # Check confidence values match
        expected_conf = torch.tensor([0.9, 0.7], device=device)
        torch.testing.assert_close(
            result["confidence"],
            expected_conf,
            atol=1e-5,
            rtol=0,
        )

    def test_no_confidence_defaults_to_ones(self, reference_camera, device):
        """Test that missing confidence defaults to ones."""
        H, W = 16, 16
        depth_map = torch.full((H, W), float("nan"), device=device)
        depth_map[8, 8] = 0.5

        image = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)

        result = backproject_depth_map(reference_camera, depth_map, image)

        assert result["confidence"].shape == (1,)
        assert result["confidence"][0] == 1.0


class TestFuseDepthMaps:
    """Tests for fuse_depth_maps."""

    def test_multi_camera_merge(
        self, reference_camera, offset_camera, third_camera, device
    ):
        """Test that points from multiple cameras are merged."""
        H, W = 16, 16

        # Create depth maps for each camera
        depth1 = torch.full((H, W), float("nan"), device=device)
        depth1[8, 8] = 0.5
        depth1[9, 9] = 0.6

        depth2 = torch.full((H, W), float("nan"), device=device)
        depth2[8, 8] = 0.5
        depth2[7, 7] = 0.55

        depth3 = torch.full((H, W), float("nan"), device=device)
        depth3[10, 10] = 0.7

        conf1 = torch.ones((H, W), device=device)
        conf2 = torch.ones((H, W), device=device)
        conf3 = torch.ones((H, W), device=device)

        # Create dummy images
        image1 = torch.ones((H, W, 3), dtype=torch.uint8, device=device) * 100
        image2 = torch.ones((H, W, 3), dtype=torch.uint8, device=device) * 150
        image3 = torch.ones((H, W, 3), dtype=torch.uint8, device=device) * 200

        ring_cameras = ["cam1", "cam2", "cam3"]
        projection_models = {
            "cam1": reference_camera,
            "cam2": offset_camera,
            "cam3": third_camera,
        }
        filtered_depth_maps = {
            "cam1": depth1,
            "cam2": depth2,
            "cam3": depth3,
        }
        filtered_confidence_maps = {
            "cam1": conf1,
            "cam2": conf2,
            "cam3": conf3,
        }
        images = {
            "cam1": image1,
            "cam2": image2,
            "cam3": image3,
        }

        config = FusionConfig(voxel_size=0.001)

        # Fuse
        pcd = fuse_depth_maps(
            ring_cameras,
            projection_models,
            filtered_depth_maps,
            filtered_confidence_maps,
            images,
            config,
        )

        # Verify we have points (after voxel downsampling, count may be <= total)
        assert len(pcd.points) > 0
        # Total input points: 2 + 2 + 1 = 5, after voxel downsampling <= 5
        assert len(pcd.points) <= 5

        # Verify we have colors
        assert len(pcd.colors) == len(pcd.points)

        # Verify we have normals
        assert pcd.has_normals()

    def test_voxel_deduplication(self, reference_camera, offset_camera, device):
        """Test that voxel downsampling reduces point count."""
        H, W = 16, 16

        # Create overlapping depth maps (same 3D region viewed from different cameras)
        depth1 = torch.full((H, W), float("nan"), device=device)
        depth2 = torch.full((H, W), float("nan"), device=device)

        # Fill a small region with valid depths
        depth1[6:10, 6:10] = 0.5
        depth2[6:10, 6:10] = 0.5

        conf1 = torch.ones((H, W), device=device)
        conf2 = torch.ones((H, W), device=device)

        image1 = torch.ones((H, W, 3), dtype=torch.uint8, device=device) * 128
        image2 = torch.ones((H, W, 3), dtype=torch.uint8, device=device) * 128

        ring_cameras = ["cam1", "cam2"]
        projection_models = {
            "cam1": reference_camera,
            "cam2": offset_camera,
        }
        filtered_depth_maps = {
            "cam1": depth1,
            "cam2": depth2,
        }
        filtered_confidence_maps = {
            "cam1": conf1,
            "cam2": conf2,
        }
        images = {
            "cam1": image1,
            "cam2": image2,
        }

        # Use a larger voxel size to ensure deduplication
        config = FusionConfig(voxel_size=0.01)

        pcd = fuse_depth_maps(
            ring_cameras,
            projection_models,
            filtered_depth_maps,
            filtered_confidence_maps,
            images,
            config,
        )

        # Total input points: 16 + 16 = 32
        # After voxel downsampling with 0.01m voxels, should be significantly fewer
        assert len(pcd.points) < 32
        assert len(pcd.points) > 0

    def test_normals_estimated(self, reference_camera, device):
        """Test that normals are estimated and are unit vectors."""
        H, W = 16, 16

        # Create a depth map with enough points for normal estimation
        depth1 = torch.full((H, W), float("nan"), device=device)
        depth1[6:12, 6:12] = 0.5

        conf1 = torch.ones((H, W), device=device)
        image1 = torch.ones((H, W, 3), dtype=torch.uint8, device=device) * 128

        ring_cameras = ["cam1"]
        projection_models = {"cam1": reference_camera}
        filtered_depth_maps = {"cam1": depth1}
        filtered_confidence_maps = {"cam1": conf1}
        images = {"cam1": image1}

        config = FusionConfig(voxel_size=0.001)

        pcd = fuse_depth_maps(
            ring_cameras,
            projection_models,
            filtered_depth_maps,
            filtered_confidence_maps,
            images,
            config,
        )

        # Verify normals exist
        assert pcd.has_normals()

        # Verify normals are unit vectors
        normals = np.asarray(pcd.normals)
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_empty_inputs(self, device):
        """Test that empty depth maps return an empty point cloud."""
        H, W = 16, 16

        # All-NaN depth maps
        depth1 = torch.full((H, W), float("nan"), device=device)
        depth2 = torch.full((H, W), float("nan"), device=device)

        conf1 = torch.zeros((H, W), device=device)
        conf2 = torch.zeros((H, W), device=device)

        image1 = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
        image2 = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)

        # Need cameras but they won't be used
        R = torch.eye(3, dtype=torch.float32, device=device)
        t = torch.zeros(3, dtype=torch.float32, device=device)
        K = torch.tensor(
            [[400.0, 0.0, 8.0], [0.0, 400.0, 8.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        cam = RefractiveProjectionModel(
            K, R, t, 1.0, torch.tensor([0.0, 0.0, -1.0], device=device), 1.0, 1.333
        )

        ring_cameras = ["cam1", "cam2"]
        projection_models = {"cam1": cam, "cam2": cam}
        filtered_depth_maps = {"cam1": depth1, "cam2": depth2}
        filtered_confidence_maps = {"cam1": conf1, "cam2": conf2}
        images = {"cam1": image1, "cam2": image2}

        config = FusionConfig(voxel_size=0.001)

        pcd = fuse_depth_maps(
            ring_cameras,
            projection_models,
            filtered_depth_maps,
            filtered_confidence_maps,
            images,
            config,
        )

        # Verify empty point cloud
        assert len(pcd.points) == 0
        assert len(pcd.colors) == 0


class TestPointCloudIO:
    """Tests for save_point_cloud and load_point_cloud."""

    def test_save_load_roundtrip(self, reference_camera, device):
        """Test saving and loading a point cloud preserves data."""
        import open3d as o3d

        # Create a simple point cloud
        H, W = 16, 16
        depth1 = torch.full((H, W), float("nan"), device=device)
        depth1[8, 8] = 0.5
        depth1[9, 9] = 0.6
        depth1[10, 10] = 0.7

        conf1 = torch.ones((H, W), device=device)

        # Create an image with distinct colors
        image1 = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
        image1[8, 8] = torch.tensor([100, 150, 200], dtype=torch.uint8, device=device)
        image1[9, 9] = torch.tensor([50, 100, 150], dtype=torch.uint8, device=device)
        image1[10, 10] = torch.tensor([200, 100, 50], dtype=torch.uint8, device=device)

        ring_cameras = ["cam1"]
        projection_models = {"cam1": reference_camera}
        filtered_depth_maps = {"cam1": depth1}
        filtered_confidence_maps = {"cam1": conf1}
        images = {"cam1": image1}

        config = FusionConfig(voxel_size=0.001)

        pcd_original = fuse_depth_maps(
            ring_cameras,
            projection_models,
            filtered_depth_maps,
            filtered_confidence_maps,
            images,
            config,
        )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            temp_path = Path(f.name)

        try:
            save_point_cloud(pcd_original, temp_path)

            # Load back
            pcd_loaded = load_point_cloud(temp_path)

            # Verify points match
            points_orig = np.asarray(pcd_original.points)
            points_loaded = np.asarray(pcd_loaded.points)
            np.testing.assert_allclose(points_orig, points_loaded, atol=1e-5)

            # Verify colors match
            colors_orig = np.asarray(pcd_original.colors)
            colors_loaded = np.asarray(pcd_loaded.colors)
            np.testing.assert_allclose(colors_orig, colors_loaded, atol=1e-5)

        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()

    def test_normal_orientation(self, reference_camera, device):
        """Test that normals are oriented toward -Z (upward)."""
        H, W = 16, 16

        # Create a roughly horizontal plane
        depth1 = torch.full((H, W), float("nan"), device=device)
        depth1[4:12, 4:12] = 0.5

        conf1 = torch.ones((H, W), device=device)
        image1 = torch.ones((H, W, 3), dtype=torch.uint8, device=device) * 128

        ring_cameras = ["cam1"]
        projection_models = {"cam1": reference_camera}
        filtered_depth_maps = {"cam1": depth1}
        filtered_confidence_maps = {"cam1": conf1}
        images = {"cam1": image1}

        config = FusionConfig(voxel_size=0.001)

        pcd = fuse_depth_maps(
            ring_cameras,
            projection_models,
            filtered_depth_maps,
            filtered_confidence_maps,
            images,
            config,
        )

        # Verify normals exist
        assert pcd.has_normals()

        # Verify normals point approximately toward -Z (upward)
        normals = np.asarray(pcd.normals)
        # Z component should be negative (pointing up in our coordinate system)
        mean_z = np.mean(normals[:, 2])
        assert mean_z < 0, f"Expected normals to point upward (-Z), but mean Z={mean_z}"
