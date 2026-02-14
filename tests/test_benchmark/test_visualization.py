"""Tests for benchmark visualization functions."""

import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

from aquamvs.benchmark.visualization import render_comparison_grids, render_config_outputs
from aquamvs.config import SurfaceConfig


@pytest.fixture
def mock_images():
    """Create mock undistorted images."""
    return {
        "cam0": np.zeros((480, 640, 3), dtype=np.uint8),
        "cam1": np.zeros((480, 640, 3), dtype=np.uint8),
    }


@pytest.fixture
def mock_features():
    """Create mock feature extraction results."""
    return {
        "cam0": {
            "keypoints": torch.rand(10, 2) * 640,
            "descriptors": torch.rand(10, 256),
            "scores": torch.rand(10),
        },
        "cam1": {
            "keypoints": torch.rand(15, 2) * 640,
            "descriptors": torch.rand(15, 256),
            "scores": torch.rand(15),
        },
    }


@pytest.fixture
def mock_matches():
    """Create mock match results."""
    return {
        ("cam0", "cam1"): {
            "ref_keypoints": torch.rand(8, 2) * 640,
            "src_keypoints": torch.rand(8, 2) * 640,
            "scores": torch.rand(8),
        },
    }


@pytest.fixture
def mock_sparse_cloud():
    """Create mock sparse triangulation result."""
    return {
        "points_3d": torch.rand(20, 3),
        "scores": torch.rand(20),
    }


@pytest.fixture
def mock_projection_models():
    """Create mock projection models."""
    return {
        "cam0": MagicMock(),
        "cam1": MagicMock(),
    }


@pytest.fixture
def mock_camera_centers():
    """Create mock camera centers."""
    return {
        "cam0": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "cam1": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    }


def test_render_config_outputs_creates_keypoints(tmp_path, mock_images, mock_features, mock_matches, mock_sparse_cloud, mock_projection_models, mock_camera_centers):
    """Test that keypoint overlays are created for each camera."""
    with patch("aquamvs.benchmark.visualization.render_keypoints") as mock_render_kpts, \
         patch("aquamvs.benchmark.visualization.render_matches"), \
         patch("aquamvs.benchmark.visualization._sparse_cloud_to_open3d"), \
         patch("aquamvs.benchmark.visualization.render_scene"), \
         patch("aquamvs.benchmark.visualization.reconstruct_surface"), \
         patch("aquamvs.benchmark.visualization.o3d"):

        undistorted_tensors = {
            name: torch.from_numpy(img) for name, img in mock_images.items()
        }

        render_config_outputs(
            config_name="test_config",
            undistorted_images=mock_images,
            all_features=mock_features,
            all_matches=mock_matches,
            sparse_cloud=mock_sparse_cloud,
            projection_models=mock_projection_models,
            undistorted_tensors=undistorted_tensors,
            voxel_size=0.01,
            surface_config=SurfaceConfig(),
            output_dir=tmp_path,
            camera_centers=mock_camera_centers,
        )

        # Check that render_keypoints was called for each camera
        assert mock_render_kpts.call_count == 2
        config_dir = tmp_path / "test_config"
        assert config_dir.exists()


def test_render_config_outputs_creates_matches(tmp_path, mock_images, mock_features, mock_matches, mock_sparse_cloud, mock_projection_models, mock_camera_centers):
    """Test that match overlays are created for each pair."""
    with patch("aquamvs.benchmark.visualization.render_keypoints"), \
         patch("aquamvs.benchmark.visualization.render_matches") as mock_render_matches, \
         patch("aquamvs.benchmark.visualization._sparse_cloud_to_open3d"), \
         patch("aquamvs.benchmark.visualization.render_scene"), \
         patch("aquamvs.benchmark.visualization.reconstruct_surface"), \
         patch("aquamvs.benchmark.visualization.o3d"):

        undistorted_tensors = {
            name: torch.from_numpy(img) for name, img in mock_images.items()
        }

        render_config_outputs(
            config_name="test_config",
            undistorted_images=mock_images,
            all_features=mock_features,
            all_matches=mock_matches,
            sparse_cloud=mock_sparse_cloud,
            projection_models=mock_projection_models,
            undistorted_tensors=undistorted_tensors,
            voxel_size=0.01,
            surface_config=SurfaceConfig(),
            output_dir=tmp_path,
            camera_centers=mock_camera_centers,
        )

        # Check that render_matches was called for each pair
        assert mock_render_matches.call_count == 1


def test_render_config_outputs_creates_sparse_ply(tmp_path, mock_images, mock_features, mock_matches, mock_sparse_cloud, mock_projection_models, mock_camera_centers):
    """Test that sparse PLY is created with colors and normals."""
    with patch("aquamvs.benchmark.visualization.render_keypoints"), \
         patch("aquamvs.benchmark.visualization.render_matches"), \
         patch("aquamvs.benchmark.visualization._sparse_cloud_to_open3d") as mock_to_o3d, \
         patch("aquamvs.benchmark.visualization.render_scene"), \
         patch("aquamvs.benchmark.visualization.reconstruct_surface"), \
         patch("aquamvs.benchmark.visualization.o3d") as mock_o3d:

        # Mock Open3D PointCloud
        mock_pcd = MagicMock()
        mock_to_o3d.return_value = mock_pcd

        undistorted_tensors = {
            name: torch.from_numpy(img) for name, img in mock_images.items()
        }

        render_config_outputs(
            config_name="test_config",
            undistorted_images=mock_images,
            all_features=mock_features,
            all_matches=mock_matches,
            sparse_cloud=mock_sparse_cloud,
            projection_models=mock_projection_models,
            undistorted_tensors=undistorted_tensors,
            voxel_size=0.01,
            surface_config=SurfaceConfig(),
            output_dir=tmp_path,
            camera_centers=mock_camera_centers,
        )

        # Check that _sparse_cloud_to_open3d was called
        mock_to_o3d.assert_called_once()

        # Check that write_point_cloud was called
        mock_o3d.io.write_point_cloud.assert_called_once()


def test_render_config_outputs_creates_3d_renders(tmp_path, mock_images, mock_features, mock_matches, mock_sparse_cloud, mock_projection_models, mock_camera_centers):
    """Test that 3D renders are created from canonical viewpoints."""
    with patch("aquamvs.benchmark.visualization.render_keypoints"), \
         patch("aquamvs.benchmark.visualization.render_matches"), \
         patch("aquamvs.benchmark.visualization._sparse_cloud_to_open3d") as mock_to_o3d, \
         patch("aquamvs.benchmark.visualization.render_scene") as mock_render_scene, \
         patch("aquamvs.benchmark.visualization.reconstruct_surface") as mock_reconstruct, \
         patch("aquamvs.benchmark.visualization.o3d"):

        # Mock Open3D PointCloud and Mesh
        mock_pcd = MagicMock()
        mock_to_o3d.return_value = mock_pcd
        mock_mesh = MagicMock()
        mock_reconstruct.return_value = mock_mesh

        undistorted_tensors = {
            name: torch.from_numpy(img) for name, img in mock_images.items()
        }

        render_config_outputs(
            config_name="test_config",
            undistorted_images=mock_images,
            all_features=mock_features,
            all_matches=mock_matches,
            sparse_cloud=mock_sparse_cloud,
            projection_models=mock_projection_models,
            undistorted_tensors=undistorted_tensors,
            voxel_size=0.01,
            surface_config=SurfaceConfig(),
            output_dir=tmp_path,
            camera_centers=mock_camera_centers,
        )

        # Check that render_scene was called twice (sparse and mesh)
        assert mock_render_scene.call_count == 2
        # First call: sparse
        call_args = mock_render_scene.call_args_list[0]
        assert call_args[0][0] == mock_pcd
        assert call_args[1]["prefix"] == "sparse"
        # Second call: mesh
        call_args = mock_render_scene.call_args_list[1]
        assert call_args[0][0] == mock_mesh
        assert call_args[1]["prefix"] == "mesh"


def test_render_config_outputs_empty_sparse_cloud(tmp_path, mock_images, mock_features, mock_matches, mock_projection_models, mock_camera_centers):
    """Test that empty sparse cloud is handled gracefully."""
    empty_cloud = {
        "points_3d": torch.zeros((0, 3)),
        "scores": torch.zeros(0),
    }

    with patch("aquamvs.benchmark.visualization.render_keypoints"), \
         patch("aquamvs.benchmark.visualization.render_matches"), \
         patch("aquamvs.benchmark.visualization._sparse_cloud_to_open3d") as mock_to_o3d, \
         patch("aquamvs.benchmark.visualization.render_scene") as mock_render_scene, \
         patch("aquamvs.benchmark.visualization.reconstruct_surface") as mock_reconstruct, \
         patch("aquamvs.benchmark.visualization.o3d"):

        undistorted_tensors = {
            name: torch.from_numpy(img) for name, img in mock_images.items()
        }

        render_config_outputs(
            config_name="test_config",
            undistorted_images=mock_images,
            all_features=mock_features,
            all_matches=mock_matches,
            sparse_cloud=empty_cloud,
            projection_models=mock_projection_models,
            undistorted_tensors=undistorted_tensors,
            voxel_size=0.01,
            surface_config=SurfaceConfig(),
            output_dir=tmp_path,
            camera_centers=mock_camera_centers,
        )

        # Empty cloud should skip PLY, render_scene, and mesh reconstruction
        mock_to_o3d.assert_not_called()
        mock_render_scene.assert_not_called()
        mock_reconstruct.assert_not_called()


def test_render_config_outputs_creates_mesh_ply(tmp_path, mock_images, mock_features, mock_matches, mock_sparse_cloud, mock_projection_models, mock_camera_centers):
    """Test that mesh PLY is created with colors and faces."""
    with patch("aquamvs.benchmark.visualization.render_keypoints"), \
         patch("aquamvs.benchmark.visualization.render_matches"), \
         patch("aquamvs.benchmark.visualization._sparse_cloud_to_open3d") as mock_to_o3d, \
         patch("aquamvs.benchmark.visualization.render_scene"), \
         patch("aquamvs.benchmark.visualization.reconstruct_surface") as mock_reconstruct, \
         patch("aquamvs.benchmark.visualization.o3d") as mock_o3d:

        # Mock Open3D PointCloud and Mesh
        mock_pcd = MagicMock()
        mock_to_o3d.return_value = mock_pcd
        mock_mesh = MagicMock()
        mock_reconstruct.return_value = mock_mesh

        undistorted_tensors = {
            name: torch.from_numpy(img) for name, img in mock_images.items()
        }

        render_config_outputs(
            config_name="test_config",
            undistorted_images=mock_images,
            all_features=mock_features,
            all_matches=mock_matches,
            sparse_cloud=mock_sparse_cloud,
            projection_models=mock_projection_models,
            undistorted_tensors=undistorted_tensors,
            voxel_size=0.01,
            surface_config=SurfaceConfig(),
            output_dir=tmp_path,
            camera_centers=mock_camera_centers,
        )

        # Check that reconstruct_surface was called after _sparse_cloud_to_open3d
        mock_reconstruct.assert_called_once_with(mock_pcd, SurfaceConfig())

        # Check that write_triangle_mesh was called
        mock_o3d.io.write_triangle_mesh.assert_called_once()


def test_render_config_outputs_mesh_failure_graceful(tmp_path, mock_images, mock_features, mock_matches, mock_sparse_cloud, mock_projection_models, mock_camera_centers):
    """Test that mesh reconstruction failure does not crash the benchmark."""
    with patch("aquamvs.benchmark.visualization.render_keypoints"), \
         patch("aquamvs.benchmark.visualization.render_matches"), \
         patch("aquamvs.benchmark.visualization._sparse_cloud_to_open3d") as mock_to_o3d, \
         patch("aquamvs.benchmark.visualization.render_scene") as mock_render_scene, \
         patch("aquamvs.benchmark.visualization.reconstruct_surface") as mock_reconstruct, \
         patch("aquamvs.benchmark.visualization.o3d") as mock_o3d:

        # Mock Open3D PointCloud
        mock_pcd = MagicMock()
        mock_to_o3d.return_value = mock_pcd

        # Mock reconstruct_surface to raise an exception
        mock_reconstruct.side_effect = RuntimeError("Degenerate point cloud")

        undistorted_tensors = {
            name: torch.from_numpy(img) for name, img in mock_images.items()
        }

        # Should not raise an error
        render_config_outputs(
            config_name="test_config",
            undistorted_images=mock_images,
            all_features=mock_features,
            all_matches=mock_matches,
            sparse_cloud=mock_sparse_cloud,
            projection_models=mock_projection_models,
            undistorted_tensors=undistorted_tensors,
            voxel_size=0.01,
            surface_config=SurfaceConfig(),
            output_dir=tmp_path,
            camera_centers=mock_camera_centers,
        )

        # Sparse PLY and renders should still be produced
        mock_to_o3d.assert_called_once()
        mock_o3d.io.write_point_cloud.assert_called_once()
        # render_scene called once for sparse (not for mesh since it failed)
        assert mock_render_scene.call_count == 1

        # Mesh PLY should not be written
        mock_o3d.io.write_triangle_mesh.assert_not_called()


def test_render_comparison_grids_keypoints(tmp_path):
    """Test that keypoint comparison grid is created."""
    # Create mock keypoint images with actual image data
    config_names = ["config1", "config2"]
    camera_names = ["cam0", "cam1"]

    for config_name in config_names:
        config_dir = tmp_path / config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        for cam_name in camera_names:
            # Create a real dummy image using cv2
            import cv2
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            img_path = config_dir / f"keypoints_{cam_name}.png"
            cv2.imwrite(str(img_path), img)

    # Run the actual function
    render_comparison_grids(config_names, camera_names, tmp_path)

    # Check that output file was created
    grid_path = tmp_path / "comparison" / "keypoints_grid.png"
    assert grid_path.exists()


def test_render_comparison_grids_sparse_renders(tmp_path):
    """Test that sparse render comparison grid is created."""
    # Create mock sparse render images with actual image data
    config_names = ["config1", "config2"]
    viewpoints = ["top", "oblique", "side"]

    for config_name in config_names:
        config_dir = tmp_path / config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        for viewpoint in viewpoints:
            # Create a real dummy image using matplotlib
            import matplotlib.pyplot as plt_local
            fig, ax = plt_local.subplots(figsize=(2, 2))
            ax.axis("off")
            img_path = config_dir / f"sparse_{viewpoint}.png"
            plt_local.savefig(img_path, dpi=50)
            plt_local.close()

    # Run the actual function
    render_comparison_grids(config_names, ["cam0"], tmp_path)

    # Check that output file was created
    grid_path = tmp_path / "comparison" / "sparse_renders_grid.png"
    assert grid_path.exists()


def test_render_comparison_grids_mesh_grid(tmp_path):
    """Test that mesh render comparison grid is created."""
    # Create mock mesh render images with actual image data
    config_names = ["config1", "config2"]
    viewpoints = ["top", "oblique", "side"]

    for config_name in config_names:
        config_dir = tmp_path / config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        for viewpoint in viewpoints:
            # Create a real dummy image using matplotlib
            import matplotlib.pyplot as plt_local
            fig, ax = plt_local.subplots(figsize=(2, 2))
            ax.axis("off")
            img_path = config_dir / f"mesh_{viewpoint}.png"
            plt_local.savefig(img_path, dpi=50)
            plt_local.close()

    # Run the actual function
    render_comparison_grids(config_names, ["cam0"], tmp_path)

    # Check that output file was created
    grid_path = tmp_path / "comparison" / "mesh_grid.png"
    assert grid_path.exists()


def test_render_comparison_grids_missing_images(tmp_path):
    """Test that missing images are handled gracefully."""
    config_names = ["config1"]
    camera_names = ["cam0", "cam1"]

    # Create directory but no images
    (tmp_path / "config1").mkdir(parents=True, exist_ok=True)

    # Should not raise an error - missing images are handled
    render_comparison_grids(config_names, camera_names, tmp_path)

    # Check that output files were still created
    grid_path = tmp_path / "comparison" / "keypoints_grid.png"
    assert grid_path.exists()


def test_render_comparison_grids_single_config(tmp_path):
    """Test grid generation with a single configuration."""
    config_names = ["config1"]
    camera_names = ["cam0"]

    config_dir = tmp_path / "config1"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Create a real image
    import cv2
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(config_dir / "keypoints_cam0.png"), img)

    # Should not raise an error
    render_comparison_grids(config_names, camera_names, tmp_path)

    # Check that output was created
    grid_path = tmp_path / "comparison" / "keypoints_grid.png"
    assert grid_path.exists()
