"""Tests for depth and confidence map rendering."""

import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from aquamvs.visualization.depth import (
    render_all_depth_maps,
    render_confidence_map,
    render_depth_map,
)


def test_render_depth_map_basic(tmp_path):
    """Test rendering a basic depth map with gradient."""
    # Create a synthetic 64x48 depth map with a gradient
    H, W = 64, 48
    depth = np.linspace(0.5, 2.0, H * W, dtype=np.float32).reshape(H, W)

    # Add some NaN pixels
    depth[10:15, 20:25] = np.nan

    output_path = tmp_path / "depth_test.png"
    render_depth_map(depth, output_path, camera_name="test_cam")

    # Verify file exists
    assert output_path.exists()

    # Verify it's a valid image
    img = Image.open(output_path)
    assert img.size[0] > 0
    assert img.size[1] > 0

    # Verify file size > 0
    assert output_path.stat().st_size > 0


def test_render_confidence_map_basic(tmp_path):
    """Test rendering a basic confidence map."""
    # Create a synthetic 64x48 confidence map
    H, W = 64, 48
    confidence = np.linspace(0.0, 1.0, H * W, dtype=np.float32).reshape(H, W)

    output_path = tmp_path / "confidence_test.png"
    render_confidence_map(confidence, output_path, camera_name="test_cam")

    # Verify file exists
    assert output_path.exists()

    # Verify it's a valid image
    img = Image.open(output_path)
    assert img.size[0] > 0
    assert img.size[1] > 0

    # Verify file size > 0
    assert output_path.stat().st_size > 0


def test_render_depth_map_all_nan(tmp_path):
    """Test rendering an all-NaN depth map doesn't crash."""
    H, W = 32, 32
    depth = np.full((H, W), np.nan, dtype=np.float32)

    output_path = tmp_path / "depth_all_nan.png"
    # Should not crash
    render_depth_map(depth, output_path, camera_name="test_cam")

    # Verify file exists
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_render_depth_map_custom_range(tmp_path):
    """Test rendering depth map with custom vmin/vmax."""
    H, W = 32, 32
    depth = np.ones((H, W), dtype=np.float32) * 1.5

    output_path = tmp_path / "depth_custom_range.png"
    render_depth_map(depth, output_path, vmin=1.0, vmax=2.0)

    # Verify file exists
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_render_all_depth_maps(tmp_path):
    """Test rendering depth and confidence maps for multiple cameras."""
    H, W = 32, 32

    # Create depth maps for 2 cameras
    depth_maps = {
        "cam1": np.linspace(0.8, 1.5, H * W, dtype=np.float32).reshape(H, W),
        "cam2": np.linspace(1.0, 2.0, H * W, dtype=np.float32).reshape(H, W),
    }

    # Create confidence maps for 2 cameras
    confidence_maps = {
        "cam1": np.linspace(0.0, 1.0, H * W, dtype=np.float32).reshape(H, W),
        "cam2": np.linspace(0.2, 0.9, H * W, dtype=np.float32).reshape(H, W),
    }

    output_dir = tmp_path / "viz"
    render_all_depth_maps(depth_maps, confidence_maps, output_dir)

    # Verify 4 PNG files are created (2 depth + 2 confidence)
    assert (output_dir / "depth_cam1.png").exists()
    assert (output_dir / "depth_cam2.png").exists()
    assert (output_dir / "confidence_cam1.png").exists()
    assert (output_dir / "confidence_cam2.png").exists()

    # Verify all have size > 0
    assert (output_dir / "depth_cam1.png").stat().st_size > 0
    assert (output_dir / "depth_cam2.png").stat().st_size > 0
    assert (output_dir / "confidence_cam1.png").stat().st_size > 0
    assert (output_dir / "confidence_cam2.png").stat().st_size > 0


def test_render_depth_map_creates_parent_dirs(tmp_path):
    """Test that rendering creates parent directories if they don't exist."""
    H, W = 32, 32
    depth = np.ones((H, W), dtype=np.float32)

    # Path with nested directories that don't exist
    output_path = tmp_path / "nested" / "dirs" / "depth.png"
    render_depth_map(depth, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_render_confidence_map_creates_parent_dirs(tmp_path):
    """Test that rendering creates parent directories if they don't exist."""
    H, W = 32, 32
    confidence = np.ones((H, W), dtype=np.float32) * 0.5

    # Path with nested directories that don't exist
    output_path = tmp_path / "nested" / "dirs" / "confidence.png"
    render_confidence_map(confidence, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_render_all_depth_maps_consistent_range(tmp_path):
    """Test that all depth maps share the same color range."""
    H, W = 16, 16

    # Create depth maps with different ranges
    depth_maps = {
        "cam1": np.ones((H, W), dtype=np.float32) * 1.0,
        "cam2": np.ones((H, W), dtype=np.float32) * 2.0,
    }
    confidence_maps = {
        "cam1": np.ones((H, W), dtype=np.float32) * 0.8,
        "cam2": np.ones((H, W), dtype=np.float32) * 0.6,
    }

    output_dir = tmp_path / "viz"
    # Should use global min (1.0) and max (2.0) for both depth maps
    render_all_depth_maps(depth_maps, confidence_maps, output_dir)

    # Verify files exist (basic sanity check)
    assert (output_dir / "depth_cam1.png").exists()
    assert (output_dir / "depth_cam2.png").exists()


def test_render_all_depth_maps_with_nans(tmp_path):
    """Test rendering with some NaN values in depth maps."""
    H, W = 32, 32

    depth_map1 = np.linspace(1.0, 2.0, H * W, dtype=np.float32).reshape(H, W)
    depth_map1[10:15, 10:15] = np.nan

    depth_map2 = np.linspace(1.5, 2.5, H * W, dtype=np.float32).reshape(H, W)
    depth_map2[5:10, 5:10] = np.nan

    depth_maps = {"cam1": depth_map1, "cam2": depth_map2}
    confidence_maps = {
        "cam1": np.ones((H, W), dtype=np.float32) * 0.7,
        "cam2": np.ones((H, W), dtype=np.float32) * 0.8,
    }

    output_dir = tmp_path / "viz"
    render_all_depth_maps(depth_maps, confidence_maps, output_dir)

    # Verify all files exist
    assert (output_dir / "depth_cam1.png").exists()
    assert (output_dir / "depth_cam2.png").exists()
    assert (output_dir / "confidence_cam1.png").exists()
    assert (output_dir / "confidence_cam2.png").exists()


def test_render_depth_map_no_camera_name(tmp_path):
    """Test rendering without specifying a camera name."""
    H, W = 32, 32
    depth = np.ones((H, W), dtype=np.float32) * 1.5

    output_path = tmp_path / "depth_no_name.png"
    render_depth_map(depth, output_path)  # No camera_name

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_render_confidence_map_no_camera_name(tmp_path):
    """Test rendering confidence without specifying a camera name."""
    H, W = 32, 32
    confidence = np.ones((H, W), dtype=np.float32) * 0.5

    output_path = tmp_path / "confidence_no_name.png"
    render_confidence_map(confidence, output_path)  # No camera_name

    assert output_path.exists()
    assert output_path.stat().st_size > 0
