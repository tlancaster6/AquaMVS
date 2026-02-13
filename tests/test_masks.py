"""Tests for ROI mask loading and application."""

import logging
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from aquamvs.calibration import CameraData
from aquamvs.masks import (
    apply_mask_to_depth,
    apply_mask_to_features,
    load_all_masks,
    load_mask,
)


@pytest.fixture
def mock_camera_data():
    """Create mock CameraData for testing."""
    return CameraData(
        name="cam0",
        K=torch.eye(3),
        R=torch.eye(3),
        t=torch.zeros(3),
        dist_coeffs=torch.zeros(5),
        image_size=(640, 480),  # (width, height)
        is_fisheye=False,
        is_auxiliary=False,
    )


def test_load_mask_found(tmp_path: Path, mock_camera_data: CameraData):
    """Test load_mask successfully loads a valid mask."""
    # Create a 640x480 white PNG
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()

    mask_path = mask_dir / "cam0.png"
    mask_img = np.full((480, 640), 255, dtype=np.uint8)
    cv2.imwrite(str(mask_path), mask_img)

    # Load it
    loaded = load_mask(mask_dir, "cam0", (640, 480))

    assert loaded is not None
    assert loaded.shape == (480, 640)
    assert loaded.dtype == np.uint8
    assert np.all(loaded == 255)


def test_load_mask_not_found(tmp_path: Path):
    """Test load_mask returns None for nonexistent camera."""
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()

    loaded = load_mask(mask_dir, "nonexistent", (640, 480))

    assert loaded is None


def test_load_mask_size_mismatch(tmp_path: Path, caplog):
    """Test load_mask returns None on size mismatch with warning."""
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()

    # Create a 100x100 mask
    mask_path = mask_dir / "cam0.png"
    mask_img = np.full((100, 100), 255, dtype=np.uint8)
    cv2.imwrite(str(mask_path), mask_img)

    # Try to load with expected size 640x480
    with caplog.at_level(logging.WARNING):
        loaded = load_mask(mask_dir, "cam0", (640, 480))

    assert loaded is None
    assert "Mask size mismatch" in caplog.text


def test_load_all_masks_none_dir():
    """Test load_all_masks with None mask_dir returns empty dict."""
    cameras = {
        "cam0": CameraData(
            name="cam0",
            K=torch.eye(3),
            R=torch.eye(3),
            t=torch.zeros(3),
            dist_coeffs=torch.zeros(5),
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        ),
    }

    masks = load_all_masks(None, cameras)

    assert masks == {}


def test_load_all_masks_missing_dir(tmp_path: Path, caplog):
    """Test load_all_masks with nonexistent directory returns empty dict with warning."""
    cameras = {
        "cam0": CameraData(
            name="cam0",
            K=torch.eye(3),
            R=torch.eye(3),
            t=torch.zeros(3),
            dist_coeffs=torch.zeros(5),
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        ),
    }

    nonexistent_dir = tmp_path / "nonexistent"

    with caplog.at_level(logging.WARNING):
        masks = load_all_masks(nonexistent_dir, cameras)

    assert masks == {}
    assert "does not exist" in caplog.text


def test_load_all_masks_partial(tmp_path: Path):
    """Test load_all_masks with masks for 2 of 3 cameras."""
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()

    # Create cameras
    cameras = {
        "cam0": CameraData(
            name="cam0",
            K=torch.eye(3),
            R=torch.eye(3),
            t=torch.zeros(3),
            dist_coeffs=torch.zeros(5),
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        ),
        "cam1": CameraData(
            name="cam1",
            K=torch.eye(3),
            R=torch.eye(3),
            t=torch.zeros(3),
            dist_coeffs=torch.zeros(5),
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        ),
        "cam2": CameraData(
            name="cam2",
            K=torch.eye(3),
            R=torch.eye(3),
            t=torch.zeros(3),
            dist_coeffs=torch.zeros(5),
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        ),
    }

    # Create masks for cam0 and cam1 only
    for cam_name in ["cam0", "cam1"]:
        mask_path = mask_dir / f"{cam_name}.png"
        mask_img = np.full((480, 640), 255, dtype=np.uint8)
        cv2.imwrite(str(mask_path), mask_img)

    masks = load_all_masks(mask_dir, cameras)

    assert len(masks) == 2
    assert "cam0" in masks
    assert "cam1" in masks
    assert "cam2" not in masks


def test_apply_mask_to_features_filters():
    """Test apply_mask_to_features filters keypoints correctly."""
    # Create a mask: left half (u < 320) is 0, right half is 255
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[:, 320:] = 255

    # Create features with keypoints spanning both halves
    keypoints = torch.tensor(
        [
            [100.0, 240.0],  # Left half (u=100) - should be filtered
            [400.0, 240.0],  # Right half (u=400) - should survive
            [150.0, 100.0],  # Left half (u=150) - should be filtered
            [500.0, 300.0],  # Right half (u=500) - should survive
        ]
    )
    descriptors = torch.randn(4, 256)
    scores = torch.ones(4)

    features = {
        "keypoints": keypoints,
        "descriptors": descriptors,
        "scores": scores,
    }

    filtered = apply_mask_to_features(features, mask)

    # Should keep only keypoints 1 and 3 (indices from original)
    assert filtered["keypoints"].shape[0] == 2
    assert filtered["descriptors"].shape[0] == 2
    assert filtered["scores"].shape[0] == 2

    # Check that the right keypoints survived
    assert torch.allclose(filtered["keypoints"][0], torch.tensor([400.0, 240.0]))
    assert torch.allclose(filtered["keypoints"][1], torch.tensor([500.0, 300.0]))


def test_apply_mask_to_features_all_masked():
    """Test apply_mask_to_features when all keypoints are masked."""
    # All-zero mask
    mask = np.zeros((480, 640), dtype=np.uint8)

    keypoints = torch.tensor([[100.0, 240.0], [400.0, 240.0]])
    descriptors = torch.randn(2, 256)
    scores = torch.ones(2)

    features = {
        "keypoints": keypoints,
        "descriptors": descriptors,
        "scores": scores,
    }

    filtered = apply_mask_to_features(features, mask)

    # All should be filtered out
    assert filtered["keypoints"].shape[0] == 0
    assert filtered["descriptors"].shape[0] == 0
    assert filtered["scores"].shape[0] == 0


def test_apply_mask_to_features_none_masked():
    """Test apply_mask_to_features when no keypoints are masked."""
    # All-255 mask (all valid)
    mask = np.full((480, 640), 255, dtype=np.uint8)

    keypoints = torch.tensor([[100.0, 240.0], [400.0, 240.0], [500.0, 300.0]])
    descriptors = torch.randn(3, 256)
    scores = torch.ones(3)

    features = {
        "keypoints": keypoints,
        "descriptors": descriptors,
        "scores": scores,
    }

    filtered = apply_mask_to_features(features, mask)

    # All should survive
    assert filtered["keypoints"].shape[0] == 3
    assert filtered["descriptors"].shape[0] == 3
    assert filtered["scores"].shape[0] == 3
    assert torch.allclose(filtered["keypoints"], keypoints)


def test_apply_mask_to_depth():
    """Test apply_mask_to_depth sets excluded pixels to NaN/0."""
    # Create a 10x20 depth map and confidence map
    depth_map = torch.rand(10, 20) * 0.5 + 0.3  # Random depths in [0.3, 0.8]
    confidence = torch.rand(10, 20) * 0.5 + 0.5  # Random confidence in [0.5, 1.0]

    # Create a mask with a 5x10 excluded region (top-left corner)
    mask = np.full((10, 20), 255, dtype=np.uint8)
    mask[:5, :10] = 0  # Exclude top-left 5x10 region

    masked_depth, masked_conf = apply_mask_to_depth(depth_map, confidence, mask)

    # Check excluded region
    assert torch.all(torch.isnan(masked_depth[:5, :10]))
    assert torch.all(masked_conf[:5, :10] == 0.0)

    # Check valid region (bottom-right)
    assert torch.allclose(masked_depth[5:, 10:], depth_map[5:, 10:])
    assert torch.allclose(masked_conf[5:, 10:], confidence[5:, 10:])


def test_apply_mask_to_depth_no_inplace():
    """Test apply_mask_to_depth does not modify inputs in-place."""
    depth_map = torch.rand(10, 20) * 0.5 + 0.3
    confidence = torch.rand(10, 20) * 0.5 + 0.5

    # Keep copies of originals
    depth_orig = depth_map.clone()
    conf_orig = confidence.clone()

    # Create a mask that excludes the top half
    mask = np.full((10, 20), 255, dtype=np.uint8)
    mask[:5, :] = 0

    masked_depth, masked_conf = apply_mask_to_depth(depth_map, confidence, mask)

    # Originals should be unchanged
    assert torch.allclose(depth_map, depth_orig)
    assert torch.allclose(confidence, conf_orig)

    # Outputs should differ in the excluded region
    assert torch.all(torch.isnan(masked_depth[:5, :]))
    assert torch.all(masked_conf[:5, :] == 0.0)
    assert not torch.all(torch.isnan(depth_orig[:5, :]))
