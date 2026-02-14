"""Tests for feature and match overlay rendering."""

import sys

import cv2
import numpy as np

from aquamvs.visualization.features import (
    render_all_features,
    render_keypoints,
    render_matches,
    render_sparse_overlay,
)


def test_render_keypoints_basic(tmp_path):
    """Test basic keypoint rendering without scores."""
    # Create synthetic image
    image = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)

    # Create 10 random keypoints
    keypoints = np.random.rand(10, 2).astype(np.float32) * [160, 120]

    # Render
    output_path = tmp_path / "keypoints.png"
    result = render_keypoints(image, keypoints, output_path=output_path)

    # Check output
    assert result.shape == (120, 160, 3)
    assert result.dtype == np.uint8
    assert output_path.exists()

    # Verify saved image is loadable
    loaded = cv2.imread(str(output_path))
    assert loaded is not None
    assert loaded.shape == (120, 160, 3)


def test_render_keypoints_with_scores(tmp_path):
    """Test keypoint rendering with score-based coloring."""
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    keypoints = np.array([[80, 60], [40, 30], [120, 90]], dtype=np.float32)
    scores = np.array([0.9, 0.5, 0.1], dtype=np.float32)

    output_path = tmp_path / "keypoints_scored.png"
    result = render_keypoints(image, keypoints, scores, output_path)

    # Check output
    assert result.shape == (120, 160, 3)
    assert result.dtype == np.uint8
    assert output_path.exists()

    # Verify that pixels were modified (markers were drawn)
    assert not np.array_equal(result, image)


def test_render_keypoints_empty():
    """Test keypoint rendering with no keypoints."""
    image = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    keypoints = np.zeros((0, 2), dtype=np.float32)

    result = render_keypoints(image, keypoints)

    # Should return copy of original image
    assert result.shape == image.shape
    assert np.array_equal(result, image)


def test_render_matches_basic(tmp_path):
    """Test basic match rendering."""
    # Create two synthetic images
    image_ref = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    image_src = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)

    # Create 5 matched point pairs
    ref_keypoints = np.random.rand(5, 2).astype(np.float32) * [160, 120]
    src_keypoints = np.random.rand(5, 2).astype(np.float32) * [160, 120]

    # Render
    output_path = tmp_path / "matches.png"
    result = render_matches(
        image_ref, image_src, ref_keypoints, src_keypoints, output_path=output_path
    )

    # Check output - should be side-by-side
    assert result.shape == (120, 320, 3)  # width doubled
    assert result.dtype == np.uint8
    assert output_path.exists()

    # Verify saved image is loadable
    loaded = cv2.imread(str(output_path))
    assert loaded is not None
    assert loaded.shape == (120, 320, 3)


def test_render_matches_with_scores(tmp_path):
    """Test match rendering with confidence scores."""
    image_ref = np.zeros((120, 160, 3), dtype=np.uint8)
    image_src = np.zeros((120, 160, 3), dtype=np.uint8)

    ref_keypoints = np.array([[80, 60], [40, 30]], dtype=np.float32)
    src_keypoints = np.array([[85, 65], [45, 35]], dtype=np.float32)
    scores = np.array([0.9, 0.3], dtype=np.float32)

    output_path = tmp_path / "matches_scored.png"
    result = render_matches(
        image_ref, image_src, ref_keypoints, src_keypoints, scores, output_path
    )

    # Check output
    assert result.shape == (120, 320, 3)
    assert result.dtype == np.uint8
    assert output_path.exists()

    # Verify that pixels were modified (lines and markers were drawn)
    assert result.max() > 0  # Some non-zero pixels from drawing


def test_render_matches_empty():
    """Test match rendering with no matches."""
    image_ref = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    image_src = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)

    ref_keypoints = np.zeros((0, 2), dtype=np.float32)
    src_keypoints = np.zeros((0, 2), dtype=np.float32)

    result = render_matches(image_ref, image_src, ref_keypoints, src_keypoints)

    # Should return side-by-side canvas with original images
    assert result.shape == (120, 320, 3)
    assert np.array_equal(result[:, :160], image_ref)
    assert np.array_equal(result[:, 160:], image_src)


def test_render_matches_different_heights(tmp_path):
    """Test match rendering with images of different heights."""
    image_ref = np.random.randint(0, 256, (100, 160, 3), dtype=np.uint8)
    image_src = np.random.randint(0, 256, (150, 160, 3), dtype=np.uint8)

    ref_keypoints = np.array([[80, 50]], dtype=np.float32)
    src_keypoints = np.array([[85, 75]], dtype=np.float32)

    result = render_matches(image_ref, image_src, ref_keypoints, src_keypoints)

    # Height should be max of both images
    assert result.shape == (150, 320, 3)


def test_render_sparse_overlay_basic(tmp_path):
    """Test sparse overlay rendering without errors."""
    image = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    projected_points = np.random.rand(8, 2).astype(np.float32) * [160, 120]

    output_path = tmp_path / "sparse_overlay.png"
    result = render_sparse_overlay(image, projected_points, output_path=output_path)

    # Check output
    assert result.shape == (120, 160, 3)
    assert result.dtype == np.uint8
    assert output_path.exists()

    # Verify saved image is loadable
    loaded = cv2.imread(str(output_path))
    assert loaded is not None
    assert loaded.shape == (120, 160, 3)


def test_render_sparse_overlay_with_errors(tmp_path):
    """Test sparse overlay rendering with reprojection errors."""
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    projected_points = np.array([[80, 60], [40, 30], [120, 90]], dtype=np.float32)
    errors = np.array([0.5, 2.0, 8.0], dtype=np.float32)  # Low, medium, high errors

    output_path = tmp_path / "sparse_overlay_errors.png"
    result = render_sparse_overlay(
        image, projected_points, errors, output_path, error_threshold=5.0
    )

    # Check output
    assert result.shape == (120, 160, 3)
    assert result.dtype == np.uint8
    assert output_path.exists()

    # Verify that pixels were modified (markers were drawn)
    assert result.max() > 0


def test_render_sparse_overlay_empty():
    """Test sparse overlay rendering with no points."""
    image = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    projected_points = np.zeros((0, 2), dtype=np.float32)

    result = render_sparse_overlay(image, projected_points)

    # Should return copy of original image
    assert result.shape == image.shape
    assert np.array_equal(result, image)


def test_render_all_features_basic(tmp_path):
    """Test rendering all features for multiple cameras."""
    # Create 2 camera images
    images = {
        "cam1": np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8),
        "cam2": np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8),
    }

    # Create features for each camera
    all_features = {
        "cam1": {
            "keypoints": np.random.rand(10, 2).astype(np.float32) * [160, 120],
            "scores": np.random.rand(10).astype(np.float32),
        },
        "cam2": {
            "keypoints": np.random.rand(8, 2).astype(np.float32) * [160, 120],
            "scores": np.random.rand(8).astype(np.float32),
        },
    }

    # Create matches for one pair
    all_matches = {
        ("cam1", "cam2"): {
            "ref_keypoints": np.random.rand(5, 2).astype(np.float32) * [160, 120],
            "src_keypoints": np.random.rand(5, 2).astype(np.float32) * [160, 120],
            "scores": np.random.rand(5).astype(np.float32),
        }
    }

    # Render
    render_all_features(images, all_features, all_matches, output_dir=tmp_path)

    # Verify expected files were created
    assert (tmp_path / "sparse_cam1.png").exists()
    assert (tmp_path / "sparse_cam2.png").exists()
    assert (tmp_path / "matches_cam1_cam2.png").exists()

    # Verify files are loadable
    img1 = cv2.imread(str(tmp_path / "sparse_cam1.png"))
    img2 = cv2.imread(str(tmp_path / "sparse_cam2.png"))
    img_match = cv2.imread(str(tmp_path / "matches_cam1_cam2.png"))

    assert img1 is not None and img1.shape == (120, 160, 3)
    assert img2 is not None and img2.shape == (120, 160, 3)
    assert img_match is not None and img_match.shape == (120, 320, 3)


def test_render_all_features_missing_data(tmp_path):
    """Test render_all_features with missing features or images."""
    images = {
        "cam1": np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8),
        "cam2": np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8),
    }

    # Features only for cam1
    all_features = {
        "cam1": {
            "keypoints": np.random.rand(10, 2).astype(np.float32) * [160, 120],
            "scores": np.random.rand(10).astype(np.float32),
        },
    }

    # Matches for a pair where one camera has no features
    all_matches = {
        ("cam1", "cam3"): {  # cam3 doesn't exist in images
            "ref_keypoints": np.random.rand(5, 2).astype(np.float32) * [160, 120],
            "src_keypoints": np.random.rand(5, 2).astype(np.float32) * [160, 120],
            "scores": np.random.rand(5).astype(np.float32),
        }
    }

    # Should not crash
    render_all_features(images, all_features, all_matches, output_dir=tmp_path)

    # Only cam1 sparse should be created
    assert (tmp_path / "sparse_cam1.png").exists()
    assert not (tmp_path / "sparse_cam2.png").exists()  # No features for cam2
    assert not (tmp_path / "matches_cam1_cam3.png").exists()  # cam3 doesn't exist


def test_render_all_features_creates_output_dir(tmp_path):
    """Test that render_all_features creates output directory if it doesn't exist."""
    output_dir = tmp_path / "subdir" / "output"
    assert not output_dir.exists()

    images = {
        "cam1": np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8),
    }
    all_features = {
        "cam1": {
            "keypoints": np.random.rand(5, 2).astype(np.float32) * [160, 120],
        },
    }
    all_matches = {}

    render_all_features(images, all_features, all_matches, output_dir=output_dir)

    assert output_dir.exists()
    assert (output_dir / "sparse_cam1.png").exists()


def test_no_torch_dependency():
    """Verify that the module does not import torch."""
    # Check if torch was imported when we imported the features module
    assert (
        "torch" not in sys.modules
        or "aquamvs.visualization.features" not in sys.modules.get("torch", []).__dict__
    )

    # More direct check: verify torch is not in the features module namespace
    import aquamvs.visualization.features as features_mod

    # Get all imported modules used by this module
    module_globals = vars(features_mod)
    imported_modules = [
        v.__name__
        for v in module_globals.values()
        if hasattr(v, "__name__") and hasattr(v, "__file__")
    ]

    assert "torch" not in imported_modules


def test_marker_customization(tmp_path):
    """Test that marker_size parameter works correctly."""
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    keypoints = np.array([[80, 60]], dtype=np.float32)

    # Render with different marker sizes
    result_small = render_keypoints(image, keypoints, marker_size=1)
    result_large = render_keypoints(image, keypoints, marker_size=10)

    # Larger marker should modify more pixels
    assert result_large.sum() > result_small.sum()


def test_line_thickness_customization(tmp_path):
    """Test that line_thickness parameter works correctly."""
    image_ref = np.zeros((120, 160, 3), dtype=np.uint8)
    image_src = np.zeros((120, 160, 3), dtype=np.uint8)

    ref_keypoints = np.array([[10, 10]], dtype=np.float32)
    src_keypoints = np.array([[150, 110]], dtype=np.float32)

    # Render with different line thicknesses
    result_thin = render_matches(
        image_ref, image_src, ref_keypoints, src_keypoints, line_thickness=1
    )
    result_thick = render_matches(
        image_ref, image_src, ref_keypoints, src_keypoints, line_thickness=3
    )

    # Thicker line should modify more pixels
    assert result_thick.sum() > result_thin.sum()
