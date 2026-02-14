"""Tests for best-view color selection and color normalization."""

import numpy as np
import pytest
import torch

from aquamvs.coloring import best_view_colors, normalize_colors


class MockProjectionModel:
    """Mock projection model for testing."""

    def __init__(self, image_size, valid_mask=None):
        """Initialize mock model.

        Args:
            image_size: (width, height) tuple
            valid_mask: Optional callable that takes points (N, 3) and returns bool (N,)
        """
        self.image_size = image_size
        self.valid_mask = valid_mask
        self.K = torch.eye(3)  # Dummy K for device inference

    def project(self, points):
        """Project points to pixels.

        Simple projection: (u, v) = (x * 100 + width/2, y * 100 + height/2)
        """
        W, H = self.image_size
        pixels = torch.zeros(points.shape[0], 2, dtype=torch.float32)
        pixels[:, 0] = points[:, 0] * 100 + W / 2
        pixels[:, 1] = points[:, 1] * 100 + H / 2

        # Check bounds
        valid = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < W)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < H)
        )

        # Apply custom validity mask if provided
        if self.valid_mask is not None:
            valid = valid & self.valid_mask(points)

        return pixels, valid


def test_frontal_preferred():
    """Test that frontal camera (aligned with normal) is preferred over oblique."""
    # Horizontal surface at origin with normal pointing up (-Z in our coords)
    points = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    normals = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)

    # Camera 1: directly above (view direction aligned with normal)
    cam1_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # Camera 2: at 45 degrees (less aligned)
    cam2_center = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # Create mock models
    image_size = (640, 480)
    models = {
        "cam1": MockProjectionModel(image_size),
        "cam2": MockProjectionModel(image_size),
    }

    # Create test images with distinct colors
    img1 = torch.full(
        (480, 640, 3), 255, dtype=torch.uint8
    )  # White (B=255, G=255, R=255)
    img1[:, :, :] = torch.tensor([0, 0, 255], dtype=torch.uint8)  # Red in BGR

    img2 = torch.full((480, 640, 3), 255, dtype=torch.uint8)
    img2[:, :, :] = torch.tensor([255, 0, 0], dtype=torch.uint8)  # Blue in BGR

    images = {"cam1": img1, "cam2": img2}
    centers = {"cam1": cam1_center, "cam2": cam2_center}

    # Get colors
    colors = best_view_colors(points, normals, models, images, centers)

    # Should pick cam1 (red in RGB) because it's more aligned
    # cam1 view direction: (0, 0, 1) - (0, 0, 0) = (0, 0, 1), normalized = (0, 0, 1)
    # alignment: |dot((0, 0, 1), (0, 0, -1))| = 1.0
    # cam2 view direction: (0, 0, 1) - (1, 0, 0) = (-1, 0, 1), normalized ≈ (-0.707, 0, 0.707)
    # alignment: |dot((-0.707, 0, 0.707), (0, 0, -1))| ≈ 0.707

    assert colors.shape == (1, 3)
    np.testing.assert_allclose(colors[0], [1.0, 0.0, 0.0], atol=0.01)  # Red


def test_invalid_projection_skipped():
    """Test that camera with invalid projection is skipped."""
    points = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    normals = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)

    cam1_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    cam2_center = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    image_size = (640, 480)

    # cam1 always returns invalid
    def always_invalid(pts):
        return torch.zeros(pts.shape[0], dtype=torch.bool)

    models = {
        "cam1": MockProjectionModel(image_size, valid_mask=always_invalid),
        "cam2": MockProjectionModel(image_size),
    }

    img1 = torch.full((480, 640, 3), 255, dtype=torch.uint8)
    img1[:, :, :] = torch.tensor([0, 0, 255], dtype=torch.uint8)  # Red

    img2 = torch.full((480, 640, 3), 255, dtype=torch.uint8)
    img2[:, :, :] = torch.tensor([255, 0, 0], dtype=torch.uint8)  # Blue

    images = {"cam1": img1, "cam2": img2}
    centers = {"cam1": cam1_center, "cam2": cam2_center}

    colors = best_view_colors(points, normals, models, images, centers)

    # Should pick cam2 (blue) because cam1 is invalid
    assert colors.shape == (1, 3)
    np.testing.assert_allclose(colors[0], [0.0, 0.0, 1.0], atol=0.01)  # Blue


def test_all_invalid_returns_gray():
    """Test that points with no valid projection get default gray."""
    points = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    normals = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)

    cam1_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    image_size = (640, 480)

    # All cameras return invalid
    def always_invalid(pts):
        return torch.zeros(pts.shape[0], dtype=torch.bool)

    models = {"cam1": MockProjectionModel(image_size, valid_mask=always_invalid)}

    img1 = torch.full((480, 640, 3), 255, dtype=torch.uint8)
    img1[:, :, :] = torch.tensor([0, 0, 255], dtype=torch.uint8)

    images = {"cam1": img1}
    centers = {"cam1": cam1_center}

    colors = best_view_colors(points, normals, models, images, centers)

    # Should be gray (0.5, 0.5, 0.5)
    assert colors.shape == (1, 3)
    np.testing.assert_allclose(colors[0], [0.5, 0.5, 0.5], atol=1e-6)


def test_vectorized_multiple_points():
    """Test that function handles multiple points correctly (batch behavior)."""
    # Three points in a row
    points = np.array(
        [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]], dtype=np.float64
    )
    normals = np.array(
        [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=np.float64
    )

    cam1_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    image_size = (640, 480)
    models = {"cam1": MockProjectionModel(image_size)}

    # Create gradient image: red on left, blue on right
    img1 = torch.zeros((480, 640, 3), dtype=torch.uint8)
    for u in range(640):
        r = int(255 * u / 640)
        b = 255 - r
        img1[:, u, :] = torch.tensor([b, 0, r], dtype=torch.uint8)

    images = {"cam1": img1}
    centers = {"cam1": cam1_center}

    colors = best_view_colors(points, normals, models, images, centers)

    # All points should get colors, with different values
    assert colors.shape == (3, 3)
    assert not np.allclose(colors[0], colors[1])  # Different colors
    assert not np.allclose(colors[1], colors[2])  # Different colors


def test_single_camera():
    """Test that with only one camera, its color is always used for valid projections."""
    points = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    normals = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)

    cam1_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    image_size = (640, 480)
    models = {"cam1": MockProjectionModel(image_size)}

    img1 = torch.full((480, 640, 3), 255, dtype=torch.uint8)
    img1[:, :, :] = torch.tensor([0, 255, 0], dtype=torch.uint8)  # Green in BGR

    images = {"cam1": img1}
    centers = {"cam1": cam1_center}

    colors = best_view_colors(points, normals, models, images, centers)

    # Should get green (0, 1, 0) in RGB
    assert colors.shape == (1, 3)
    np.testing.assert_allclose(colors[0], [0.0, 1.0, 0.0], atol=0.01)


def test_empty_input():
    """Test that empty input returns empty output."""
    points = np.zeros((0, 3), dtype=np.float64)
    normals = np.zeros((0, 3), dtype=np.float64)

    models = {}
    images = {}
    centers = {}

    colors = best_view_colors(points, normals, models, images, centers)

    assert colors.shape == (0, 3)


def test_out_of_bounds_projection_skipped():
    """Test that points projecting outside image bounds are skipped."""
    # Point very far from camera center, will project outside bounds
    points = np.array([[10.0, 10.0, 1.0]], dtype=np.float64)
    normals = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)

    cam1_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    image_size = (640, 480)
    models = {"cam1": MockProjectionModel(image_size)}

    img1 = torch.full((480, 640, 3), 255, dtype=torch.uint8)
    img1[:, :, :] = torch.tensor([0, 0, 255], dtype=torch.uint8)

    images = {"cam1": img1}
    centers = {"cam1": cam1_center}

    colors = best_view_colors(points, normals, models, images, centers)

    # Should get gray because projection is out of bounds
    assert colors.shape == (1, 3)
    np.testing.assert_allclose(colors[0], [0.5, 0.5, 0.5], atol=1e-6)


def test_missing_camera_in_images():
    """Test that camera in models but not in images is skipped gracefully."""
    points = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    normals = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)

    cam1_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    cam2_center = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    image_size = (640, 480)
    models = {
        "cam1": MockProjectionModel(image_size),
        "cam2": MockProjectionModel(image_size),  # No image for cam2
    }

    img1 = torch.full((480, 640, 3), 255, dtype=torch.uint8)
    img1[:, :, :] = torch.tensor([0, 0, 255], dtype=torch.uint8)

    images = {"cam1": img1}  # Only cam1 has image
    centers = {"cam1": cam1_center, "cam2": cam2_center}

    colors = best_view_colors(points, normals, models, images, centers)

    # Should use cam1's color (red)
    assert colors.shape == (1, 3)
    np.testing.assert_allclose(colors[0], [1.0, 0.0, 0.0], atol=0.01)


def test_missing_camera_in_centers():
    """Test that camera in models but not in centers is skipped gracefully."""
    points = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    normals = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)

    cam1_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    image_size = (640, 480)
    models = {
        "cam1": MockProjectionModel(image_size),
        "cam2": MockProjectionModel(image_size),  # No center for cam2
    }

    img1 = torch.full((480, 640, 3), 255, dtype=torch.uint8)
    img1[:, :, :] = torch.tensor([0, 0, 255], dtype=torch.uint8)

    img2 = torch.full((480, 640, 3), 255, dtype=torch.uint8)
    img2[:, :, :] = torch.tensor([255, 0, 0], dtype=torch.uint8)

    images = {"cam1": img1, "cam2": img2}
    centers = {"cam1": cam1_center}  # Only cam1 has center

    colors = best_view_colors(points, normals, models, images, centers)

    # Should use cam1's color (red)
    assert colors.shape == (1, 3)
    np.testing.assert_allclose(colors[0], [1.0, 0.0, 0.0], atol=0.01)


class TestNormalizeColors:
    """Tests for color normalization across cameras."""

    def test_gain_normalization_equalizes_means(self):
        """Test that gain normalization equalizes per-channel means."""
        # Two cameras with different brightness
        img1 = np.full((100, 100, 3), 100, dtype=np.uint8)  # Mean 100 in all channels
        img2 = np.full((100, 100, 3), 200, dtype=np.uint8)  # Mean 200 in all channels

        images = {"cam1": img1, "cam2": img2}

        # Normalize with gain method
        normalized = normalize_colors(images, method="gain")

        # Both should now have the same mean (target = average of 100 and 200 = 150)
        mean1 = normalized["cam1"].mean(axis=(0, 1))
        mean2 = normalized["cam2"].mean(axis=(0, 1))

        # Both should be close to 150 in all channels
        np.testing.assert_allclose(mean1, 150, atol=1)
        np.testing.assert_allclose(mean2, 150, atol=1)

    def test_gain_normalization_preserves_relative_color(self):
        """Test that gain normalization preserves color ratios."""
        # Camera with known color ratio: R:G:B = 2:1:1 (in BGR: B:G:R = 1:1:2)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 100  # B
        img[:, :, 1] = 100  # G
        img[:, :, 2] = 200  # R

        images = {"cam1": img}

        # Normalize (should be almost a no-op since only one camera, but floating point may differ)
        normalized = normalize_colors(images, method="gain")

        # The image should be nearly unchanged since there's only one camera
        # Allow for 1 unit difference due to floating point rounding
        np.testing.assert_allclose(normalized["cam1"], img, atol=1)

    def test_histogram_normalization_equalizes_cdfs(self):
        """Test that histogram normalization makes CDFs similar."""
        # Camera 1: bright image (mean 200)
        img1 = np.full((100, 100, 3), 200, dtype=np.uint8)
        # Camera 2: dark image (mean 100)
        img2 = np.full((100, 100, 3), 100, dtype=np.uint8)

        images = {"cam1": img1, "cam2": img2}

        normalized = normalize_colors(images, method="histogram")

        # After histogram normalization, the images should be closer in intensity
        mean1 = normalized["cam1"].mean()
        mean2 = normalized["cam2"].mean()

        # The means should be closer than the original 200 vs 100
        diff = abs(mean1 - mean2)
        original_diff = 100
        assert diff < original_diff or abs(mean1 - mean2) < 5  # Allow some variance

    def test_normalization_does_not_modify_input(self):
        """Test that original images dict is not mutated."""
        img1 = np.full((50, 50, 3), 100, dtype=np.uint8)
        img2 = np.full((50, 50, 3), 200, dtype=np.uint8)

        images = {"cam1": img1.copy(), "cam2": img2.copy()}
        original_sum1 = images["cam1"].sum()
        original_sum2 = images["cam2"].sum()

        # Normalize
        normalized = normalize_colors(images, method="gain")

        # Original should be unchanged
        assert images["cam1"].sum() == original_sum1
        assert images["cam2"].sum() == original_sum2

        # Result should be different
        assert not np.array_equal(normalized["cam1"], images["cam1"])

    def test_single_camera_noop(self):
        """Test that with only one camera, output is nearly identical to input."""
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        images = {"cam1": img}

        normalized = normalize_colors(images, method="gain")

        # With one camera, gain normalization should be nearly identity (floating point may differ)
        # Allow for 1 unit difference due to floating point rounding
        np.testing.assert_allclose(normalized["cam1"], img, atol=1)

    def test_normalization_clamps_to_uint8(self):
        """Test that output stays in [0, 255] uint8 range."""
        # Very dark image
        img1 = np.full((50, 50, 3), 10, dtype=np.uint8)
        # Very bright image
        img2 = np.full((50, 50, 3), 245, dtype=np.uint8)

        images = {"cam1": img1, "cam2": img2}

        normalized = normalize_colors(images, method="gain")

        # Both should be uint8 and in valid range
        assert normalized["cam1"].dtype == np.uint8
        assert normalized["cam2"].dtype == np.uint8
        assert normalized["cam1"].min() >= 0
        assert normalized["cam1"].max() <= 255
        assert normalized["cam2"].min() >= 0
        assert normalized["cam2"].max() <= 255

    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        images = {"cam1": np.ones((50, 50, 3), dtype=np.uint8)}

        with pytest.raises(ValueError, match="Invalid normalization method"):
            normalize_colors(images, method="invalid")

    def test_empty_images_dict(self):
        """Test that empty dict returns empty dict."""
        images = {}
        result = normalize_colors(images, method="gain")
        assert result == {}

    def test_gain_default_method(self):
        """Test that gain is the default method."""
        img1 = np.full((50, 50, 3), 100, dtype=np.uint8)
        img2 = np.full((50, 50, 3), 200, dtype=np.uint8)

        images = {"cam1": img1, "cam2": img2}

        # Call without specifying method
        normalized = normalize_colors(images)

        # Should apply gain normalization
        mean1 = normalized["cam1"].mean(axis=(0, 1))
        mean2 = normalized["cam2"].mean(axis=(0, 1))
        # Target mean is 150. Allow some tolerance for rounding.
        np.testing.assert_allclose(mean1, 150, atol=2)
        np.testing.assert_allclose(mean2, 150, atol=2)
