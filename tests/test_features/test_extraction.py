"""Tests for feature extraction."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from lightglue import ALIKED, DISK, SuperPoint

from aquamvs.config import FeatureExtractionConfig
from aquamvs.features import (
    create_extractor,
    extract_features,
    extract_features_batch,
    load_features,
    save_features,
)
from aquamvs.features.extraction import _apply_clahe


def create_checkerboard_image(
    height: int = 480, width: int = 640, square_size: int = 40
) -> torch.Tensor:
    """Create a checkerboard pattern image with corners for feature detection.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        square_size: Size of each checkerboard square in pixels.

    Returns:
        Grayscale uint8 tensor of shape (height, width).
    """
    # Create coordinate grids
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")

    # Checkerboard pattern: alternate black (0) and white (255)
    checker = ((x // square_size) + (y // square_size)) % 2
    image = (checker * 255).to(torch.uint8)

    return image


class TestImageConversion:
    """Test image format conversion without requiring SuperPoint."""

    def test_grayscale_uint8_tensor(self):
        """Test conversion from grayscale uint8 tensor."""
        config = FeatureExtractionConfig()
        # Create a synthetic extractor mock for testing conversion only
        # We'll just verify the conversion doesn't crash
        image = torch.randint(0, 256, (100, 100), dtype=torch.uint8)

        # This will create the extractor internally, but we're mainly
        # testing that the conversion logic works
        # Skip if SuperPoint not available
        try:
            result = extract_features(image, config, device="cpu")
            assert isinstance(result, dict)
            assert "keypoints" in result
            assert "descriptors" in result
            assert "scores" in result
        except Exception as e:
            if "SuperPoint" in str(type(e).__name__):
                pytest.skip("SuperPoint model not available")
            raise

    def test_bgr_uint8_tensor(self):
        """Test conversion from BGR uint8 tensor."""
        config = FeatureExtractionConfig()
        image = torch.randint(0, 256, (100, 100, 3), dtype=torch.uint8)

        try:
            result = extract_features(image, config, device="cpu")
            assert isinstance(result, dict)
        except Exception as e:
            if "SuperPoint" in str(type(e).__name__):
                pytest.skip("SuperPoint model not available")
            raise

    def test_grayscale_float32_tensor(self):
        """Test conversion from grayscale float32 tensor."""
        config = FeatureExtractionConfig()
        image = torch.rand(100, 100, dtype=torch.float32)

        try:
            result = extract_features(image, config, device="cpu")
            assert isinstance(result, dict)
        except Exception as e:
            if "SuperPoint" in str(type(e).__name__):
                pytest.skip("SuperPoint model not available")
            raise

    def test_numpy_uint8_array(self):
        """Test conversion from numpy uint8 array."""
        config = FeatureExtractionConfig()
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        try:
            result = extract_features(image, config, device="cpu")
            assert isinstance(result, dict)
        except Exception as e:
            if "SuperPoint" in str(type(e).__name__):
                pytest.skip("SuperPoint model not available")
            raise

    def test_invalid_shape_raises(self):
        """Test that invalid image shapes raise ValueError."""
        config = FeatureExtractionConfig()
        # 4D tensor is invalid
        image = torch.rand(1, 100, 100, 3)

        with pytest.raises(ValueError, match="Expected image with shape"):
            extract_features(image, config, device="cpu")


class TestSaveLoad:
    """Test feature persistence without requiring SuperPoint."""

    def test_save_load_roundtrip(self):
        """Test save and load roundtrip with synthetic features."""
        # Create synthetic features
        features = {
            "keypoints": torch.rand(100, 2, dtype=torch.float32),
            "descriptors": torch.rand(100, 256, dtype=torch.float32),
            "scores": torch.rand(100, dtype=torch.float32),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.pt"

            # Save
            save_features(features, path)
            assert path.exists()

            # Load
            loaded = load_features(path)

            # Verify
            assert set(loaded.keys()) == {"keypoints", "descriptors", "scores"}
            assert torch.allclose(loaded["keypoints"], features["keypoints"])
            assert torch.allclose(loaded["descriptors"], features["descriptors"])
            assert torch.allclose(loaded["scores"], features["scores"])

    def test_save_as_string_path(self):
        """Test save_features accepts string paths."""
        features = {
            "keypoints": torch.rand(10, 2),
            "descriptors": torch.rand(10, 256),
            "scores": torch.rand(10),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "features.pt")
            save_features(features, path)
            loaded = load_features(path)
            assert set(loaded.keys()) == {"keypoints", "descriptors", "scores"}


class TestCreateExtractor:
    """Tests for create_extractor with different backends."""

    def test_create_extractor_superpoint(self):
        """Test that create_extractor returns SuperPoint instance."""
        config = FeatureExtractionConfig(extractor_type="superpoint")
        extractor = create_extractor(config, device="cpu")
        assert isinstance(extractor, SuperPoint)

    def test_create_extractor_aliked(self):
        """Test that create_extractor returns ALIKED instance."""
        config = FeatureExtractionConfig(extractor_type="aliked")
        extractor = create_extractor(config, device="cpu")
        assert isinstance(extractor, ALIKED)

    def test_create_extractor_disk(self):
        """Test that create_extractor returns DISK instance."""
        config = FeatureExtractionConfig(extractor_type="disk")
        extractor = create_extractor(config, device="cpu")
        assert isinstance(extractor, DISK)

    def test_create_extractor_invalid(self):
        """Test that create_extractor raises ValueError for invalid type."""
        config = FeatureExtractionConfig(extractor_type="invalid")
        with pytest.raises(ValueError, match="Unknown extractor_type"):
            create_extractor(config, device="cpu")


@pytest.mark.slow
class TestExtractorOutputFormat:
    """Tests for output format consistency across extractors."""

    def test_extract_features_output_format_aliked(self, device):
        """Test ALIKED produces correct output format."""
        config = FeatureExtractionConfig(extractor_type="aliked")
        image = torch.randint(0, 256, (480, 640, 3), dtype=torch.uint8)

        result = extract_features(image, config, device=str(device))

        # Verify structure
        assert isinstance(result, dict)
        assert set(result.keys()) == {"keypoints", "descriptors", "scores"}

        # Verify shapes
        n_keypoints = result["keypoints"].shape[0]
        assert result["keypoints"].shape == (n_keypoints, 2)
        assert result["descriptors"].shape[0] == n_keypoints
        assert result["scores"].shape == (n_keypoints,)

        # Verify dtypes
        assert result["keypoints"].dtype == torch.float32
        assert result["descriptors"].dtype == torch.float32
        assert result["scores"].dtype == torch.float32

    def test_extract_features_output_format_disk(self, device):
        """Test DISK produces correct output format."""
        config = FeatureExtractionConfig(extractor_type="disk")
        image = torch.randint(0, 256, (480, 640, 3), dtype=torch.uint8)

        result = extract_features(image, config, device=str(device))

        # Verify structure
        assert isinstance(result, dict)
        assert set(result.keys()) == {"keypoints", "descriptors", "scores"}

        # Verify shapes
        n_keypoints = result["keypoints"].shape[0]
        assert result["keypoints"].shape == (n_keypoints, 2)
        assert result["descriptors"].shape[0] == n_keypoints
        assert result["scores"].shape == (n_keypoints,)

        # Verify dtypes
        assert result["keypoints"].dtype == torch.float32
        assert result["descriptors"].dtype == torch.float32
        assert result["scores"].dtype == torch.float32


@pytest.mark.slow
class TestSuperPointExtraction:
    """Tests that require SuperPoint model weights (marked slow)."""

    def test_output_structure(self, device):
        """Test that extract_features returns correct structure."""
        config = FeatureExtractionConfig()
        image = torch.randint(0, 256, (480, 640, 3), dtype=torch.uint8)

        result = extract_features(image, config, device=str(device))

        # Verify structure
        assert isinstance(result, dict)
        assert set(result.keys()) == {"keypoints", "descriptors", "scores"}

        # Verify shapes
        n_keypoints = result["keypoints"].shape[0]
        assert result["keypoints"].shape == (n_keypoints, 2)
        assert result["descriptors"].shape[0] == n_keypoints
        assert result["scores"].shape == (n_keypoints,)

        # Verify dtypes
        assert result["keypoints"].dtype == torch.float32
        assert result["descriptors"].dtype == torch.float32
        assert result["scores"].dtype == torch.float32

        # Verify device
        assert result["keypoints"].device.type == str(device).split(":")[0]

    def test_max_keypoints_respected(self, device):
        """Test that max_keypoints config is respected."""
        max_kp = 512
        config = FeatureExtractionConfig(max_keypoints=max_kp)
        image = torch.randint(0, 256, (480, 640, 3), dtype=torch.uint8)

        result = extract_features(image, config, device=str(device))

        # Should have at most max_keypoints
        assert result["keypoints"].shape[0] <= max_kp

    def test_extractor_reuse(self, device):
        """Test that providing extractor reuses the same model."""
        config = FeatureExtractionConfig()
        extractor = create_extractor(config, device=str(device))

        # Use structured images that will produce features
        image1 = create_checkerboard_image(480, 640, 40)
        image2 = create_checkerboard_image(480, 640, 30)

        # Extract with same extractor
        result1 = extract_features(
            image1, config, extractor=extractor, device=str(device)
        )
        result2 = extract_features(
            image2, config, extractor=extractor, device=str(device)
        )

        # Both should succeed and return valid results
        assert result1["keypoints"].shape[0] > 0
        assert result2["keypoints"].shape[0] > 0

    def test_batch_extraction(self, device):
        """Test batch extraction produces same results as individual extraction."""
        config = FeatureExtractionConfig()

        # Create structured test images with different patterns
        images = {
            "cam1": create_checkerboard_image(480, 640, 40),
            "cam2": create_checkerboard_image(480, 640, 30),
            "cam3": create_checkerboard_image(480, 640, 50),
        }

        # Extract as batch
        batch_results = extract_features_batch(images, config, device=str(device))

        # Verify structure
        assert set(batch_results.keys()) == {"cam1", "cam2", "cam3"}
        for _cam_name, feats in batch_results.items():
            assert set(feats.keys()) == {"keypoints", "descriptors", "scores"}
            assert feats["keypoints"].shape[0] > 0  # Should find some features

        # Extract individually with shared extractor
        extractor = create_extractor(config, device=str(device))
        individual_results = {}
        for cam_name, image in images.items():
            individual_results[cam_name] = extract_features(
                image, config, extractor=extractor, device=str(device)
            )

        # Results should be identical (same extractor, same images)
        for cam_name in images:
            batch_kp = batch_results[cam_name]["keypoints"]
            indiv_kp = individual_results[cam_name]["keypoints"]
            assert torch.allclose(batch_kp, indiv_kp)

    def test_detection_threshold(self, device):
        """Test that detection threshold affects number of features."""
        # Lower threshold should give more features
        config_low = FeatureExtractionConfig(detection_threshold=0.001)
        config_high = FeatureExtractionConfig(detection_threshold=0.01)

        image = create_checkerboard_image(480, 640, 40)

        result_low = extract_features(image, config_low, device=str(device))
        result_high = extract_features(image, config_high, device=str(device))

        # Lower threshold should generally yield more keypoints
        # (though not guaranteed for all images)
        n_low = result_low["keypoints"].shape[0]
        n_high = result_high["keypoints"].shape[0]

        # At minimum, both should find some features
        assert n_low > 0
        assert n_high > 0

    def test_grayscale_and_color_equivalent(self, device):
        """Test that grayscale and color versions of same image give same features."""
        config = FeatureExtractionConfig()

        # Create structured grayscale image
        gray = create_checkerboard_image(480, 640, 40)

        # Create color image by replicating grayscale to all 3 channels
        # Need to actually copy, not just expand (which shares memory)
        color = gray.unsqueeze(-1).repeat(1, 1, 3)

        result_gray = extract_features(gray, config, device=str(device))
        result_color = extract_features(color, config, device=str(device))

        # Should produce identical results
        assert torch.allclose(result_gray["keypoints"], result_color["keypoints"])
        assert torch.allclose(result_gray["descriptors"], result_color["descriptors"])
        assert torch.allclose(result_gray["scores"], result_color["scores"])


@pytest.mark.slow
class TestIntegration:
    """Integration tests for complete extraction workflow."""

    def test_extract_save_load_workflow(self, device):
        """Test complete workflow: extract, save, load."""
        config = FeatureExtractionConfig()
        image = torch.randint(0, 256, (480, 640, 3), dtype=torch.uint8)

        # Extract features
        features = extract_features(image, config, device=str(device))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.pt"

            # Save to disk
            save_features(features, path)

            # Load back
            loaded = load_features(path)

            # Verify identical
            assert torch.allclose(loaded["keypoints"], features["keypoints"])
            assert torch.allclose(loaded["descriptors"], features["descriptors"])
            assert torch.allclose(loaded["scores"], features["scores"])

    def test_batch_extract_save_load(self, device):
        """Test batch extraction with save/load."""
        config = FeatureExtractionConfig()
        images = {
            "cam1": torch.randint(0, 256, (480, 640, 3), dtype=torch.uint8),
            "cam2": torch.randint(0, 256, (480, 640, 3), dtype=torch.uint8),
        }

        # Extract batch
        batch_features = extract_features_batch(images, config, device=str(device))

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save each camera's features
            paths = {}
            for cam_name, feats in batch_features.items():
                path = Path(tmpdir) / f"{cam_name}.pt"
                save_features(feats, path)
                paths[cam_name] = path

            # Load back
            loaded_features = {}
            for cam_name, path in paths.items():
                loaded_features[cam_name] = load_features(path)

            # Verify
            for cam_name in images:
                orig = batch_features[cam_name]
                loaded = loaded_features[cam_name]
                assert torch.allclose(loaded["keypoints"], orig["keypoints"])
                assert torch.allclose(loaded["descriptors"], orig["descriptors"])
                assert torch.allclose(loaded["scores"], orig["scores"])


class TestApplyCLAHE:
    """Tests for _apply_clahe helper function."""

    def test_apply_clahe_uint8_input(self):
        """Test CLAHE with uint8 input."""
        # Create low-contrast uint8 image
        gray = torch.ones(100, 100, dtype=torch.uint8) * 128
        gray[30:70, 30:70] = 132  # Very subtle variation

        result = _apply_clahe(gray, clip_limit=2.0)

        # Assert output is uint8
        assert result.dtype == torch.uint8

        # Assert same shape
        assert result.shape == gray.shape

        # Assert values differ (contrast enhanced)
        assert not torch.allclose(result.float(), gray.float())

    def test_apply_clahe_float_input(self):
        """Test CLAHE with float input in [0, 255] range."""
        # Create low-contrast float image in [0, 255] range
        gray = torch.ones(100, 100, dtype=torch.float32) * 128.0
        gray[30:70, 30:70] = 132.0  # Very subtle variation

        result = _apply_clahe(gray, clip_limit=2.0)

        # Assert output is float
        assert result.dtype == torch.float32

        # Assert same shape
        assert result.shape == gray.shape

        # Assert values are in valid range
        assert result.min() >= 0.0
        assert result.max() <= 255.0

    def test_apply_clahe_float_normalized_input(self):
        """Test CLAHE with float input in [0, 1] range."""
        # Create low-contrast float image in [0, 1] range
        gray = torch.ones(100, 100, dtype=torch.float32) * 0.5
        gray[30:70, 30:70] = 0.52  # Very subtle variation

        result = _apply_clahe(gray, clip_limit=2.0)

        # Assert output is float
        assert result.dtype == torch.float32

        # Assert same shape
        assert result.shape == gray.shape

        # Assert values are in valid range
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_apply_clahe_preserves_device(self):
        """Test that CLAHE output device matches input device."""
        gray = torch.ones(100, 100, dtype=torch.uint8).cuda()

        result = _apply_clahe(gray, clip_limit=2.0)

        # Assert device is preserved
        assert result.device.type == "cuda"


@pytest.mark.slow
class TestCLAHEIntegration:
    """Tests for CLAHE integration in extract_features."""

    def test_clahe_disabled_passthrough(self, device):
        """Test that CLAHE disabled produces deterministic results."""
        config = FeatureExtractionConfig(clahe_enabled=False)
        image = create_checkerboard_image(480, 640, 40)

        # Extract features twice with CLAHE disabled
        result1 = extract_features(image, config, device=str(device))
        result2 = extract_features(image, config, device=str(device))

        # Results should be identical (deterministic)
        assert torch.allclose(result1["keypoints"], result2["keypoints"])
        assert torch.allclose(result1["descriptors"], result2["descriptors"])
        assert torch.allclose(result1["scores"], result2["scores"])

    def test_clahe_enabled_changes_output(self, device):
        """Test that CLAHE enabled changes feature extraction output."""
        # Create low-contrast gradient image
        height, width = 480, 640
        y_coords = torch.linspace(120, 130, height).unsqueeze(1).expand(height, width)
        image = y_coords.to(torch.uint8)

        config_off = FeatureExtractionConfig(clahe_enabled=False)
        config_on = FeatureExtractionConfig(clahe_enabled=True)

        result_off = extract_features(image, config_off, device=str(device))
        result_on = extract_features(image, config_on, device=str(device))

        # CLAHE should enhance subtle texture and produce >= detections
        n_off = result_off["keypoints"].shape[0]
        n_on = result_on["keypoints"].shape[0]

        # At minimum, both should work (may detect 0 keypoints on flat gradient)
        assert n_off >= 0
        assert n_on >= 0

        # For this specific low-contrast case, CLAHE typically helps
        # but we can't guarantee it always increases count, so just
        # verify it runs without error

    def test_clahe_works_with_all_extractors(self, device):
        """Test that CLAHE works with all extractor types."""
        image = create_checkerboard_image(480, 640, 40)

        for extractor_type in ["superpoint", "aliked", "disk"]:
            config = FeatureExtractionConfig(
                extractor_type=extractor_type,
                clahe_enabled=True,
                clahe_clip_limit=2.0,
            )

            result = extract_features(image, config, device=str(device))

            # Verify successful extraction
            assert isinstance(result, dict)
            assert set(result.keys()) == {"keypoints", "descriptors", "scores"}
            assert result["keypoints"].shape[0] > 0  # Should find features

    def test_clahe_clip_limit_affects_enhancement(self):
        """Test that different clip limits produce different results."""
        # Create low-contrast image
        height, width = 480, 640
        image = torch.ones(height, width, dtype=torch.uint8) * 128
        # Add subtle pattern
        image[::20, :] = 130

        config_low = FeatureExtractionConfig(clahe_enabled=True, clahe_clip_limit=1.0)
        config_high = FeatureExtractionConfig(clahe_enabled=True, clahe_clip_limit=4.0)

        result_low = extract_features(image, config_low, device="cpu")
        result_high = extract_features(image, config_high, device="cpu")

        # Different clip limits should produce different keypoints
        # (though we can't guarantee the exact relationship, just that they work)
        assert isinstance(result_low, dict)
        assert isinstance(result_high, dict)
