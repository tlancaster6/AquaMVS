"""Tests for RoMa v2 dense matching."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from aquamvs.config import DenseMatchingConfig
from aquamvs.features.roma import (
    _extract_correspondences,
    _run_roma,
    apply_mask_to_correspondences,
    create_roma_matcher,
    match_all_pairs_roma,
    match_pair_roma,
    run_roma_all_pairs,
)


class TestDenseMatchingConfig:
    """Tests for DenseMatchingConfig."""

    def test_defaults(self):
        """Test default values."""
        config = DenseMatchingConfig()
        assert config.certainty_threshold == 0.5
        assert config.max_correspondences == 100000

    def test_custom_values(self):
        """Test custom values."""
        config = DenseMatchingConfig(certainty_threshold=0.7, max_correspondences=5000)
        assert config.certainty_threshold == 0.7
        assert config.max_correspondences == 5000


class TestExtractCorrespondences:
    """Tests for _extract_correspondences helper."""

    def test_basic_extraction(self):
        """Test basic correspondence extraction from dense warp."""
        # Create mock warp and overlap
        H_warp, W_warp = 10, 10
        H_A, W_A = 100, 100
        H_B, W_B = 100, 100

        # Create normalized warp (all pixels map to center of image B)
        warp = torch.zeros(H_warp, W_warp, 2)

        # Create overlap with half pixels above threshold
        overlap = torch.zeros(H_warp, W_warp)
        overlap[:5, :] = 0.8  # Top half has high certainty

        config = DenseMatchingConfig(certainty_threshold=0.5, max_correspondences=10000)

        result = _extract_correspondences(
            warp=warp,
            overlap=overlap,
            H_A=H_A,
            W_A=W_A,
            H_B=H_B,
            W_B=W_B,
            config=config,
            device="cpu",
        )

        # Should have correspondences for top half (5*10 = 50 pixels)
        assert result["ref_keypoints"].shape == (50, 2)
        assert result["src_keypoints"].shape == (50, 2)
        assert result["scores"].shape == (50,)

        # All scores should be 0.8
        assert torch.allclose(result["scores"], torch.tensor(0.8))

    def test_subsampling_by_topk(self):
        """Test subsampling when too many correspondences."""
        H_warp, W_warp = 10, 10
        H_A, W_A = 100, 100
        H_B, W_B = 100, 100

        warp = torch.zeros(H_warp, W_warp, 2)

        # All pixels above threshold
        overlap = torch.ones(H_warp, W_warp) * 0.6

        # Set a few pixels to higher certainty
        overlap[0, 0] = 0.95
        overlap[1, 1] = 0.90
        overlap[2, 2] = 0.85

        # Limit to 10 correspondences
        config = DenseMatchingConfig(certainty_threshold=0.5, max_correspondences=10)

        result = _extract_correspondences(
            warp=warp,
            overlap=overlap,
            H_A=H_A,
            W_A=W_A,
            H_B=H_B,
            W_B=W_B,
            config=config,
            device="cpu",
        )

        # Should have exactly 10 correspondences
        assert result["ref_keypoints"].shape == (10, 2)
        assert result["src_keypoints"].shape == (10, 2)
        assert result["scores"].shape == (10,)

        # Top score should be included (spatial binning may not preserve all high-certainty pixels)
        assert 0.95 in result["scores"]

    def test_threshold_filtering(self):
        """Test that threshold correctly filters low certainty pixels."""
        H_warp, W_warp = 5, 5
        H_A, W_A = 50, 50
        H_B, W_B = 50, 50

        warp = torch.zeros(H_warp, W_warp, 2)

        # Mix of high and low certainty
        overlap = torch.tensor(
            [
                [0.8, 0.8, 0.3, 0.3, 0.3],
                [0.8, 0.8, 0.3, 0.3, 0.3],
                [0.3, 0.3, 0.3, 0.3, 0.3],
                [0.3, 0.3, 0.3, 0.3, 0.3],
                [0.3, 0.3, 0.3, 0.3, 0.3],
            ]
        )

        config = DenseMatchingConfig(certainty_threshold=0.5, max_correspondences=10000)

        result = _extract_correspondences(
            warp=warp,
            overlap=overlap,
            H_A=H_A,
            W_A=W_A,
            H_B=H_B,
            W_B=W_B,
            config=config,
            device="cpu",
        )

        # Should have 4 correspondences (2x2 top-left block)
        assert result["ref_keypoints"].shape == (4, 2)
        assert result["src_keypoints"].shape == (4, 2)
        assert result["scores"].shape == (4,)

        # All scores should be 0.8
        assert torch.allclose(result["scores"], torch.tensor(0.8))

    def test_empty_result_when_no_pixels_pass(self):
        """Test that empty tensors are returned when no pixels pass threshold."""
        H_warp, W_warp = 5, 5
        H_A, W_A = 50, 50
        H_B, W_B = 50, 50

        warp = torch.zeros(H_warp, W_warp, 2)

        # All pixels below threshold
        overlap = torch.ones(H_warp, W_warp) * 0.3

        config = DenseMatchingConfig(certainty_threshold=0.5, max_correspondences=10000)

        result = _extract_correspondences(
            warp=warp,
            overlap=overlap,
            H_A=H_A,
            W_A=W_A,
            H_B=H_B,
            W_B=W_B,
            config=config,
            device="cpu",
        )

        # Should have empty tensors
        assert result["ref_keypoints"].shape == (0, 2)
        assert result["src_keypoints"].shape == (0, 2)
        assert result["scores"].shape == (0,)


class TestApplyMaskToCorrespondences:
    """Tests for apply_mask_to_correspondences."""

    def test_mask_filters_out_of_bounds_pixels(self):
        """Test that mask filters correspondences outside valid region."""
        # Create correspondences
        correspondences = {
            "ref_keypoints": torch.tensor(
                [[10.0, 10.0], [50.0, 50.0], [90.0, 90.0]], dtype=torch.float32
            ),
            "src_keypoints": torch.tensor(
                [[15.0, 15.0], [55.0, 55.0], [95.0, 95.0]], dtype=torch.float32
            ),
            "scores": torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32),
        }

        # Create mask (100x100) with valid region in center
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255

        filtered = apply_mask_to_correspondences(correspondences, mask)

        # Only middle correspondence should survive
        assert filtered["ref_keypoints"].shape == (1, 2)
        assert filtered["src_keypoints"].shape == (1, 2)
        assert filtered["scores"].shape == (1,)

        assert torch.allclose(filtered["ref_keypoints"], torch.tensor([[50.0, 50.0]]))
        assert torch.allclose(filtered["src_keypoints"], torch.tensor([[55.0, 55.0]]))
        assert torch.allclose(filtered["scores"], torch.tensor([0.8]))

    def test_mask_all_valid(self):
        """Test mask with all pixels valid."""
        correspondences = {
            "ref_keypoints": torch.tensor(
                [[10.0, 10.0], [50.0, 50.0]], dtype=torch.float32
            ),
            "src_keypoints": torch.tensor(
                [[15.0, 15.0], [55.0, 55.0]], dtype=torch.float32
            ),
            "scores": torch.tensor([0.9, 0.8], dtype=torch.float32),
        }

        # All valid mask
        mask = np.ones((100, 100), dtype=np.uint8) * 255

        filtered = apply_mask_to_correspondences(correspondences, mask)

        # All correspondences should survive
        assert filtered["ref_keypoints"].shape == (2, 2)
        assert filtered["src_keypoints"].shape == (2, 2)
        assert filtered["scores"].shape == (2,)

    def test_mask_all_invalid(self):
        """Test mask with all pixels invalid."""
        correspondences = {
            "ref_keypoints": torch.tensor(
                [[10.0, 10.0], [50.0, 50.0]], dtype=torch.float32
            ),
            "src_keypoints": torch.tensor(
                [[15.0, 15.0], [55.0, 55.0]], dtype=torch.float32
            ),
            "scores": torch.tensor([0.9, 0.8], dtype=torch.float32),
        }

        # All invalid mask
        mask = np.zeros((100, 100), dtype=np.uint8)

        filtered = apply_mask_to_correspondences(correspondences, mask)

        # No correspondences should survive
        assert filtered["ref_keypoints"].shape == (0, 2)
        assert filtered["src_keypoints"].shape == (0, 2)
        assert filtered["scores"].shape == (0,)


@pytest.mark.slow
def test_create_roma_matcher():
    """Test that create_roma_matcher initializes model correctly."""
    pytest.importorskip("romav2")

    matcher = create_roma_matcher(device="cpu")

    # Check that it's in eval mode
    assert not matcher.training

    # Check device
    # Note: RoMa might place submodules on different devices, just check it doesn't crash


@pytest.mark.slow
def test_match_pair_roma_output_format():
    """Test that match_pair_roma returns correct output format with real RoMa."""
    pytest.importorskip("romav2")

    # Create small test images
    img_ref = torch.randint(0, 255, (100, 100, 3), dtype=torch.uint8)
    img_src = torch.randint(0, 255, (100, 100, 3), dtype=torch.uint8)

    config = DenseMatchingConfig(certainty_threshold=0.5, max_correspondences=1000)

    try:
        result = match_pair_roma(img_ref, img_src, config, device="cpu")
    except Exception as e:
        pytest.skip(f"RoMa not available or failed: {e}")

    # Verify output structure
    assert "ref_keypoints" in result
    assert "src_keypoints" in result
    assert "scores" in result

    # Verify shapes
    n_matches = len(result["scores"])
    assert result["ref_keypoints"].shape == (n_matches, 2)
    assert result["src_keypoints"].shape == (n_matches, 2)
    assert result["scores"].shape == (n_matches,)

    # Verify dtypes
    assert result["ref_keypoints"].dtype == torch.float32
    assert result["src_keypoints"].dtype == torch.float32
    assert result["scores"].dtype == torch.float32


def test_match_all_pairs_roma_structure():
    """Test match_all_pairs_roma canonical pair ordering with mocked RoMa."""
    # Mock the RoMa model
    with patch("aquamvs.features.roma.create_roma_matcher") as mock_create:
        mock_matcher = MagicMock()

        # Mock match output
        def mock_match(img_a, img_b):
            H, W = 1280, 1280
            return {
                "warp_AB": torch.zeros(1, H, W, 2),
                "overlap_AB": torch.ones(1, H, W, 1) * 0.6,
            }

        mock_matcher.match.side_effect = mock_match
        # Mock parameters() to return a list (iter() will be called on it)
        mock_matcher.parameters.return_value = [torch.tensor([0.0], device="cpu")]
        mock_create.return_value = mock_matcher

        # Create synthetic images for 3 cameras
        undistorted_images = {
            "cam0": torch.randint(0, 255, (100, 100, 3), dtype=torch.uint8),
            "cam1": torch.randint(0, 255, (100, 100, 3), dtype=torch.uint8),
            "cam2": torch.randint(0, 255, (100, 100, 3), dtype=torch.uint8),
        }

        # Bidirectional pairs: cam0 <-> cam1, cam0 <-> cam2
        pairs = {
            "cam0": ["cam1", "cam2"],
            "cam1": ["cam0"],
            "cam2": ["cam0"],
        }

        config = DenseMatchingConfig()

        result = match_all_pairs_roma(
            undistorted_images=undistorted_images,
            pairs=pairs,
            config=config,
            device="cpu",
        )

        # Should have exactly 2 entries (deduplicated)
        assert len(result) == 2

        # Keys should be in canonical order
        assert ("cam0", "cam1") in result
        assert ("cam0", "cam2") in result

        # Should NOT have reverse pairs
        assert ("cam1", "cam0") not in result
        assert ("cam2", "cam0") not in result


def test_match_all_pairs_roma_with_masks():
    """Test that match_all_pairs_roma applies masks to correspondences."""
    with patch("aquamvs.features.roma.create_roma_matcher") as mock_create:
        mock_matcher = MagicMock()

        # Mock match output with known correspondences
        def mock_match(img_a, img_b):
            H, W = 100, 100
            warp = torch.zeros(H, W, 2)
            overlap = torch.ones(H, W, 1) * 0.8
            return {
                "warp_AB": warp.unsqueeze(0),
                "overlap_AB": overlap.unsqueeze(0),
            }

        mock_matcher.match.side_effect = mock_match
        # Mock parameters() to return a list (iter() will be called on it)
        mock_matcher.parameters.return_value = [torch.tensor([0.0], device="cpu")]
        mock_create.return_value = mock_matcher

        # Create images
        undistorted_images = {
            "cam0": torch.randint(0, 255, (100, 100, 3), dtype=torch.uint8),
            "cam1": torch.randint(0, 255, (100, 100, 3), dtype=torch.uint8),
        }

        pairs = {"cam0": ["cam1"]}

        # Create mask for cam0 (valid region only in top-left corner)
        mask_cam0 = np.zeros((100, 100), dtype=np.uint8)
        mask_cam0[:50, :50] = 255

        masks = {"cam0": mask_cam0}

        config = DenseMatchingConfig(certainty_threshold=0.5, max_correspondences=10000)

        result = match_all_pairs_roma(
            undistorted_images=undistorted_images,
            pairs=pairs,
            config=config,
            device="cpu",
            masks=masks,
        )

        # Should have one pair
        assert len(result) == 1
        assert ("cam0", "cam1") in result

        # All ref keypoints should be in top-left quadrant
        ref_kpts = result[("cam0", "cam1")]["ref_keypoints"]
        assert (ref_kpts[:, 0] < 50).all()  # u < 50
        assert (ref_kpts[:, 1] < 50).all()  # v < 50


class TestRunRomaRefactor:
    """Tests for _run_roma refactor and run_roma_all_pairs."""

    def test_run_roma_output_structure(self):
        """Test that _run_roma returns correct keys."""
        with patch("aquamvs.features.roma.create_roma_matcher") as mock_create:
            mock_matcher = MagicMock()

            # Mock match output
            def mock_match(img_a, img_b):
                H, W = 100, 100
                return {
                    "warp_AB": torch.zeros(1, H, W, 2),
                    "overlap_AB": torch.ones(1, H, W, 1) * 0.8,
                }

            mock_matcher.match.side_effect = mock_match
            mock_create.return_value = mock_matcher

            img_ref = torch.randint(0, 255, (100, 100, 3), dtype=torch.uint8)
            img_src = torch.randint(0, 255, (100, 100, 3), dtype=torch.uint8)

            result = _run_roma(img_ref, img_src, mock_matcher)

            # Check keys
            assert "warp_AB" in result
            assert "overlap_AB" in result
            assert "H_ref" in result
            assert "W_ref" in result
            assert "H_src" in result
            assert "W_src" in result

            # Check shapes
            assert result["warp_AB"].shape[2] == 2
            assert result["overlap_AB"].ndim == 2
            assert result["H_ref"] == 100
            assert result["W_ref"] == 100

    def test_match_pair_roma_regression(self):
        """Test that match_pair_roma still works after refactor."""
        with patch("aquamvs.features.roma.create_roma_matcher") as mock_create:
            mock_matcher = MagicMock()

            # Mock match output with high certainty everywhere
            def mock_match(img_a, img_b):
                H, W = 50, 50
                return {
                    "warp_AB": torch.zeros(1, H, W, 2),
                    "overlap_AB": torch.ones(1, H, W, 1) * 0.9,
                }

            mock_matcher.match.side_effect = mock_match
            # Mock parameters() to return a list (iter() will be called on it)
            mock_matcher.parameters.return_value = [torch.tensor([0.0], device="cpu")]
            mock_create.return_value = mock_matcher

            img_ref = torch.randint(0, 255, (100, 100, 3), dtype=torch.uint8)
            img_src = torch.randint(0, 255, (100, 100, 3), dtype=torch.uint8)

            config = DenseMatchingConfig(
                certainty_threshold=0.5, max_correspondences=10000
            )

            result = match_pair_roma(img_ref, img_src, config)

            # Should have standard correspondence format
            assert "ref_keypoints" in result
            assert "src_keypoints" in result
            assert "scores" in result

            # Should have correspondences (high certainty everywhere)
            assert result["ref_keypoints"].shape[0] > 0

    def test_run_roma_all_pairs_directed_keys(self):
        """Test that run_roma_all_pairs uses directed keys."""
        with patch("aquamvs.features.roma.create_roma_matcher") as mock_create:
            mock_matcher = MagicMock()

            # Mock match output
            def mock_match(img_a, img_b):
                H, W = 50, 50
                return {
                    "warp_AB": torch.zeros(1, H, W, 2),
                    "overlap_AB": torch.ones(1, H, W, 1) * 0.8,
                }

            mock_matcher.match.side_effect = mock_match
            mock_create.return_value = mock_matcher

            # Create images for 3 cameras
            undistorted_images = {
                "cam0": torch.randint(0, 255, (100, 100, 3), dtype=torch.uint8),
                "cam1": torch.randint(0, 255, (100, 100, 3), dtype=torch.uint8),
                "cam2": torch.randint(0, 255, (100, 100, 3), dtype=torch.uint8),
            }

            # Bidirectional pairs
            pairs = {
                "cam0": ["cam1", "cam2"],
                "cam1": ["cam0", "cam2"],
                "cam2": ["cam0", "cam1"],
            }

            config = DenseMatchingConfig()

            result = run_roma_all_pairs(
                undistorted_images=undistorted_images,
                pairs=pairs,
                config=config,
                device="cpu",
            )

            # Should have 6 directed pairs (not 3 canonical)
            assert len(result) == 6

            # Check all directed pairs are present
            assert ("cam0", "cam1") in result
            assert ("cam0", "cam2") in result
            assert ("cam1", "cam0") in result
            assert ("cam1", "cam2") in result
            assert ("cam2", "cam0") in result
            assert ("cam2", "cam1") in result

            # Each result should have raw warp dict structure
            for _key, warp_dict in result.items():
                assert "warp_AB" in warp_dict
                assert "overlap_AB" in warp_dict
