"""Tests for RoMa warp-to-depth conversion."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from aquamvs.config import DenseMatchingConfig, ReconstructionConfig
from aquamvs.dense.roma_depth import (
    aggregate_pairwise_depths,
    roma_warps_to_depth_maps,
    warp_to_pairwise_depth,
)


@pytest.fixture
def mock_projection_models():
    """Create mock projection models for testing."""
    models = {}
    for cam_name in ["cam0", "cam1"]:
        model = MagicMock()

        # Mock cast_ray: returns simple rays pointing into +Z
        def cast_ray_fn(pixels):
            N = pixels.shape[0]
            origins = torch.zeros(N, 3)
            origins[:, 2] = 0.5  # Water surface at Z=0.5
            directions = torch.zeros(N, 3)
            directions[:, 2] = 1.0  # Point into water (+Z)
            return origins, directions

        model.cast_ray.side_effect = cast_ray_fn
        models[cam_name] = model

    return models


class TestWarpToPairwiseDepth:
    """Tests for warp_to_pairwise_depth."""

    def test_basic_conversion(self, mock_projection_models):
        """Test basic warp to depth conversion."""
        # Create synthetic warp (identity: ref and src pixels are the same)
        H_warp, W_warp = 10, 10
        warp_AB = torch.zeros(H_warp, W_warp, 2)  # Identity warp in normalized coords
        overlap_AB = torch.ones(H_warp, W_warp) * 0.8

        roma_result = {
            "warp_AB": warp_AB,
            "overlap_AB": overlap_AB,
            "H_ref": 100,
            "W_ref": 100,
            "H_src": 100,
            "W_src": 100,
        }

        ref_model = mock_projection_models["cam0"]
        src_model = mock_projection_models["cam1"]

        depth_map, certainty = warp_to_pairwise_depth(
            roma_result,
            ref_model,
            src_model,
            certainty_threshold=0.5,
        )

        # Should have correct shape
        assert depth_map.shape == (H_warp, W_warp)
        assert certainty.shape == (H_warp, W_warp)

        # With parallel rays pointing +Z, triangulation may be degenerate
        # but at least some pixels should be valid (those above threshold)
        # Check that certainty values match overlap where valid
        valid_mask = ~torch.isnan(depth_map)
        if valid_mask.any():
            assert torch.allclose(certainty[valid_mask], overlap_AB[valid_mask])

    def test_certainty_threshold_filtering(self, mock_projection_models):
        """Test that pixels below certainty threshold are filtered."""
        H_warp, W_warp = 5, 5
        warp_AB = torch.zeros(H_warp, W_warp, 2)

        # Create overlap with mixed certainty
        overlap_AB = torch.tensor(
            [
                [0.8, 0.8, 0.3, 0.3, 0.3],
                [0.8, 0.8, 0.3, 0.3, 0.3],
                [0.3, 0.3, 0.3, 0.3, 0.3],
                [0.3, 0.3, 0.3, 0.3, 0.3],
                [0.3, 0.3, 0.3, 0.3, 0.3],
            ]
        )

        roma_result = {
            "warp_AB": warp_AB,
            "overlap_AB": overlap_AB,
            "H_ref": 50,
            "W_ref": 50,
            "H_src": 50,
            "W_src": 50,
        }

        ref_model = mock_projection_models["cam0"]
        src_model = mock_projection_models["cam1"]

        depth_map, certainty = warp_to_pairwise_depth(
            roma_result,
            ref_model,
            src_model,
            certainty_threshold=0.5,
        )

        # Low certainty pixels (< 0.5) should be NaN
        assert torch.isnan(depth_map[2:, :]).all()
        assert torch.isnan(depth_map[:2, 2:]).all()

        # Certainty map should be 0 for invalid pixels
        assert (certainty[2:, :] == 0).all()
        assert (certainty[:2, 2:] == 0).all()

    def test_negative_depth_filtering(self):
        """Test that negative depths are filtered out."""
        # Create a projection model that produces negative depths
        ref_model = MagicMock()
        src_model = MagicMock()

        def cast_ray_ref(pixels):
            N = pixels.shape[0]
            origins = torch.zeros(N, 3)
            directions = torch.zeros(N, 3)
            directions[:, 2] = 1.0  # Forward
            return origins, directions

        def cast_ray_src(pixels):
            N = pixels.shape[0]
            origins = torch.zeros(N, 3)
            origins[:, 2] = 1.0  # Ahead of ref origin
            directions = torch.zeros(N, 3)
            directions[:, 2] = -1.0  # Backward (will give negative depth)
            return origins, directions

        ref_model.cast_ray.side_effect = cast_ray_ref
        src_model.cast_ray.side_effect = cast_ray_src

        H_warp, W_warp = 5, 5
        warp_AB = torch.zeros(H_warp, W_warp, 2)
        overlap_AB = torch.ones(H_warp, W_warp) * 0.9

        roma_result = {
            "warp_AB": warp_AB,
            "overlap_AB": overlap_AB,
            "H_ref": 50,
            "W_ref": 50,
            "H_src": 50,
            "W_src": 50,
        }

        depth_map, certainty = warp_to_pairwise_depth(
            roma_result,
            ref_model,
            src_model,
            certainty_threshold=0.5,
        )

        # All depths should be NaN (negative depths filtered)
        assert torch.isnan(depth_map).all()
        assert (certainty == 0).all()


class TestAggregatePairwiseDepths:
    """Tests for aggregate_pairwise_depths."""

    def test_consensus_aggregation(self):
        """Test that consensus depths are aggregated correctly."""
        # Create 5 depth maps: 3 agreeing, 2 outliers
        H, W = 10, 10
        device = "cpu"

        depth1 = torch.full((H, W), 1.0, device=device)
        depth2 = torch.full((H, W), 1.002, device=device)  # Within tolerance
        depth3 = torch.full((H, W), 0.998, device=device)  # Within tolerance
        depth4 = torch.full((H, W), 2.0, device=device)  # Outlier
        depth5 = torch.full((H, W), 0.5, device=device)  # Outlier

        pairwise_depths = [depth1, depth2, depth3, depth4, depth5]

        depth_map, confidence, consistency = aggregate_pairwise_depths(
            pairwise_depths,
            depth_tolerance=0.005,
            min_consistent_views=3,
        )

        # All pixels should be valid (3 agreeing sources)
        assert not torch.isnan(depth_map).any()

        # Median of agreeing values should be close to 1.0
        assert torch.allclose(depth_map, torch.tensor(1.0), atol=0.005)

        # Confidence should be 3/5 = 0.6
        assert torch.allclose(confidence, torch.tensor(0.6))

        # Consistency count should be 3 (int32)
        assert consistency.dtype == torch.int32
        assert (consistency == 3).all()

    def test_insufficient_agreement(self):
        """Test that pixels with insufficient agreement are filtered."""
        H, W = 5, 5

        # All depths disagree
        depth1 = torch.full((H, W), 1.0)
        depth2 = torch.full((H, W), 2.0)
        depth3 = torch.full((H, W), 3.0)

        pairwise_depths = [depth1, depth2, depth3]

        depth_map, confidence, _ = aggregate_pairwise_depths(
            pairwise_depths,
            depth_tolerance=0.01,
            min_consistent_views=3,
        )

        # All pixels should be invalid (no consensus)
        assert torch.isnan(depth_map).all()
        assert (confidence == 0).all()

    def test_nan_handling(self):
        """Test that NaN pixels are excluded from aggregation."""
        H, W = 5, 5

        # Create depth maps with NaN in different locations
        depth1 = torch.ones(H, W)
        depth1[0, 0] = float("nan")

        depth2 = torch.ones(H, W) * 1.001
        depth2[1, 1] = float("nan")

        depth3 = torch.ones(H, W) * 0.999

        pairwise_depths = [depth1, depth2, depth3]

        depth_map, confidence, _ = aggregate_pairwise_depths(
            pairwise_depths,
            depth_tolerance=0.005,
            min_consistent_views=2,
        )

        # Pixel (0, 0) has only 2 valid sources
        assert not torch.isnan(depth_map[0, 0])
        assert torch.isclose(confidence[0, 0], torch.tensor(2.0 / 2.0))

        # Pixel (1, 1) has only 2 valid sources
        assert not torch.isnan(depth_map[1, 1])
        assert torch.isclose(confidence[1, 1], torch.tensor(2.0 / 2.0))

        # Other pixels have 3 valid sources
        assert not torch.isnan(depth_map[2, 2])
        assert torch.isclose(confidence[2, 2], torch.tensor(3.0 / 3.0))

    def test_mixed_valid_invalid(self):
        """Test aggregation with mix of valid and invalid pixels."""
        H, W = 3, 3

        # Create depth maps where different pixels are valid
        depth1 = torch.full((H, W), float("nan"))
        depth1[0, 0] = 1.0
        depth1[1, 1] = 1.0

        depth2 = torch.full((H, W), float("nan"))
        depth2[0, 0] = 1.001
        depth2[2, 2] = 1.0

        depth3 = torch.full((H, W), float("nan"))
        depth3[1, 1] = 0.999
        depth3[2, 2] = 1.001

        pairwise_depths = [depth1, depth2, depth3]

        depth_map, confidence, _ = aggregate_pairwise_depths(
            pairwise_depths,
            depth_tolerance=0.005,
            min_consistent_views=2,
        )

        # (0, 0) has 2 sources, should be valid
        assert not torch.isnan(depth_map[0, 0])

        # (1, 1) has 2 sources, should be valid
        assert not torch.isnan(depth_map[1, 1])

        # (2, 2) has 2 sources, should be valid
        assert not torch.isnan(depth_map[2, 2])

        # (0, 1) has 0 sources, should be NaN
        assert torch.isnan(depth_map[0, 1])


class TestRomaWarpsToDepthMaps:
    """Tests for roma_warps_to_depth_maps integration."""

    def test_end_to_end_conversion(self):
        """Test complete warp-to-depth pipeline."""
        # Create synthetic warps
        H_warp, W_warp = 20, 20
        warp_AB = torch.zeros(H_warp, W_warp, 2)
        overlap_AB = torch.ones(H_warp, W_warp) * 0.9

        all_warps = {
            ("cam0", "cam1"): {
                "warp_AB": warp_AB,
                "overlap_AB": overlap_AB,
                "H_ref": 100,
                "W_ref": 100,
                "H_src": 100,
                "W_src": 100,
            },
            ("cam0", "cam2"): {
                "warp_AB": warp_AB.clone(),
                "overlap_AB": overlap_AB.clone(),
                "H_ref": 100,
                "W_ref": 100,
                "H_src": 100,
                "W_src": 100,
            },
        }

        # Create mock projection models
        projection_models = {}
        for cam_name in ["cam0", "cam1", "cam2"]:
            model = MagicMock()

            def cast_ray_fn(pixels):
                N = pixels.shape[0]
                # Create convergent rays that triangulate to valid points
                origins = torch.randn(N, 3)
                origins[:, 2] = 0.5  # Water surface
                directions = torch.randn(N, 3)
                directions[:, 2] = torch.abs(directions[:, 2])  # Positive Z
                # Normalize
                directions = directions / directions.norm(dim=-1, keepdim=True)
                return origins, directions

            model.cast_ray.side_effect = cast_ray_fn
            projection_models[cam_name] = model

        pairs = {
            "cam0": ["cam1", "cam2"],
        }

        ring_cameras = ["cam0"]

        dense_matching_config = DenseMatchingConfig(certainty_threshold=0.5)
        reconstruction_config = ReconstructionConfig(
            depth_tolerance=0.01,
            min_consistent_views=2,
        )

        depth_maps, confidence_maps, _ = roma_warps_to_depth_maps(
            ring_cameras=ring_cameras,
            pairs=pairs,
            all_warps=all_warps,
            projection_models=projection_models,
            dense_matching_config=dense_matching_config,
            reconstruction_config=reconstruction_config,
            image_size=(100, 100),
            masks=None,
        )

        # Should have depth map for cam0
        assert "cam0" in depth_maps
        assert "cam0" in confidence_maps

        # Should be upsampled to full resolution
        assert depth_maps["cam0"].shape == (100, 100)
        assert confidence_maps["cam0"].shape == (100, 100)

    def test_mask_application(self):
        """Test that ROI masks are applied to depth maps."""
        H_warp, W_warp = 20, 20
        warp_AB = torch.zeros(H_warp, W_warp, 2)
        overlap_AB = torch.ones(H_warp, W_warp) * 0.9

        all_warps = {
            ("cam0", "cam1"): {
                "warp_AB": warp_AB,
                "overlap_AB": overlap_AB,
                "H_ref": 100,
                "W_ref": 100,
                "H_src": 100,
                "W_src": 100,
            },
        }

        # Mock projection model
        model = MagicMock()

        def cast_ray_fn(pixels):
            N = pixels.shape[0]
            origins = torch.zeros(N, 3)
            origins[:, 2] = 0.5
            directions = torch.zeros(N, 3)
            directions[:, 2] = 1.0
            return origins, directions

        model.cast_ray.side_effect = cast_ray_fn

        projection_models = {"cam0": model, "cam1": model}
        pairs = {"cam0": ["cam1"]}
        ring_cameras = ["cam0"]

        # Create mask (valid only in top-left corner)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:50, :50] = 255
        masks = {"cam0": mask}

        dense_matching_config = DenseMatchingConfig(certainty_threshold=0.5)
        reconstruction_config = ReconstructionConfig(
            depth_tolerance=0.01,
            min_consistent_views=1,
        )

        depth_maps, confidence_maps, _ = roma_warps_to_depth_maps(
            ring_cameras=ring_cameras,
            pairs=pairs,
            all_warps=all_warps,
            projection_models=projection_models,
            dense_matching_config=dense_matching_config,
            reconstruction_config=reconstruction_config,
            image_size=(100, 100),
            masks=masks,
        )

        # Pixels outside mask (bottom-right) should be NaN
        assert torch.isnan(depth_maps["cam0"][60:, 60:]).all()
        assert (confidence_maps["cam0"][60:, 60:] == 0).all()

    def test_no_sources(self):
        """Test handling when a camera has no source cameras."""
        all_warps = {}
        projection_models = {"cam0": MagicMock()}
        pairs = {"cam0": []}  # No sources
        ring_cameras = ["cam0"]

        dense_matching_config = DenseMatchingConfig()
        reconstruction_config = ReconstructionConfig()

        depth_maps, confidence_maps, _ = roma_warps_to_depth_maps(
            ring_cameras=ring_cameras,
            pairs=pairs,
            all_warps=all_warps,
            projection_models=projection_models,
            dense_matching_config=dense_matching_config,
            reconstruction_config=reconstruction_config,
            image_size=(100, 100),
            masks=None,
        )

        # Should have no depth maps
        assert len(depth_maps) == 0
        assert len(confidence_maps) == 0
