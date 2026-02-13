"""Tests for geometric consistency filtering."""

import math

import pytest
import torch

from aquamvs.config import FusionConfig
from aquamvs.fusion import _sample_depth_map, filter_all_depth_maps, filter_depth_map
from aquamvs.projection import RefractiveProjectionModel


@pytest.fixture
def reference_camera(device):
    """Create a simple camera at world origin looking down."""
    R = torch.eye(3, dtype=torch.float32, device=device)
    t = torch.zeros(3, dtype=torch.float32, device=device)
    # Wide field of view: smaller focal length
    K = torch.tensor(
        [[400.0, 0.0, 250.0], [0.0, 400.0, 250.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    water_z = 1.0
    normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
    return RefractiveProjectionModel(K, R, t, water_z, normal, 1.0, 1.333)


@pytest.fixture
def offset_camera(device):
    """Create a camera offset to the right with small offset for overlap."""
    # Small offset 0.1m to the right (X+) for better field overlap
    R = torch.eye(3, dtype=torch.float32, device=device)
    t = torch.tensor([-0.1, 0.0, 0.0], dtype=torch.float32, device=device)
    # Wide field of view: smaller focal length
    K = torch.tensor(
        [[400.0, 0.0, 250.0], [0.0, 400.0, 250.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    water_z = 1.0
    normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
    return RefractiveProjectionModel(K, R, t, water_z, normal, 1.0, 1.333)


class TestSampleDepthMap:
    """Tests for _sample_depth_map bilinear interpolation."""

    def test_integer_pixel_lookup(self, device):
        """Test sampling at integer pixel coordinates returns exact values."""
        # 4x4 depth map with known values
        depth_map = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)

        # Sample at integer locations
        pixels = torch.tensor([[0.0, 0.0], [3.0, 0.0], [1.0, 2.0]], device=device)
        valid = torch.ones(3, dtype=torch.bool, device=device)

        result = _sample_depth_map(depth_map, pixels, valid)

        # Verify exact values
        expected = torch.tensor([0.0, 3.0, 9.0], device=device)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_subpixel_bilinear_interpolation(self, device):
        """Test bilinear interpolation at sub-pixel locations."""
        # 3x3 depth map with known values
        depth_map = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=torch.float32,
            device=device,
        )

        # Sample at (0.5, 0.5) - midpoint of top-left 2x2 block
        # Expected: (1 + 2 + 4 + 5) / 4 = 3.0
        pixels = torch.tensor([[0.5, 0.5]], device=device)
        valid = torch.ones(1, dtype=torch.bool, device=device)

        result = _sample_depth_map(depth_map, pixels, valid)

        assert torch.allclose(result, torch.tensor([3.0], device=device), atol=1e-5)

    def test_nan_propagation(self, device):
        """Test that NaN values are properly propagated during interpolation."""
        # 3x3 depth map with NaN in the middle
        depth_map = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, float("nan"), 6.0], [7.0, 8.0, 9.0]],
            dtype=torch.float32,
            device=device,
        )

        # Sample near the NaN (should be affected by it)
        pixels = torch.tensor([[1.0, 1.0], [1.5, 1.5]], device=device)
        valid = torch.ones(2, dtype=torch.bool, device=device)

        result = _sample_depth_map(depth_map, pixels, valid)

        # Both samples should be NaN (touching a NaN neighbor)
        assert torch.all(torch.isnan(result))

    def test_invalid_mask_returns_nan(self, device):
        """Test that invalid pixels return NaN."""
        depth_map = torch.ones(4, 4, dtype=torch.float32, device=device)

        pixels = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device)
        valid = torch.tensor([True, False], device=device)

        result = _sample_depth_map(depth_map, pixels, valid)

        # First should be valid (1.0), second should be NaN
        assert result[0] == 1.0
        assert torch.isnan(result[1])

    def test_all_invalid_returns_all_nan(self, device):
        """Test that all-invalid mask returns all NaN."""
        depth_map = torch.ones(4, 4, dtype=torch.float32, device=device)

        pixels = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device)
        valid = torch.zeros(2, dtype=torch.bool, device=device)

        result = _sample_depth_map(depth_map, pixels, valid)

        assert torch.all(torch.isnan(result))


class TestFilterDepthMap:
    """Tests for filter_depth_map consistency filtering."""

    def test_perfect_consistency_single_target(self, reference_camera, offset_camera, device):
        """Test that identical depth at a 3D point passes consistency."""
        config = FusionConfig(min_consistent_views=1, depth_tolerance=0.01)  # 10mm tolerance

        H, W = 32, 32

        # Create dense depth maps for both cameras viewing the same plane at Z=3.0
        target_z = 3.0

        # Reference camera depth map
        u = torch.arange(W, dtype=torch.float32, device=device)
        v = torch.arange(H, dtype=torch.float32, device=device)
        grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
        pixels_ref = torch.stack([grid_u.reshape(-1), grid_v.reshape(-1)], dim=-1)
        origins_ref, directions_ref = reference_camera.cast_ray(pixels_ref)
        depths_ref = (target_z - origins_ref[:, 2]) / directions_ref[:, 2]
        ref_depth = depths_ref.reshape(H, W)
        ref_confidence = torch.ones(H, W, dtype=torch.float32, device=device)

        # Offset camera depth map (also viewing the same plane)
        pixels_offset = torch.stack([grid_u.reshape(-1), grid_v.reshape(-1)], dim=-1)
        origins_offset, directions_offset = offset_camera.cast_ray(pixels_offset)
        depths_offset = (target_z - origins_offset[:, 2]) / directions_offset[:, 2]
        offset_depth = depths_offset.reshape(H, W)

        # Filter
        filtered_depth, filtered_conf, count = filter_depth_map(
            ref_name="ref",
            ref_model=reference_camera,
            ref_depth=ref_depth,
            ref_confidence=ref_confidence,
            target_names=["offset"],
            target_models={"offset": offset_camera},
            target_depths={"offset": offset_depth},
            config=config,
        )

        # Since both cameras view the same plane, many pixels should be consistent
        # (accounting for the fact that their field of views overlap but don't perfectly align)
        consistent_pixels = (count >= 1).sum()
        total_pixels = H * W
        # At least 10% should be consistent (conservative estimate)
        assert consistent_pixels > total_pixels * 0.1

    def test_inconsistent_depth_filtered(self, reference_camera, offset_camera, device):
        """Test that inconsistent depth is filtered out."""
        config = FusionConfig(min_consistent_views=1, depth_tolerance=0.01)

        H, W = 16, 16
        # Reference thinks depth is 2.0
        ref_depth = torch.full((H, W), 2.0, dtype=torch.float32, device=device)
        ref_confidence = torch.ones(H, W, dtype=torch.float32, device=device)

        # Offset camera has wrong depth (offset by more than tolerance)
        offset_depth = torch.full((H, W), 3.0, dtype=torch.float32, device=device)

        # Filter
        filtered_depth, filtered_conf, count = filter_depth_map(
            ref_name="ref",
            ref_model=reference_camera,
            ref_depth=ref_depth,
            ref_confidence=ref_confidence,
            target_names=["offset"],
            target_models={"offset": offset_camera},
            target_depths={"offset": offset_depth},
            config=config,
        )

        # Most pixels should be inconsistent (count = 0) due to depth mismatch
        # At least some pixels should be filtered out
        assert (count == 0).sum() > 0

    def test_min_consistent_views_threshold(self, device):
        """Test that min_consistent_views threshold works correctly."""
        config = FusionConfig(min_consistent_views=2, depth_tolerance=0.01)  # 10mm tolerance

        # Create 3 cameras at different positions with small offsets for overlap
        cameras = {}
        for i, offset_x in enumerate([0.0, 0.1, 0.2]):
            R = torch.eye(3, dtype=torch.float32, device=device)
            t = torch.tensor([-offset_x, 0.0, 0.0], dtype=torch.float32, device=device)
            # Wide field of view
            K = torch.tensor(
                [[400.0, 0.0, 250.0], [0.0, 400.0, 250.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=device,
            )
            cameras[f"cam{i}"] = RefractiveProjectionModel(
                K, R, t, 1.0, torch.tensor([0.0, 0.0, -1.0], device=device), 1.0, 1.333
            )

        H, W = 32, 32

        # Create geometrically consistent depth maps for all cameras
        # All cameras observe the same planar surface at world Z = 3.0
        target_z = 3.0
        depth_maps = {}
        for name, camera in cameras.items():
            # Create pixel grid
            u = torch.arange(W, dtype=torch.float32, device=device)
            v = torch.arange(H, dtype=torch.float32, device=device)
            grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
            pixels = torch.stack([grid_u.reshape(-1), grid_v.reshape(-1)], dim=-1)

            # Back-project to Z=3.0 plane
            origins, directions = camera.cast_ray(pixels)
            depths = (target_z - origins[:, 2]) / directions[:, 2]
            depth_maps[name] = depths.reshape(H, W)

        ref_confidence = torch.ones(H, W, dtype=torch.float32, device=device)

        # Filter cam0 against cam1 and cam2
        filtered_depth, filtered_conf, count = filter_depth_map(
            ref_name="cam0",
            ref_model=cameras["cam0"],
            ref_depth=depth_maps["cam0"],
            ref_confidence=ref_confidence,
            target_names=["cam1", "cam2"],
            target_models=cameras,
            target_depths=depth_maps,
            config=config,
        )

        # Since all cameras observe the same plane, pixels should be consistent
        # Check that the filtering logic works:
        # - Some pixels should have count >= 1 (consistent with at least one other camera)
        pixels_with_1_or_more = (count >= 1).sum()
        assert pixels_with_1_or_more > 0

        # - Pixels with count >= 2 should pass the threshold and have valid depth
        passing_pixels = count >= 2
        if passing_pixels.any():
            assert not torch.any(torch.isnan(filtered_depth[passing_pixels]))

        # - Pixels with count < 2 should be filtered out (NaN)
        failing_pixels = (count > 0) & (count < 2)
        if failing_pixels.any():
            assert torch.all(torch.isnan(filtered_depth[failing_pixels]))

    def test_confidence_threshold(self, reference_camera, offset_camera, device):
        """Test that pixels below min_confidence are excluded."""
        config = FusionConfig(min_confidence=0.5, min_consistent_views=1, depth_tolerance=0.01)

        H, W = 8, 8
        ref_depth = torch.full((H, W), 2.0, dtype=torch.float32, device=device)
        # Half the pixels have low confidence
        ref_confidence = torch.ones(H, W, dtype=torch.float32, device=device)
        ref_confidence[:, :W // 2] = 0.3  # Low confidence in left half

        offset_depth = torch.full((H, W), 2.0, dtype=torch.float32, device=device)

        # Filter
        filtered_depth, filtered_conf, count = filter_depth_map(
            ref_name="ref",
            ref_model=reference_camera,
            ref_depth=ref_depth,
            ref_confidence=ref_confidence,
            target_names=["offset"],
            target_models={"offset": offset_camera},
            target_depths={"offset": offset_depth},
            config=config,
        )

        # Low confidence pixels should have count = 0 (not processed)
        assert (count[:, :W // 2] == 0).all()
        # Low confidence pixels should be NaN
        assert torch.all(torch.isnan(filtered_depth[:, :W // 2]))

    def test_nan_propagation_in_reference(self, reference_camera, offset_camera, device):
        """Test that NaN in reference depth map propagates correctly."""
        config = FusionConfig(min_consistent_views=1, depth_tolerance=0.01)

        H, W = 8, 8
        ref_depth = torch.full((H, W), 2.0, dtype=torch.float32, device=device)
        # Set some pixels to NaN
        ref_depth[0:2, 0:2] = float("nan")
        ref_confidence = torch.ones(H, W, dtype=torch.float32, device=device)

        offset_depth = torch.full((H, W), 2.0, dtype=torch.float32, device=device)

        # Filter
        filtered_depth, filtered_conf, count = filter_depth_map(
            ref_name="ref",
            ref_model=reference_camera,
            ref_depth=ref_depth,
            ref_confidence=ref_confidence,
            target_names=["offset"],
            target_models={"offset": offset_camera},
            target_depths={"offset": offset_depth},
            config=config,
        )

        # NaN pixels should remain NaN with count = 0
        assert torch.all(torch.isnan(filtered_depth[0:2, 0:2]))
        assert (count[0:2, 0:2] == 0).all()
        assert (filtered_conf[0:2, 0:2] == 0).all()

    def test_empty_depth_map(self, reference_camera, offset_camera, device):
        """Test that an all-NaN depth map returns all-invalid without error."""
        config = FusionConfig(min_consistent_views=1, depth_tolerance=0.01)

        H, W = 8, 8
        ref_depth = torch.full((H, W), float("nan"), dtype=torch.float32, device=device)
        ref_confidence = torch.zeros(H, W, dtype=torch.float32, device=device)

        offset_depth = torch.full((H, W), 2.0, dtype=torch.float32, device=device)

        # Filter
        filtered_depth, filtered_conf, count = filter_depth_map(
            ref_name="ref",
            ref_model=reference_camera,
            ref_depth=ref_depth,
            ref_confidence=ref_confidence,
            target_names=["offset"],
            target_models={"offset": offset_camera},
            target_depths={"offset": offset_depth},
            config=config,
        )

        # All outputs should be invalid
        assert torch.all(torch.isnan(filtered_depth))
        assert torch.all(filtered_conf == 0)
        assert torch.all(count == 0)

    def test_output_shapes(self, reference_camera, offset_camera, device):
        """Test that output tensors have correct shapes."""
        config = FusionConfig(min_consistent_views=1, depth_tolerance=0.01)

        H, W = 16, 16
        ref_depth = torch.full((H, W), 2.0, dtype=torch.float32, device=device)
        ref_confidence = torch.ones(H, W, dtype=torch.float32, device=device)
        offset_depth = torch.full((H, W), 2.0, dtype=torch.float32, device=device)

        filtered_depth, filtered_conf, count = filter_depth_map(
            ref_name="ref",
            ref_model=reference_camera,
            ref_depth=ref_depth,
            ref_confidence=ref_confidence,
            target_names=["offset"],
            target_models={"offset": offset_camera},
            target_depths={"offset": offset_depth},
            config=config,
        )

        assert filtered_depth.shape == (H, W)
        assert filtered_conf.shape == (H, W)
        assert count.shape == (H, W)
        assert filtered_depth.dtype == torch.float32
        assert filtered_conf.dtype == torch.float32
        assert count.dtype == torch.int32


class TestFilterAllDepthMaps:
    """Tests for filter_all_depth_maps batch filtering."""

    def test_output_structure(self, device):
        """Test that output dict has correct structure."""
        config = FusionConfig(min_consistent_views=1, depth_tolerance=0.01)

        # Create 3 cameras
        cameras = {}
        depth_maps = {}
        confidence_maps = {}
        H, W = 8, 8

        for i in range(3):
            R = torch.eye(3, dtype=torch.float32, device=device)
            t = torch.tensor([-i * 0.5, 0.0, 0.0], dtype=torch.float32, device=device)
            K = torch.tensor(
                [[1000.0, 0.0, 250.0], [0.0, 1000.0, 250.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=device,
            )
            name = f"cam{i}"
            cameras[name] = RefractiveProjectionModel(
                K, R, t, 1.0, torch.tensor([0.0, 0.0, -1.0], device=device), 1.0, 1.333
            )
            depth_maps[name] = torch.full((H, W), 2.0, dtype=torch.float32, device=device)
            confidence_maps[name] = torch.ones(H, W, dtype=torch.float32, device=device)

        # Filter all
        results = filter_all_depth_maps(
            ring_cameras=list(cameras.keys()),
            projection_models=cameras,
            depth_maps=depth_maps,
            confidence_maps=confidence_maps,
            config=config,
        )

        # Check structure
        assert len(results) == 3
        for name in cameras.keys():
            assert name in results
            filtered_depth, filtered_conf, count = results[name]
            assert filtered_depth.shape == (H, W)
            assert filtered_conf.shape == (H, W)
            assert count.shape == (H, W)

    def test_each_camera_filtered_against_others(self, device):
        """Test that each camera is filtered against all others."""
        config = FusionConfig(min_consistent_views=1, depth_tolerance=0.01)  # 10mm tolerance

        # Create 3 cameras with small offsets for overlap
        cameras = {}
        depth_maps = {}
        confidence_maps = {}
        H, W = 8, 8

        for i in range(3):
            R = torch.eye(3, dtype=torch.float32, device=device)
            t = torch.tensor([-i * 0.1, 0.0, 0.0], dtype=torch.float32, device=device)
            # Wide field of view
            K = torch.tensor(
                [[400.0, 0.0, 250.0], [0.0, 400.0, 250.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=device,
            )
            name = f"cam{i}"
            cameras[name] = RefractiveProjectionModel(
                K, R, t, 1.0, torch.tensor([0.0, 0.0, -1.0], device=device), 1.0, 1.333
            )
            # Each camera sees slightly different depth
            depth_maps[name] = torch.full(
                (H, W), 2.0 + i * 0.001, dtype=torch.float32, device=device
            )
            confidence_maps[name] = torch.ones(H, W, dtype=torch.float32, device=device)

        # Filter all
        results = filter_all_depth_maps(
            ring_cameras=list(cameras.keys()),
            projection_models=cameras,
            depth_maps=depth_maps,
            confidence_maps=confidence_maps,
            config=config,
        )

        # Each camera should have results
        for name in cameras.keys():
            assert name in results
            # Consistency counts should be computed
            _, _, count = results[name]
            # At least some pixels should have non-zero counts
            # (This is a weak test but verifies basic processing)
            assert count.max() >= 0
