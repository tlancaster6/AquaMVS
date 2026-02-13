"""Tests for sparse triangulation."""

import tempfile
from pathlib import Path

import pytest
import torch

from aquamvs.projection.refractive import RefractiveProjectionModel
from aquamvs.triangulation import (
    _triangulate_two_rays_batch,
    compute_depth_ranges,
    load_sparse_cloud,
    save_sparse_cloud,
    triangulate_all_pairs,
    triangulate_pair,
    triangulate_rays,
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Parametrize tests over CPU and CUDA devices."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


class TestTriangulateRays:
    """Tests for triangulate_rays() function."""

    def test_two_rays_intersection(self, device):
        """Test triangulation with two rays that intersect at a known point."""
        # Target point
        target = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=torch.float32)

        # Ray 1: origin at (0, 0, 0), direction toward target
        origin1 = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)
        direction1 = target - origin1
        direction1 = direction1 / torch.linalg.norm(direction1)

        # Ray 2: origin at (5, 0, 0), direction toward target
        origin2 = torch.tensor([5.0, 0.0, 0.0], device=device, dtype=torch.float32)
        direction2 = target - origin2
        direction2 = direction2 / torch.linalg.norm(direction2)

        rays = [(origin1, direction1), (origin2, direction2)]
        result = triangulate_rays(rays)

        torch.testing.assert_close(result, target, atol=1e-5, rtol=0)

    def test_three_rays_intersection(self, device):
        """Test triangulation with three rays."""
        target = torch.tensor([2.0, 1.0, 4.0], device=device, dtype=torch.float32)

        # Three different origins, all pointing at target
        origins = [
            torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32),
            torch.tensor([4.0, 0.0, 0.0], device=device, dtype=torch.float32),
            torch.tensor([0.0, 3.0, 0.0], device=device, dtype=torch.float32),
        ]

        rays = []
        for origin in origins:
            direction = target - origin
            direction = direction / torch.linalg.norm(direction)
            rays.append((origin, direction))

        result = triangulate_rays(rays)
        torch.testing.assert_close(result, target, atol=1e-5, rtol=0)

    def test_four_rays_intersection(self, device):
        """Test triangulation with four rays."""
        target = torch.tensor([1.5, 2.5, 3.5], device=device, dtype=torch.float32)

        origins = [
            torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32),
            torch.tensor([5.0, 0.0, 0.0], device=device, dtype=torch.float32),
            torch.tensor([0.0, 5.0, 0.0], device=device, dtype=torch.float32),
            torch.tensor([0.0, 0.0, 5.0], device=device, dtype=torch.float32),
        ]

        rays = []
        for origin in origins:
            direction = target - origin
            direction = direction / torch.linalg.norm(direction)
            rays.append((origin, direction))

        result = triangulate_rays(rays)
        torch.testing.assert_close(result, target, atol=1e-5, rtol=0)

    def test_parallel_rays_raises(self, device):
        """Test that parallel rays raise ValueError."""
        # Two parallel rays with same direction but different origins
        direction = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
        origin1 = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)
        origin2 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)

        rays = [(origin1, direction), (origin2, direction)]

        with pytest.raises(ValueError, match="Degenerate ray configuration"):
            triangulate_rays(rays)

    def test_single_ray_raises(self, device):
        """Test that a single ray raises ValueError."""
        origin = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)
        direction = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)

        rays = [(origin, direction)]

        with pytest.raises(ValueError, match="Need at least 2 rays"):
            triangulate_rays(rays)


class TestTriangulateTwoRaysBatch:
    """Tests for _triangulate_two_rays_batch() function."""

    def test_batch_accuracy(self, device):
        """Test that batch results match individual triangulate_rays calls."""
        # Create M pairs of rays with known intersection points
        M = 10
        targets = torch.rand(M, 3, device=device, dtype=torch.float32) * 5.0

        origins_a = torch.zeros(M, 3, device=device, dtype=torch.float32)
        origins_b = torch.tensor([5.0, 0.0, 0.0], device=device, dtype=torch.float32)
        origins_b = origins_b.unsqueeze(0).expand(M, 3)

        dirs_a = targets - origins_a
        dirs_a = dirs_a / torch.linalg.norm(dirs_a, dim=-1, keepdim=True)

        dirs_b = targets - origins_b
        dirs_b = dirs_b / torch.linalg.norm(dirs_b, dim=-1, keepdim=True)

        # Batch triangulation
        points_batch, valid_batch = _triangulate_two_rays_batch(
            origins_a, dirs_a, origins_b, dirs_b
        )

        # Individual triangulation
        for i in range(M):
            if valid_batch[i]:
                rays = [(origins_a[i], dirs_a[i]), (origins_b[i], dirs_b[i])]
                point_individual = triangulate_rays(rays)
                torch.testing.assert_close(
                    points_batch[i], point_individual, atol=1e-5, rtol=0
                )

        # All should be valid for this setup
        assert valid_batch.all()

    def test_degenerate_pairs(self, device):
        """Test that degenerate pairs are marked as invalid."""
        M = 5

        # First 3 pairs are valid, last 2 are parallel (degenerate)
        targets = torch.rand(3, 3, device=device, dtype=torch.float32) * 5.0

        origins_a_valid = torch.zeros(3, 3, device=device, dtype=torch.float32)
        origins_b_valid = (
            torch.tensor([5.0, 0.0, 0.0], device=device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(3, 3)
        )

        dirs_a_valid = targets - origins_a_valid
        dirs_a_valid = dirs_a_valid / torch.linalg.norm(dirs_a_valid, dim=-1, keepdim=True)

        dirs_b_valid = targets - origins_b_valid
        dirs_b_valid = dirs_b_valid / torch.linalg.norm(dirs_b_valid, dim=-1, keepdim=True)

        # Degenerate pairs: parallel directions
        direction_parallel = (
            torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(2, 3)
        )
        origins_a_degen = torch.zeros(2, 3, device=device, dtype=torch.float32)
        origins_b_degen = (
            torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(2, 3)
        )

        # Concatenate
        origins_a = torch.cat([origins_a_valid, origins_a_degen], dim=0)
        origins_b = torch.cat([origins_b_valid, origins_b_degen], dim=0)
        dirs_a = torch.cat([dirs_a_valid, direction_parallel], dim=0)
        dirs_b = torch.cat([dirs_b_valid, direction_parallel], dim=0)

        points, valid = _triangulate_two_rays_batch(origins_a, dirs_a, origins_b, dirs_b)

        # First 3 should be valid, last 2 invalid
        assert valid[0] and valid[1] and valid[2]
        assert not valid[3] and not valid[4]


class TestTriangulatePair:
    """Tests for triangulate_pair() function."""

    def create_test_models(self, device):
        """Create two synthetic refractive projection models."""
        # Reference geometry from DESIGN.md
        water_z = 0.978
        normal = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=torch.float32)
        n_air = 1.0
        n_water = 1.333

        # Camera 1: at origin looking down (+Z)
        K1 = torch.tensor(
            [[800.0, 0.0, 800.0], [0.0, 800.0, 600.0], [0.0, 0.0, 1.0]],
            device=device,
            dtype=torch.float32,
        )
        R1 = torch.eye(3, device=device, dtype=torch.float32)
        t1 = torch.zeros(3, device=device, dtype=torch.float32)

        # Camera 2: offset in X, looking down
        K2 = K1.clone()
        R2 = torch.eye(3, device=device, dtype=torch.float32)
        t2 = torch.tensor([0.635, 0.0, 0.0], device=device, dtype=torch.float32)

        model1 = RefractiveProjectionModel(K1, R1, t1, water_z, normal, n_air, n_water)
        model2 = RefractiveProjectionModel(K2, R2, t2, water_z, normal, n_air, n_water)

        return model1, model2

    def test_synthetic_matches(self, device):
        """Test triangulation with synthetic matches from known 3D points."""
        model1, model2 = self.create_test_models(device)

        # Create known 3D points underwater (Z > water_z)
        points_3d_true = torch.tensor(
            [[0.3, 0.2, 1.5], [0.0, 0.0, 1.4], [-0.2, 0.1, 1.3]],
            device=device,
            dtype=torch.float32,
        )

        # Project through both models
        pixels1, valid1 = model1.project(points_3d_true)
        pixels2, valid2 = model2.project(points_3d_true)

        # All should be valid
        assert valid1.all() and valid2.all()

        # Create matches dict
        matches = {
            "ref_keypoints": pixels1,
            "src_keypoints": pixels2,
            "scores": torch.ones(3, device=device, dtype=torch.float32),
        }

        # Triangulate
        result = triangulate_pair(model1, model2, matches)

        # Check structure
        assert result["points_3d"].shape == (3, 3)
        assert result["scores"].shape == (3,)
        assert result["ref_pixels"].shape == (3, 2)
        assert result["src_pixels"].shape == (3, 2)
        assert result["valid"].shape == (3,)

        # All should be valid
        assert result["valid"].all()

        # Recovered points should match original within tolerance
        torch.testing.assert_close(
            result["points_3d"], points_3d_true, atol=1e-5, rtol=0
        )

    def test_empty_matches(self, device):
        """Test that empty matches return empty tensors."""
        model1, model2 = self.create_test_models(device)

        matches = {
            "ref_keypoints": torch.empty(0, 2, device=device, dtype=torch.float32),
            "src_keypoints": torch.empty(0, 2, device=device, dtype=torch.float32),
            "scores": torch.empty(0, device=device, dtype=torch.float32),
        }

        result = triangulate_pair(model1, model2, matches)

        assert result["points_3d"].shape == (0, 3)
        assert result["scores"].shape == (0,)
        assert result["ref_pixels"].shape == (0, 2)
        assert result["src_pixels"].shape == (0, 2)
        assert result["valid"].shape == (0,)


class TestTriangulateAllPairs:
    """Tests for triangulate_all_pairs() function."""

    def create_test_models(self, device):
        """Create test projection models."""
        water_z = 0.978
        normal = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=torch.float32)
        n_air = 1.0
        n_water = 1.333

        K = torch.tensor(
            [[800.0, 0.0, 800.0], [0.0, 800.0, 600.0], [0.0, 0.0, 1.0]],
            device=device,
            dtype=torch.float32,
        )
        R = torch.eye(3, device=device, dtype=torch.float32)

        models = {}
        for i, x_offset in enumerate([0.0, 0.3, 0.6]):
            t = torch.tensor([x_offset, 0.0, 0.0], device=device, dtype=torch.float32)
            models[f"cam{i}"] = RefractiveProjectionModel(K, R, t, water_z, normal, n_air, n_water)

        return models

    def test_merge_all_pairs(self, device):
        """Test that all pairs are merged correctly."""
        models = self.create_test_models(device)

        # Create synthetic matches for pairs (0,1) and (0,2)
        points_pair01 = torch.tensor(
            [[0.0, 0.0, 1.4], [0.1, 0.1, 1.3]], device=device, dtype=torch.float32
        )
        points_pair02 = torch.tensor(
            [[0.2, 0.0, 1.5]], device=device, dtype=torch.float32
        )

        # Project and create matches
        pixels_01_ref, _ = models["cam0"].project(points_pair01)
        pixels_01_src, _ = models["cam1"].project(points_pair01)

        pixels_02_ref, _ = models["cam0"].project(points_pair02)
        pixels_02_src, _ = models["cam2"].project(points_pair02)

        all_matches = {
            ("cam0", "cam1"): {
                "ref_keypoints": pixels_01_ref,
                "src_keypoints": pixels_01_src,
                "scores": torch.ones(2, device=device, dtype=torch.float32),
            },
            ("cam0", "cam2"): {
                "ref_keypoints": pixels_02_ref,
                "src_keypoints": pixels_02_src,
                "scores": torch.ones(1, device=device, dtype=torch.float32),
            },
        }

        result = triangulate_all_pairs(models, all_matches)

        # Should have 3 total points (2 + 1)
        assert result["points_3d"].shape == (3, 3)
        assert result["scores"].shape == (3,)

    def test_empty_pairs(self, device):
        """Test with no matches."""
        models = self.create_test_models(device)
        all_matches = {}

        result = triangulate_all_pairs(models, all_matches)

        assert result["points_3d"].shape == (0, 3)
        assert result["scores"].shape == (0,)


class TestComputeDepthRanges:
    """Tests for compute_depth_ranges() function."""

    def create_test_models(self, device):
        """Create test projection models."""
        water_z = 0.978
        normal = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=torch.float32)
        n_air = 1.0
        n_water = 1.333

        K = torch.tensor(
            [[800.0, 0.0, 800.0], [0.0, 800.0, 600.0], [0.0, 0.0, 1.0]],
            device=device,
            dtype=torch.float32,
        )
        R = torch.eye(3, device=device, dtype=torch.float32)

        models = {}
        for i, x_offset in enumerate([0.0, 0.3]):
            t = torch.tensor([x_offset, 0.0, 0.0], device=device, dtype=torch.float32)
            models[f"cam{i}"] = RefractiveProjectionModel(K, R, t, water_z, normal, n_air, n_water)

        return models

    def test_synthetic_cloud(self, device):
        """Test depth range computation with synthetic point cloud."""
        models = self.create_test_models(device)

        # Create sparse cloud with known points
        points_3d = torch.tensor(
            [[0.0, 0.0, 1.2], [0.1, 0.1, 1.5], [0.05, 0.05, 1.3]],
            device=device,
            dtype=torch.float32,
        )

        sparse_cloud = {
            "points_3d": points_3d,
            "scores": torch.ones(3, device=device, dtype=torch.float32),
        }

        margin = 0.05
        depth_ranges = compute_depth_ranges(models, sparse_cloud, margin)

        # Check that all cameras have ranges
        assert "cam0" in depth_ranges
        assert "cam1" in depth_ranges

        # For each camera, verify that the range contains all visible points
        for cam_name, model in models.items():
            d_min, d_max = depth_ranges[cam_name]

            # Project points and cast rays
            pixels, valid = model.project(points_3d)
            if valid.any():
                valid_pixels = pixels[valid]
                valid_points = points_3d[valid]

                origins, directions = model.cast_ray(valid_pixels)
                diff = valid_points - origins
                depths = (diff * directions).sum(dim=-1)

                # All depths should be within [d_min, d_max]
                assert (depths >= d_min).all()
                assert (depths <= d_max).all()

                # Check margin was applied correctly
                assert d_min <= depths.min().item() - margin + 1e-5
                assert d_max >= depths.max().item() + margin - 1e-5

            # d_min should be >= 0
            assert d_min >= 0.0

    def test_empty_cloud(self, device):
        """Test that empty cloud returns fallback ranges."""
        models = self.create_test_models(device)

        sparse_cloud = {
            "points_3d": torch.empty(0, 3, device=device, dtype=torch.float32),
            "scores": torch.empty(0, device=device, dtype=torch.float32),
        }

        depth_ranges = compute_depth_ranges(models, sparse_cloud, margin=0.05)

        # Should return fallback range (0.3, 0.9) for all cameras
        for cam_name in models.keys():
            assert cam_name in depth_ranges
            d_min, d_max = depth_ranges[cam_name]
            assert d_min == 0.3
            assert d_max == 0.9


class TestSaveLoadSparseCloud:
    """Tests for save/load sparse cloud functions."""

    def test_roundtrip(self, device):
        """Test save and load roundtrip."""
        cloud = {
            "points_3d": torch.rand(100, 3, device=device, dtype=torch.float32),
            "scores": torch.rand(100, device=device, dtype=torch.float32),
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save
            save_sparse_cloud(cloud, temp_path)

            # Load
            loaded_cloud = load_sparse_cloud(temp_path)

            # Compare
            torch.testing.assert_close(loaded_cloud["points_3d"], cloud["points_3d"])
            torch.testing.assert_close(loaded_cloud["scores"], cloud["scores"])

            # Device should be preserved when loading
            assert loaded_cloud["points_3d"].device == cloud["points_3d"].device
            assert loaded_cloud["scores"].device == cloud["scores"].device
        finally:
            if temp_path.exists():
                temp_path.unlink()
