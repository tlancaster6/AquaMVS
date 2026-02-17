"""Tests for sparse triangulation."""

import tempfile
from pathlib import Path

import pytest
import torch

from aquamvs.projection.refractive import RefractiveProjectionModel
from aquamvs.triangulation import (
    _triangulate_two_rays_batch,
    compute_depth_ranges,
    filter_sparse_cloud,
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

        # First 3 pairs are valid, last 2 are parallel (degenerate)
        targets = torch.rand(3, 3, device=device, dtype=torch.float32) * 5.0

        origins_a_valid = torch.zeros(3, 3, device=device, dtype=torch.float32)
        origins_b_valid = (
            torch.tensor([5.0, 0.0, 0.0], device=device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(3, 3)
        )

        dirs_a_valid = targets - origins_a_valid
        dirs_a_valid = dirs_a_valid / torch.linalg.norm(
            dirs_a_valid, dim=-1, keepdim=True
        )

        dirs_b_valid = targets - origins_b_valid
        dirs_b_valid = dirs_b_valid / torch.linalg.norm(
            dirs_b_valid, dim=-1, keepdim=True
        )

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

        points, valid = _triangulate_two_rays_batch(
            origins_a, dirs_a, origins_b, dirs_b
        )

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

    def test_triangulate_pair_rejects_parallel_rays(self, device):
        """Test that nearly parallel rays are rejected (intersection angle < min_angle)."""
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

        # Two cameras with very small baseline (0.01m) - will produce nearly parallel rays
        t1 = torch.zeros(3, device=device, dtype=torch.float32)
        t2 = torch.tensor([0.01, 0.0, 0.0], device=device, dtype=torch.float32)

        model1 = RefractiveProjectionModel(K, R, t1, water_z, normal, n_air, n_water)
        model2 = RefractiveProjectionModel(K, R, t2, water_z, normal, n_air, n_water)

        # Create a point far away (3m depth) - will make rays nearly parallel
        point_3d = torch.tensor([[0.005, 0.0, 3.0]], device=device, dtype=torch.float32)

        # Project through both models
        pixels1, valid1 = model1.project(point_3d)
        pixels2, valid2 = model2.project(point_3d)

        assert valid1.all() and valid2.all()

        # Create matches
        matches = {
            "ref_keypoints": pixels1,
            "src_keypoints": pixels2,
            "scores": torch.ones(1, device=device, dtype=torch.float32),
        }

        # Triangulate with default min_angle=2.0 degrees
        result = triangulate_pair(model1, model2, matches)

        # The point should be marked invalid due to small intersection angle
        assert not result["valid"][0], "Nearly parallel rays should be rejected"

    def test_triangulate_pair_accepts_good_angle(self, device):
        """Test that rays with sufficient convergence angle are accepted."""
        model1, model2 = self.create_test_models(device)

        # Create known 3D points underwater (baseline is 0.635m, so good angle)
        points_3d = torch.tensor(
            [[0.3, 0.2, 1.5], [0.0, 0.0, 1.4]],
            device=device,
            dtype=torch.float32,
        )

        # Project through both models
        pixels1, valid1 = model1.project(points_3d)
        pixels2, valid2 = model2.project(points_3d)

        assert valid1.all() and valid2.all()

        # Create matches
        matches = {
            "ref_keypoints": pixels1,
            "src_keypoints": pixels2,
            "scores": torch.ones(2, device=device, dtype=torch.float32),
        }

        # Triangulate
        result = triangulate_pair(model1, model2, matches)

        # All points should be valid (good angle with 0.635m baseline)
        assert result["valid"].all(), (
            "Points with good intersection angle should be accepted"
        )

    def test_triangulate_pair_rejects_bad_reproj(self, device):
        """Test that points with high reprojection error are rejected.

        Uses matches from two different 3D points to create inconsistent rays
        that produce high reprojection error.
        """
        model1, model2 = self.create_test_models(device)

        # Create two DIFFERENT 3D points
        point_3d_1 = torch.tensor([[0.3, 0.2, 1.5]], device=device, dtype=torch.float32)
        point_3d_2 = torch.tensor(
            [[0.1, -0.1, 1.3]], device=device, dtype=torch.float32
        )

        # Project first point through model1, second point through model2
        # This creates inconsistent matches (rays point to different locations)
        pixels1, valid1 = model1.project(point_3d_1)
        pixels2, valid2 = model2.project(point_3d_2)

        assert valid1.all() and valid2.all()

        # Create matches from inconsistent points
        matches = {
            "ref_keypoints": pixels1,
            "src_keypoints": pixels2,
            "scores": torch.ones(1, device=device, dtype=torch.float32),
        }

        # Triangulate with strict max_reproj_error=1.0 pixels
        result = triangulate_pair(model1, model2, matches, max_reproj_error=1.0)

        # The point should be marked invalid due to high reprojection error
        # (the triangulated point is a compromise and won't reproject well to either)
        assert not result["valid"][0], (
            "Inconsistent matches should produce high reprojection error"
        )

    def test_triangulate_pair_min_angle_parameter(self, device):
        """Test that min_angle parameter correctly controls angle filtering."""
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

        # Two cameras with small baseline
        t1 = torch.zeros(3, device=device, dtype=torch.float32)
        t2 = torch.tensor([0.01, 0.0, 0.0], device=device, dtype=torch.float32)

        model1 = RefractiveProjectionModel(K, R, t1, water_z, normal, n_air, n_water)
        model2 = RefractiveProjectionModel(K, R, t2, water_z, normal, n_air, n_water)

        # Point far away (nearly parallel rays)
        point_3d = torch.tensor([[0.005, 0.0, 3.0]], device=device, dtype=torch.float32)

        pixels1, valid1 = model1.project(point_3d)
        pixels2, valid2 = model2.project(point_3d)

        assert valid1.all() and valid2.all()

        matches = {
            "ref_keypoints": pixels1,
            "src_keypoints": pixels2,
            "scores": torch.ones(1, device=device, dtype=torch.float32),
        }

        # With min_angle=0.0 (disabled), should pass
        result_permissive = triangulate_pair(model1, model2, matches, min_angle=0.0)
        assert result_permissive["valid"][0], (
            "min_angle=0.0 should accept nearly parallel rays"
        )

        # With min_angle=10.0 (strict), should fail
        result_strict = triangulate_pair(model1, model2, matches, min_angle=10.0)
        assert not result_strict["valid"][0], (
            "min_angle=10.0 should reject nearly parallel rays"
        )

    def test_triangulate_pair_max_reproj_parameter(self, device):
        """Test that max_reproj_error parameter correctly controls reprojection filtering."""
        model1, model2 = self.create_test_models(device)

        # Create two different 3D points to produce inconsistent matches
        point_3d_1 = torch.tensor([[0.3, 0.2, 1.5]], device=device, dtype=torch.float32)
        point_3d_2 = torch.tensor(
            [[0.15, 0.0, 1.4]], device=device, dtype=torch.float32
        )

        # Project different points through each camera
        pixels1, valid1 = model1.project(point_3d_1)
        pixels2, valid2 = model2.project(point_3d_2)

        assert valid1.all() and valid2.all()

        matches = {
            "ref_keypoints": pixels1,
            "src_keypoints": pixels2,
            "scores": torch.ones(1, device=device, dtype=torch.float32),
        }

        # With max_reproj_error=100.0 (permissive), should pass
        result_permissive = triangulate_pair(
            model1, model2, matches, max_reproj_error=100.0
        )
        assert result_permissive["valid"][0], (
            "Large max_reproj_error should accept inconsistent matches"
        )

        # With max_reproj_error=0.5 (strict), should fail
        result_strict = triangulate_pair(model1, model2, matches, max_reproj_error=0.5)
        assert not result_strict["valid"][0], (
            "Small max_reproj_error should reject inconsistent matches"
        )


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
            models[f"cam{i}"] = RefractiveProjectionModel(
                K, R, t, water_z, normal, n_air, n_water
            )

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


class TestFilterSparseCloud:
    """Tests for filter_sparse_cloud() function."""

    def test_filter_sparse_cloud_removes_above_water(self, device):
        """Test that points above water surface (Z < water_z) are removed."""
        water_z = 1.0

        # Create points with some above water (Z < 1.0)
        points_3d = torch.tensor(
            [
                [0.0, 0.0, 0.5],  # Above water - should be removed
                [0.0, 0.0, 0.8],  # Above water - should be removed
                [0.0, 0.0, 1.2],  # Below water - should be kept
                [0.0, 0.0, 1.5],  # Below water - should be kept
            ],
            device=device,
            dtype=torch.float32,
        )

        scores = torch.ones(4, device=device, dtype=torch.float32)

        sparse_cloud = {"points_3d": points_3d, "scores": scores}

        # Filter
        filtered = filter_sparse_cloud(sparse_cloud, water_z=water_z, max_depth=2.0)

        # Should keep only the last 2 points
        assert filtered["points_3d"].shape[0] == 2
        assert filtered["scores"].shape[0] == 2
        assert (filtered["points_3d"][:, 2] > water_z).all()

    def test_filter_sparse_cloud_keeps_underwater(self, device):
        """Test that points underwater within max_depth are kept."""
        water_z = 1.0

        # Create points all underwater and within range
        points_3d = torch.tensor(
            [
                [0.0, 0.0, 1.1],  # water_z < Z < water_z + max_depth
                [0.0, 0.0, 1.5],
                [0.0, 0.0, 2.5],  # Z < water_z + max_depth
            ],
            device=device,
            dtype=torch.float32,
        )

        scores = torch.ones(3, device=device, dtype=torch.float32)

        sparse_cloud = {"points_3d": points_3d, "scores": scores}

        # Filter with max_depth=2.0 (so range is [1.0, 3.0])
        filtered = filter_sparse_cloud(sparse_cloud, water_z=water_z, max_depth=2.0)

        # Should keep all 3 points
        assert filtered["points_3d"].shape[0] == 3
        assert filtered["scores"].shape[0] == 3

    def test_filter_sparse_cloud_removes_deep_outliers(self, device):
        """Test that points below water_z + max_depth are removed."""
        water_z = 1.0

        # Create points with some deep outliers
        points_3d = torch.tensor(
            [
                [0.0, 0.0, 1.5],  # Good
                [0.0, 0.0, 2.0],  # Good
                [0.0, 0.0, 10.0],  # Too deep - should be removed
                [0.0, 0.0, 25.0],  # Way too deep - should be removed
            ],
            device=device,
            dtype=torch.float32,
        )

        scores = torch.tensor([1.0, 1.0, 0.5, 0.3], device=device, dtype=torch.float32)

        sparse_cloud = {"points_3d": points_3d, "scores": scores}

        # Filter with max_depth=2.0 (so range is [1.0, 3.0])
        filtered = filter_sparse_cloud(sparse_cloud, water_z=water_z, max_depth=2.0)

        # Should keep only the first 2 points
        assert filtered["points_3d"].shape[0] == 2
        assert filtered["scores"].shape[0] == 2
        assert (filtered["points_3d"][:, 2] < water_z + 2.0).all()

    def test_filter_sparse_cloud_mixed(self, device):
        """Test filtering with a mix of valid and invalid points."""
        water_z = 1.0

        # Mix of above-water, good, and deep outliers
        points_3d = torch.tensor(
            [
                [0.0, 0.0, 0.5],  # Above water - invalid
                [0.0, 0.0, 1.2],  # Good - valid
                [0.0, 0.0, 2.5],  # Good - valid
                [0.0, 0.0, 10.0],  # Too deep - invalid
                [0.0, 0.0, 1.8],  # Good - valid
            ],
            device=device,
            dtype=torch.float32,
        )

        scores = torch.tensor(
            [0.1, 0.8, 0.9, 0.2, 0.7], device=device, dtype=torch.float32
        )

        sparse_cloud = {"points_3d": points_3d, "scores": scores}

        # Filter
        filtered = filter_sparse_cloud(sparse_cloud, water_z=water_z, max_depth=2.0)

        # Should keep indices 1, 2, 4 (the good ones)
        assert filtered["points_3d"].shape[0] == 3
        assert filtered["scores"].shape[0] == 3

        # Check that the scores match the kept points
        expected_scores = torch.tensor(
            [0.8, 0.9, 0.7], device=device, dtype=torch.float32
        )
        torch.testing.assert_close(filtered["scores"], expected_scores)

    def test_filter_sparse_cloud_empty(self, device):
        """Test that empty input returns empty output."""
        sparse_cloud = {
            "points_3d": torch.empty(0, 3, device=device, dtype=torch.float32),
            "scores": torch.empty(0, device=device, dtype=torch.float32),
        }

        filtered = filter_sparse_cloud(sparse_cloud, water_z=1.0, max_depth=2.0)

        assert filtered["points_3d"].shape[0] == 0
        assert filtered["scores"].shape[0] == 0

    def test_filter_sparse_cloud_all_removed(self, device):
        """Test that when all points are invalid, output is empty."""
        water_z = 1.0

        # All points are above water
        points_3d = torch.tensor(
            [
                [0.0, 0.0, 0.3],
                [0.0, 0.0, 0.5],
                [0.0, 0.0, 0.9],
            ],
            device=device,
            dtype=torch.float32,
        )

        scores = torch.ones(3, device=device, dtype=torch.float32)

        sparse_cloud = {"points_3d": points_3d, "scores": scores}

        filtered = filter_sparse_cloud(sparse_cloud, water_z=water_z, max_depth=2.0)

        # All should be removed
        assert filtered["points_3d"].shape[0] == 0
        assert filtered["scores"].shape[0] == 0


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
            models[f"cam{i}"] = RefractiveProjectionModel(
                K, R, t, water_z, normal, n_air, n_water
            )

        return models

    def test_synthetic_cloud(self, device):
        """Test depth range computation with synthetic point cloud.

        Uses a larger point set (100 points) so that percentile-based ranges
        (2nd/98th percentile) are stable and the range contains most points.
        """
        models = self.create_test_models(device)

        # Create sparse cloud with many points (for stable percentiles)
        # Random points in a reasonable Z range
        torch.manual_seed(42)  # For reproducibility
        points_3d = torch.rand(100, 3, device=device, dtype=torch.float32)
        points_3d[:, 0] = points_3d[:, 0] * 0.4 - 0.2  # X in [-0.2, 0.2]
        points_3d[:, 1] = points_3d[:, 1] * 0.4 - 0.2  # Y in [-0.2, 0.2]
        points_3d[:, 2] = points_3d[:, 2] * 0.5 + 1.2  # Z in [1.2, 1.7]

        sparse_cloud = {
            "points_3d": points_3d,
            "scores": torch.ones(100, device=device, dtype=torch.float32),
        }

        margin = 0.05
        depth_ranges = compute_depth_ranges(models, sparse_cloud, margin)

        # Check that all cameras have ranges
        assert "cam0" in depth_ranges
        assert "cam1" in depth_ranges

        # For each camera, verify that the range is reasonable
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

                # With percentile clipping (2nd/98th), most depths should be within range
                # At least 90% of points should be within the range
                within_range = (depths >= d_min) & (depths <= d_max)
                fraction_within = within_range.float().mean().item()
                assert fraction_within >= 0.90, (
                    f"At least 90% of points should be within range, got {fraction_within:.1%}"
                )

                # The median should definitely be within range
                median_depth = depths.median().item()
                assert d_min <= median_depth <= d_max, (
                    f"Median depth {median_depth:.3f} should be within [{d_min:.3f}, {d_max:.3f}]"
                )

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

    def test_depth_ranges_outlier_robust(self, device):
        """Test that percentile-based depth ranges are robust to outliers."""
        models = self.create_test_models(device)

        # Create a sparse cloud with 99 points near Z=1.4 and 1 extreme outlier
        normal_points = torch.rand(99, 3, device=device, dtype=torch.float32) * 0.2
        normal_points[:, 2] += 1.4  # Z around 1.4

        # Add an extreme outlier at Z=25.0
        outlier_point = torch.tensor(
            [[0.0, 0.0, 25.0]], device=device, dtype=torch.float32
        )

        points_3d = torch.cat([normal_points, outlier_point], dim=0)

        sparse_cloud = {
            "points_3d": points_3d,
            "scores": torch.ones(100, device=device, dtype=torch.float32),
        }

        margin = 0.05
        depth_ranges = compute_depth_ranges(models, sparse_cloud, margin)

        # For cam0, compute what the depth of the outlier would be
        outlier_pixel, outlier_valid = models["cam0"].project(outlier_point)
        if outlier_valid[0]:
            origin, direction = models["cam0"].cast_ray(outlier_pixel)
            diff = outlier_point - origin
            outlier_depth = (diff * direction).sum(dim=-1).item()

            # The d_max should be MUCH less than the outlier depth
            # (because we use 98th percentile, not max)
            d_min, d_max = depth_ranges["cam0"]

            # With percentile clipping, d_max should be significantly less than outlier_depth
            # The outlier is at the top 1%, so it should be excluded
            assert d_max < outlier_depth - 1.0, (
                f"Percentile-based range should exclude outlier: "
                f"d_max={d_max:.2f} should be much less than outlier_depth={outlier_depth:.2f}"
            )

    def test_camera_with_no_visible_points(self, device):
        """Test that camera with no visible points gets fallback range."""
        water_z = 0.978
        normal = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=torch.float32)
        n_air = 1.0
        n_water = 1.333

        K = torch.tensor(
            [[800.0, 0.0, 800.0], [0.0, 800.0, 600.0], [0.0, 0.0, 1.0]],
            device=device,
            dtype=torch.float32,
        )

        # Camera 0: looking down (+Z), at origin - will see the points
        R0 = torch.eye(3, device=device, dtype=torch.float32)
        t0 = torch.zeros(3, device=device, dtype=torch.float32)
        model0 = RefractiveProjectionModel(K, R0, t0, water_z, normal, n_air, n_water)

        # Camera 1: looking backward (-Z), far from origin - will NOT see the points
        # Rotate 180 degrees around Y axis to look backward
        R1 = torch.tensor(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            device=device,
            dtype=torch.float32,
        )
        # Position it far to the side so points are behind it
        t1 = torch.tensor([5.0, 0.0, 0.0], device=device, dtype=torch.float32)
        model1 = RefractiveProjectionModel(K, R1, t1, water_z, normal, n_air, n_water)

        models = {"cam0": model0, "cam1": model1}

        # Create sparse cloud with points near origin, underwater
        points_3d = torch.tensor(
            [[0.0, 0.0, 1.2], [0.1, 0.1, 1.5], [-0.1, -0.1, 1.3]],
            device=device,
            dtype=torch.float32,
        )

        sparse_cloud = {
            "points_3d": points_3d,
            "scores": torch.ones(3, device=device, dtype=torch.float32),
        }

        # Verify cam0 can see the points but cam1 cannot
        pixels0, valid0 = model0.project(points_3d)
        pixels1, valid1 = model1.project(points_3d)

        assert valid0.any(), "cam0 should see at least one point"
        assert not valid1.any(), "cam1 should see no points (all behind camera)"

        # Compute depth ranges
        margin = 0.05
        depth_ranges = compute_depth_ranges(models, sparse_cloud, margin)

        # cam0 should have a real range based on visible points
        d_min_0, d_max_0 = depth_ranges["cam0"]
        assert d_min_0 >= 0.0
        # Verify it's not the fallback range
        assert d_min_0 != 0.3 or d_max_0 != 0.9

        # cam1 should have the fallback range
        d_min_1, d_max_1 = depth_ranges["cam1"]
        assert d_min_1 == 0.3
        assert d_max_1 == 0.9


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
