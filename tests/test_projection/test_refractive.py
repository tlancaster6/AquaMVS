"""Tests for RefractiveProjectionModel."""

import math

import pytest
import torch

from aquamvs.projection import ProjectionModel, RefractiveProjectionModel


@pytest.fixture
def device():
    """Device for tests (CPU only for now, CUDA when available)."""
    return "cpu"


@pytest.fixture
def simple_camera(device):
    """Simple camera at world origin looking straight down."""
    # Identity rotation (camera frame = world frame)
    R = torch.eye(3, dtype=torch.float32, device=device)
    # Zero translation (camera at world origin)
    t = torch.zeros(3, dtype=torch.float32, device=device)
    # Simple intrinsics: focal length 1000, principal point at (500, 500)
    K = torch.tensor(
        [[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    return K, R, t


@pytest.fixture
def refractive_model_simple(simple_camera, device):
    """RefractiveProjectionModel with simple camera at origin."""
    K, R, t = simple_camera
    water_z = 1.0  # 1 meter below camera
    normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
    n_air = 1.0
    n_water = 1.333
    return RefractiveProjectionModel(K, R, t, water_z, normal, n_air, n_water)


class TestRefractiveProjectionModelInit:
    """Tests for RefractiveProjectionModel initialization."""

    def test_constructor_stores_parameters(self, simple_camera, device):
        """Test that constructor stores all parameters."""
        K, R, t = simple_camera
        water_z = 1.0
        normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
        n_air = 1.0
        n_water = 1.333

        model = RefractiveProjectionModel(K, R, t, water_z, normal, n_air, n_water)

        assert torch.allclose(model.K, K)
        assert torch.allclose(model.R, R)
        assert torch.allclose(model.t, t)
        assert model.water_z == water_z
        assert torch.allclose(model.normal, normal)
        assert model.n_air == n_air
        assert model.n_water == n_water

    def test_constructor_precomputes_K_inv(self, simple_camera, device):
        """Test that constructor precomputes K_inv."""
        K, R, t = simple_camera
        water_z = 1.0
        normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
        n_air = 1.0
        n_water = 1.333

        model = RefractiveProjectionModel(K, R, t, water_z, normal, n_air, n_water)

        # Verify K_inv is the inverse of K
        identity = model.K @ model.K_inv
        expected_identity = torch.eye(3, dtype=torch.float32, device=device)
        assert torch.allclose(identity, expected_identity, atol=1e-6)

    def test_constructor_precomputes_camera_center(self, simple_camera, device):
        """Test that constructor precomputes camera center C = -R^T @ t."""
        K, R, t = simple_camera
        water_z = 1.0
        normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
        n_air = 1.0
        n_water = 1.333

        model = RefractiveProjectionModel(K, R, t, water_z, normal, n_air, n_water)

        # For identity R and zero t, C should be zero
        expected_C = torch.zeros(3, dtype=torch.float32, device=device)
        assert torch.allclose(model.C, expected_C)

        # Test with non-zero translation
        t_nonzero = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
        model2 = RefractiveProjectionModel(
            K, R, t_nonzero, water_z, normal, n_air, n_water
        )
        expected_C2 = -R.T @ t_nonzero
        assert torch.allclose(model2.C, expected_C2)

    def test_constructor_precomputes_n_ratio(self, simple_camera, device):
        """Test that constructor precomputes n_ratio = n_air / n_water."""
        K, R, t = simple_camera
        water_z = 1.0
        normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
        n_air = 1.0
        n_water = 1.333

        model = RefractiveProjectionModel(K, R, t, water_z, normal, n_air, n_water)

        expected_n_ratio = n_air / n_water
        assert abs(model.n_ratio - expected_n_ratio) < 1e-6


class TestRefractiveProjectionModelCastRay:
    """Tests for RefractiveProjectionModel.cast_ray method."""

    def test_nadir_ray_principal_point(self, refractive_model_simple, device):
        """Test ray at principal point (straight down, no refraction)."""
        # Principal point at (500, 500)
        pixels = torch.tensor([[500.0, 500.0]], dtype=torch.float32, device=device)

        origins, directions = refractive_model_simple.cast_ray(pixels)

        # Origin should be at (0, 0, water_z)
        expected_origin = torch.tensor(
            [[0.0, 0.0, 1.0]], dtype=torch.float32, device=device
        )
        assert torch.allclose(origins, expected_origin, atol=1e-5)

        # Direction should be straight down (0, 0, 1) - no refraction at normal incidence
        expected_direction = torch.tensor(
            [[0.0, 0.0, 1.0]], dtype=torch.float32, device=device
        )
        assert torch.allclose(directions, expected_direction, atol=1e-6)

    def test_origins_on_water_surface(self, refractive_model_simple, device):
        """Test that all ray origins lie on the water surface."""
        # Multiple random pixels
        torch.manual_seed(42)
        pixels = torch.rand(100, 2, dtype=torch.float32, device=device) * 1000

        origins, _ = refractive_model_simple.cast_ray(pixels)

        # All Z coordinates should equal water_z
        assert torch.allclose(
            origins[:, 2],
            torch.full((100,), 1.0, dtype=torch.float32, device=device),
            atol=1e-5,
        )

    def test_directions_are_unit_vectors(self, refractive_model_simple, device):
        """Test that all ray directions are unit vectors."""
        # Multiple random pixels
        torch.manual_seed(42)
        pixels = torch.rand(100, 2, dtype=torch.float32, device=device) * 1000

        _, directions = refractive_model_simple.cast_ray(pixels)

        # Compute norms
        norms = torch.linalg.norm(directions, dim=-1)

        # All norms should be 1.0
        assert torch.allclose(
            norms, torch.ones(100, dtype=torch.float32, device=device), atol=1e-6
        )

    def test_off_axis_ray_refraction(self, refractive_model_simple, device):
        """Test that off-axis rays refract toward the normal."""
        # Pixel offset from principal point
        # At (600, 500), the ray is offset in +X direction
        pixels = torch.tensor([[600.0, 500.0]], dtype=torch.float32, device=device)

        origins, directions = refractive_model_simple.cast_ray(pixels)

        # The refracted ray should have positive X component (same side as incident)
        # but smaller angle to vertical (refraction toward normal)
        assert directions[0, 0] > 0, "Ray should maintain X direction sign"
        assert directions[0, 2] > 0, "Ray should point into water (+Z)"

        # The incident ray in air has angle from vertical
        # The refracted ray should have smaller angle (bend toward normal)
        # tan(incident_angle) proportional to pixel offset / focal length
        # For n_air < n_water, the refracted angle should be smaller
        incident_angle_tan = 100.0 / 1000.0  # pixel offset / focal length
        refracted_angle_tan = directions[0, 0] / directions[0, 2]

        # Refracted angle should be smaller (tan smaller)
        assert refracted_angle_tan < incident_angle_tan

    def test_batch_consistency(self, refractive_model_simple, device):
        """Test batch processing of multiple pixels."""
        # Batch of 100 random pixels
        torch.manual_seed(42)
        pixels = torch.rand(100, 2, dtype=torch.float32, device=device) * 1000

        origins, directions = refractive_model_simple.cast_ray(pixels)

        # Check output shapes
        assert origins.shape == (100, 3)
        assert directions.shape == (100, 3)

        # Check dtypes
        assert origins.dtype == torch.float32
        assert directions.dtype == torch.float32

        # Check device
        assert origins.device.type == device
        assert directions.device.type == device

    def test_known_geometry_snells_law(self, simple_camera, device):
        """Test Snell's law with a known geometry."""
        K, R, t = simple_camera
        water_z = 1.0
        normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
        n_air = 1.0
        n_water = 1.333

        model = RefractiveProjectionModel(K, R, t, water_z, normal, n_air, n_water)

        # Create a pixel that produces a known incident angle
        # For a pixel at (500 + dx, 500), the incident ray has angle:
        # tan(theta_air) = dx / focal_length
        # For 10 degrees: tan(10°) ≈ 0.1763
        dx = 1000.0 * math.tan(math.radians(10.0))  # ~176.3
        pixels = torch.tensor([[500.0 + dx, 500.0]], dtype=torch.float32, device=device)

        origins, directions = model.cast_ray(pixels)

        # The refracted angle should satisfy Snell's law:
        # n_air * sin(theta_air) = n_water * sin(theta_water)
        # sin(theta_water) = (n_air / n_water) * sin(theta_air)
        theta_air = math.radians(10.0)
        sin_theta_water = (n_air / n_water) * math.sin(theta_air)
        theta_water = math.asin(sin_theta_water)

        # In our coordinate system, the refracted ray direction is (dx', dy', dz')
        # where tan(theta_water) = sqrt(dx'^2 + dy'^2) / dz'
        # For our case (dy' = 0): tan(theta_water) = dx' / dz'
        expected_tan_water = math.tan(theta_water)
        actual_tan_water = directions[0, 0].item() / directions[0, 2].item()

        # Should match within tolerance
        assert abs(actual_tan_water - expected_tan_water) < 1e-4

    def test_differentiability(self, refractive_model_simple, device):
        """Test that cast_ray is differentiable."""
        # Pixels with gradients enabled
        pixels = torch.tensor(
            [[500.0, 500.0], [600.0, 500.0]],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        origins, directions = refractive_model_simple.cast_ray(pixels)

        # Compute a scalar loss
        loss = origins.sum() + directions.sum()

        # Backpropagate - should not raise an error
        loss.backward()

        # Check that gradients were computed
        assert pixels.grad is not None
        assert pixels.grad.shape == (2, 2)

    def test_protocol_compliance(self, refractive_model_simple):
        """Test that RefractiveProjectionModel implements ProjectionModel protocol."""
        assert isinstance(refractive_model_simple, ProjectionModel)


class TestRefractiveProjectionModelProject:
    """Tests for RefractiveProjectionModel.project method."""

    def test_nadir_point_projects_to_principal_point(
        self, refractive_model_simple, device
    ):
        """Test projection of point directly below camera."""
        # Camera at (0, 0, 0), water at z=1.0, point at (0, 0, 1.5)
        # Should project to principal point (500, 500)
        points = torch.tensor([[0.0, 0.0, 1.5]], dtype=torch.float32, device=device)

        pixels, valid = refractive_model_simple.project(points)

        # Should be valid
        assert valid[0].item() is True

        # Should project near principal point
        expected_pixels = torch.tensor(
            [[500.0, 500.0]], dtype=torch.float32, device=device
        )
        assert torch.allclose(pixels, expected_pixels, atol=1e-4)

    def test_off_axis_point(self, refractive_model_simple, device):
        """Test projection of point offset horizontally from camera."""
        # Point offset in +X direction
        points = torch.tensor([[0.5, 0.0, 1.5]], dtype=torch.float32, device=device)

        pixels, valid = refractive_model_simple.project(points)

        # Should be valid
        assert valid[0].item() is True

        # Should project to pixel offset in +X direction (u > 500)
        assert pixels[0, 0] > 500.0
        # Y should be near principal point
        assert abs(pixels[0, 1] - 500.0) < 1.0

    def test_point_above_water_surface_invalid(self, refractive_model_simple, device):
        """Test that points above water surface are marked invalid."""
        # Point at Z < water_z (water_z = 1.0)
        points = torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float32, device=device)

        pixels, valid = refractive_model_simple.project(points)

        # Should be invalid
        assert valid[0].item() is False

        # Pixel should be NaN
        assert torch.isnan(pixels[0, 0]).item()
        assert torch.isnan(pixels[0, 1]).item()

    def test_roundtrip_consistency_3d_to_pixel_to_3d(
        self, refractive_model_simple, device
    ):
        """Test roundtrip: 3D point -> pixel -> ray -> 3D point."""
        # Start with a 3D point
        original_point = torch.tensor(
            [[0.2, 0.3, 1.8]], dtype=torch.float32, device=device
        )

        # Project to pixel
        pixels, valid = refractive_model_simple.project(original_point)
        assert valid[0].item() is True

        # Cast ray from pixel
        origins, directions = refractive_model_simple.cast_ray(pixels)

        # Compute depth from origin to original point
        # point = origin + depth * direction
        # Solve for depth in the Z component (most stable)
        depth = (original_point[0, 2] - origins[0, 2]) / directions[0, 2]

        # Reconstruct 3D point
        reconstructed_point = origins + depth * directions

        # Should match original point
        assert torch.allclose(reconstructed_point, original_point, atol=1e-4)

    def test_pixel_roundtrip_pixel_to_3d_to_pixel(
        self, refractive_model_simple, device
    ):
        """Test roundtrip: pixel -> ray -> 3D point -> pixel."""
        # Start with pixels
        original_pixels = torch.tensor(
            [[500.0, 500.0], [600.0, 500.0], [500.0, 600.0]],
            dtype=torch.float32,
            device=device,
        )

        # Cast rays
        origins, directions = refractive_model_simple.cast_ray(original_pixels)

        # Create 3D points at arbitrary depth (e.g., 0.8 meters along ray)
        depth = 0.8
        points_3d = origins + depth * directions

        # Project back to pixels
        recovered_pixels, valid = refractive_model_simple.project(points_3d)

        # All should be valid
        assert torch.all(valid).item()

        # Should recover original pixels within tolerance
        assert torch.allclose(recovered_pixels, original_pixels, atol=1e-4)

    def test_batch_consistency(self, refractive_model_simple, device):
        """Test batch processing of multiple points."""
        # Generate 100 random underwater points
        torch.manual_seed(42)
        # Points in a cube: X, Y in [-1, 1], Z in [1.1, 2.0] (below water at z=1.0)
        points = torch.rand(100, 3, dtype=torch.float32, device=device)
        points[:, 0] = points[:, 0] * 2.0 - 1.0  # X: [-1, 1]
        points[:, 1] = points[:, 1] * 2.0 - 1.0  # Y: [-1, 1]
        points[:, 2] = points[:, 2] * 0.9 + 1.1  # Z: [1.1, 2.0]

        pixels, valid = refractive_model_simple.project(points)

        # Check output shapes
        assert pixels.shape == (100, 2)
        assert valid.shape == (100,)

        # Check dtypes
        assert pixels.dtype == torch.float32
        assert valid.dtype == torch.bool

        # Check device
        assert pixels.device.type == device
        assert valid.device.type == device

        # Most points should be valid (all are below water)
        assert torch.sum(valid) > 90

    def test_differentiability(self, refractive_model_simple, device):
        """Test that project is differentiable."""
        # Points with gradients enabled
        points = torch.tensor(
            [[0.0, 0.0, 1.5], [0.2, 0.1, 1.3]],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        pixels, valid = refractive_model_simple.project(points)

        # Compute a scalar loss (only from valid pixels)
        loss = pixels[valid].sum()

        # Backpropagate - should not raise an error
        loss.backward()

        # Check that gradients were computed
        assert points.grad is not None
        assert points.grad.shape == (2, 3)

        # Gradients should be finite
        assert torch.all(torch.isfinite(points.grad)).item()

    def test_with_non_identity_camera(self, device):
        """Test projection with rotated camera and non-zero translation."""
        # Create a camera rotated 30 degrees around Z axis
        angle = math.radians(30.0)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        R = torch.tensor(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )

        # Camera translated to (1.0, 0.5, -0.2) in world frame
        # t = -R @ C, so C = -R^T @ t
        C_world = torch.tensor([1.0, 0.5, -0.2], dtype=torch.float32, device=device)
        t = -R @ C_world

        K = torch.tensor(
            [[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        water_z = 1.0
        normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
        n_air = 1.0
        n_water = 1.333

        model = RefractiveProjectionModel(K, R, t, water_z, normal, n_air, n_water)

        # Project a point
        points = torch.tensor([[1.0, 0.5, 1.5]], dtype=torch.float32, device=device)
        pixels, valid = model.project(points)

        # Should be valid
        assert valid[0].item() is True

        # Pixels should be finite
        assert torch.all(torch.isfinite(pixels)).item()

        # Test roundtrip
        origins, directions = model.cast_ray(pixels)
        depth = (points[0, 2] - origins[0, 2]) / directions[0, 2]
        reconstructed = origins + depth * directions
        assert torch.allclose(reconstructed, points, atol=1e-4)

    def test_output_shapes_and_types(self, refractive_model_simple, device):
        """Test that output shapes and types are correct."""
        points = torch.tensor([[0.0, 0.0, 1.5]], dtype=torch.float32, device=device)

        pixels, valid = refractive_model_simple.project(points)

        # Check types
        assert isinstance(pixels, torch.Tensor)
        assert isinstance(valid, torch.Tensor)

        # Check shapes
        assert pixels.shape == (1, 2)
        assert valid.shape == (1,)

        # Check dtypes
        assert pixels.dtype == torch.float32
        assert valid.dtype == torch.bool

    def test_multiple_invalid_points(self, refractive_model_simple, device):
        """Test handling of multiple invalid points."""
        # Mix of valid and invalid points
        points = torch.tensor(
            [
                [0.0, 0.0, 1.5],  # Valid (below water)
                [0.0, 0.0, 0.5],  # Invalid (above water)
                [0.2, 0.1, 1.3],  # Valid
                [0.0, 0.0, 0.9],  # Invalid (above water)
            ],
            dtype=torch.float32,
            device=device,
        )

        pixels, valid = refractive_model_simple.project(points)

        # Check validity
        assert valid[0].item() is True
        assert valid[1].item() is False
        assert valid[2].item() is True
        assert valid[3].item() is False

        # Invalid pixels should be NaN
        assert torch.all(torch.isfinite(pixels[0])).item()
        assert torch.all(torch.isnan(pixels[1])).item()
        assert torch.all(torch.isfinite(pixels[2])).item()
        assert torch.all(torch.isnan(pixels[3])).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestRefractiveProjectionModelCUDA:
    """Tests for RefractiveProjectionModel on CUDA."""

    def test_cuda_device_cast_ray(self):
        """Test that cast_ray works on CUDA device."""
        device = "cuda"

        # Create camera on CUDA
        R = torch.eye(3, dtype=torch.float32, device=device)
        t = torch.zeros(3, dtype=torch.float32, device=device)
        K = torch.tensor(
            [[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        water_z = 1.0
        normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
        n_air = 1.0
        n_water = 1.333

        model = RefractiveProjectionModel(K, R, t, water_z, normal, n_air, n_water)

        # Cast rays on CUDA
        pixels = torch.tensor([[500.0, 500.0]], dtype=torch.float32, device=device)
        origins, directions = model.cast_ray(pixels)

        # Check device
        assert origins.device.type == "cuda"
        assert directions.device.type == "cuda"

        # Check correctness
        expected_origin = torch.tensor(
            [[0.0, 0.0, 1.0]], dtype=torch.float32, device=device
        )
        expected_direction = torch.tensor(
            [[0.0, 0.0, 1.0]], dtype=torch.float32, device=device
        )
        assert torch.allclose(origins, expected_origin, atol=1e-5)
        assert torch.allclose(directions, expected_direction, atol=1e-6)

    def test_cuda_device_project(self):
        """Test that project works on CUDA device."""
        device = "cuda"

        # Create camera on CUDA
        R = torch.eye(3, dtype=torch.float32, device=device)
        t = torch.zeros(3, dtype=torch.float32, device=device)
        K = torch.tensor(
            [[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        water_z = 1.0
        normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
        n_air = 1.0
        n_water = 1.333

        model = RefractiveProjectionModel(K, R, t, water_z, normal, n_air, n_water)

        # Project points on CUDA
        points = torch.tensor([[0.0, 0.0, 1.5]], dtype=torch.float32, device=device)
        pixels, valid = model.project(points)

        # Check device
        assert pixels.device.type == "cuda"
        assert valid.device.type == "cuda"

        # Check correctness
        assert valid[0].item() is True
        expected_pixels = torch.tensor(
            [[500.0, 500.0]], dtype=torch.float32, device=device
        )
        assert torch.allclose(pixels, expected_pixels, atol=1e-4)
