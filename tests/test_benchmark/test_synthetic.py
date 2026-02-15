"""Tests for synthetic scene generation."""

import numpy as np
import open3d as o3d
import pytest

from aquamvs.benchmark.synthetic import (
    create_flat_plane_scene,
    create_undulating_scene,
    generate_ground_truth_depth_maps,
    get_reference_geometry,
)


def test_get_reference_geometry():
    """Test reference geometry returns expected constants."""
    geom = get_reference_geometry()

    assert "ring_radius_m" in geom
    assert "water_z_m" in geom
    assert "n_water" in geom
    assert "num_cameras" in geom

    # Check expected values
    assert geom["ring_radius_m"] == pytest.approx(0.635, abs=1e-6)
    assert geom["water_z_m"] == pytest.approx(0.978, abs=1e-3)
    assert geom["n_water"] == pytest.approx(1.333, abs=1e-6)
    assert geom["num_cameras"] == 12


def test_create_flat_plane_scene():
    """Test flat plane scene generation."""
    depth_z = 1.1
    bounds = (-0.2, 0.2, -0.2, 0.2)
    resolution = 0.02  # 2cm for faster test

    mesh, analytic_fn = create_flat_plane_scene(depth_z, bounds, resolution)

    # Check mesh is valid
    assert mesh.has_vertices()
    assert mesh.has_triangles()
    assert len(mesh.vertices) > 0
    assert len(mesh.triangles) > 0

    # Check vertices are at correct Z
    vertices = np.asarray(mesh.vertices)
    assert np.all(vertices[:, 2] == pytest.approx(depth_z, abs=1e-6))

    # Check vertices are within bounds
    assert np.all(vertices[:, 0] >= bounds[0] - resolution)
    assert np.all(vertices[:, 0] <= bounds[1] + resolution)
    assert np.all(vertices[:, 1] >= bounds[2] - resolution)
    assert np.all(vertices[:, 1] <= bounds[3] + resolution)

    # Check analytic function returns constant depth
    x_test = np.array([0.0, 0.1, -0.1])
    y_test = np.array([0.0, 0.05, -0.15])
    z_result = analytic_fn(x_test, y_test)

    assert z_result.shape == x_test.shape
    assert np.all(z_result == pytest.approx(depth_z, abs=1e-6))


def test_create_flat_plane_scene_vertex_count():
    """Test flat plane mesh has expected vertex count."""
    depth_z = 1.0
    bounds = (0.0, 0.1, 0.0, 0.1)  # 10cm x 10cm
    resolution = 0.01  # 1cm spacing

    mesh, _ = create_flat_plane_scene(depth_z, bounds, resolution)

    # With 1cm spacing over 10cm range: 11 vertices per dimension
    # Total: 11 * 11 = 121 vertices
    vertices = np.asarray(mesh.vertices)
    expected_vertices = 11 * 11
    assert len(vertices) == expected_vertices


def test_create_undulating_scene():
    """Test undulating surface generation."""
    base_depth_z = 1.1
    amplitude = 0.005  # 5mm
    wavelength = 0.05  # 5cm
    bounds = (-0.2, 0.2, -0.2, 0.2)
    resolution = 0.02  # 2cm for faster test

    mesh, analytic_fn = create_undulating_scene(
        base_depth_z, amplitude, wavelength, bounds, resolution
    )

    # Check mesh is valid
    assert mesh.has_vertices()
    assert mesh.has_triangles()
    assert len(mesh.vertices) > 0
    assert len(mesh.triangles) > 0

    # Check vertices vary in Z (not flat)
    vertices = np.asarray(mesh.vertices)
    z_values = vertices[:, 2]
    assert z_values.max() > base_depth_z  # Some points above base
    assert z_values.min() < base_depth_z  # Some points below base

    # Check amplitude constraint (max deviation should be ~2*amplitude)
    # Since we have sin + sin, max is 2*amplitude, min is -2*amplitude
    assert z_values.max() <= base_depth_z + 2 * amplitude + 1e-6
    assert z_values.min() >= base_depth_z - 2 * amplitude - 1e-6


def test_undulating_scene_analytic_function():
    """Test undulating surface analytic function matches mesh vertices."""
    base_depth_z = 1.0
    amplitude = 0.01
    wavelength = 0.1
    bounds = (0.0, 0.2, 0.0, 0.2)
    resolution = 0.05

    mesh, analytic_fn = create_undulating_scene(
        base_depth_z, amplitude, wavelength, bounds, resolution
    )

    # Sample mesh vertices
    vertices = np.asarray(mesh.vertices)
    x_mesh = vertices[:, 0]
    y_mesh = vertices[:, 1]
    z_mesh = vertices[:, 2]

    # Evaluate analytic function at mesh vertex positions
    z_analytic = analytic_fn(x_mesh, y_mesh)

    # Should match exactly (mesh was generated from this function)
    assert np.allclose(z_mesh, z_analytic, atol=1e-10)


def test_undulating_scene_symmetry():
    """Test undulating surface has expected periodic behavior."""
    base_depth_z = 1.0
    amplitude = 0.005
    wavelength = 0.1
    bounds = (0.0, 0.3, 0.0, 0.3)

    _, analytic_fn = create_undulating_scene(
        base_depth_z, amplitude, wavelength, bounds
    )

    # Test periodicity in X direction
    x1 = 0.0
    x2 = wavelength  # One wavelength away
    y = 0.1

    z1 = analytic_fn(np.array([x1]), np.array([y]))
    z2 = analytic_fn(np.array([x2]), np.array([y]))

    # Should be approximately equal (sin(0) ≈ sin(2π))
    assert z1 == pytest.approx(z2, abs=1e-6)


class MockProjectionModel:
    """Mock projection model for testing depth map generation."""

    def __init__(self, camera_position: np.ndarray, target_z: float):
        """Initialize mock projection model.

        Args:
            camera_position: 3D position of camera.
            target_z: Z coordinate camera is looking at.
        """
        self.camera_position = camera_position
        self.target_z = target_z

    def cast_ray(self, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cast rays from camera through pixels.

        For testing, we create rays pointing toward target_z plane.

        Args:
            pixels: N × 2 array of pixel coordinates (u, v).

        Returns:
            Tuple of (origins, directions) as N × 3 arrays.
        """
        n = len(pixels)

        # All rays start at camera position
        origins = np.tile(self.camera_position, (n, 1))

        # For simplicity, rays point straight down toward target_z
        # In real model this would be more complex
        directions = np.zeros((n, 3))
        directions[:, 2] = 1.0  # Point in +Z direction

        return origins, directions


def test_generate_ground_truth_depth_maps():
    """Test ground truth depth map generation via ray casting."""
    # Create a simple flat plane
    depth_z = 1.0
    bounds = (-0.1, 0.1, -0.1, 0.1)
    mesh, _ = create_flat_plane_scene(depth_z, bounds, resolution=0.01)

    # Create mock projection model (camera above plane looking down)
    camera_pos = np.array([0.0, 0.0, 0.5])  # Above the plane
    model = MockProjectionModel(camera_pos, depth_z)

    # Generate depth map
    image_shape = (20, 20)  # Small for test
    depth_maps = generate_ground_truth_depth_maps(
        mesh, {"test_cam": model}, image_shape
    )

    # Check result
    assert "test_cam" in depth_maps
    depth_map = depth_maps["test_cam"]

    # Check shape
    assert depth_map.shape == image_shape

    # Check that we got some valid depths
    # (not all NaN - some rays should hit the mesh)
    valid_pixels = ~np.isnan(depth_map)
    assert np.sum(valid_pixels) > 0


def test_generate_ground_truth_depth_maps_shape():
    """Test depth maps have correct shape."""
    # Create mesh
    mesh, _ = create_flat_plane_scene(1.0, (-0.1, 0.1, -0.1, 0.1), resolution=0.02)

    # Mock projection models for multiple cameras
    models = {
        "cam1": MockProjectionModel(np.array([0.0, 0.0, 0.5]), 1.0),
        "cam2": MockProjectionModel(np.array([0.1, 0.0, 0.5]), 1.0),
    }

    image_shape = (32, 48)
    depth_maps = generate_ground_truth_depth_maps(mesh, models, image_shape)

    # Check all cameras have correct shape
    assert len(depth_maps) == 2
    for cam_name, depth_map in depth_maps.items():
        assert depth_map.shape == image_shape
        assert depth_map.dtype == np.float64


def test_generate_ground_truth_depth_maps_invalid_pixels():
    """Test that rays missing the mesh produce NaN depths."""
    # Create small mesh
    mesh, _ = create_flat_plane_scene(1.0, (-0.05, 0.05, -0.05, 0.05), resolution=0.02)

    # Camera looking at mesh from above
    model = MockProjectionModel(np.array([0.0, 0.0, 0.5]), 1.0)

    # Large image - many rays will miss small mesh
    image_shape = (100, 100)
    depth_maps = generate_ground_truth_depth_maps(mesh, {"cam": model}, image_shape)

    depth_map = depth_maps["cam"]

    # Should have a mix of valid and invalid (NaN) pixels
    valid = ~np.isnan(depth_map)
    invalid = np.isnan(depth_map)

    assert np.sum(valid) > 0  # Some hits
    assert np.sum(invalid) > 0  # Some misses
