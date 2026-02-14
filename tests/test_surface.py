"""Tests for surface reconstruction."""

import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from aquamvs.config import SurfaceConfig
from aquamvs.surface import (
    load_mesh,
    reconstruct_bpa,
    reconstruct_heightfield,
    reconstruct_poisson,
    reconstruct_surface,
    save_mesh,
)


def create_flat_plane_cloud(z=1.5, n_points=100):
    """Create a synthetic point cloud of a flat horizontal plane."""
    # Random XY points over a small region
    xy = np.random.uniform(-0.1, 0.1, size=(n_points, 2))
    z_vals = np.full(n_points, z)
    points = np.column_stack([xy, z_vals])

    # Upward-pointing normals (-Z in our coordinate system)
    normals = np.tile([0.0, 0.0, -1.0], (n_points, 1))

    # Random colors
    colors = np.random.uniform(0, 1, size=(n_points, 3))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def create_sine_surface_cloud(n_points=200):
    """Create a point cloud on Z = 1.5 + 0.01 * sin(X)."""
    xy = np.random.uniform(-0.1, 0.1, size=(n_points, 2))
    x_vals = xy[:, 0]
    z_vals = 1.5 + 0.01 * np.sin(x_vals * 10)  # Sine wave in X
    points = np.column_stack([xy, z_vals])

    colors = np.random.uniform(0, 1, size=(n_points, 3))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


class TestPoissonReconstruction:
    """Tests for Poisson surface reconstruction."""

    def test_basic_reconstruction(self):
        """Test that Poisson reconstruction produces a valid mesh."""
        pcd = create_flat_plane_cloud(z=1.5, n_points=100)
        config = SurfaceConfig(method="poisson", poisson_depth=6)

        mesh = reconstruct_poisson(pcd, config)

        # Verify mesh has vertices and triangles
        assert len(mesh.vertices) > 0
        assert len(mesh.triangles) > 0

        # Verify mesh has colors
        assert len(mesh.vertex_colors) == len(mesh.vertices)

        # Verify vertices are near Z = 1.5
        vertices = np.asarray(mesh.vertices)
        z_vals = vertices[:, 2]
        assert np.abs(z_vals.mean() - 1.5) < 0.1

    def test_requires_normals(self):
        """Test that Poisson raises ValueError without normals."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
        # No normals set

        config = SurfaceConfig(method="poisson")

        with pytest.raises(ValueError, match="normals"):
            reconstruct_poisson(pcd, config)

    def test_density_trimming(self):
        """Test that density trimming produces a reasonable mesh size."""
        # Create a cloud with a dense region and a few outliers
        dense_points = np.random.uniform(-0.05, 0.05, size=(200, 2))
        dense_z = np.full(200, 1.5)
        dense = np.column_stack([dense_points, dense_z])

        # Add a few outliers far away
        outlier_points = np.array([[0.5, 0.5, 1.5], [0.6, 0.6, 1.5]])
        points = np.vstack([dense, outlier_points])

        normals = np.tile([0.0, 0.0, -1.0], (len(points), 1))
        colors = np.random.uniform(0, 1, size=(len(points), 3))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        config = SurfaceConfig(method="poisson", poisson_depth=6)
        mesh = reconstruct_poisson(pcd, config)

        # Verify mesh exists and is non-trivial
        assert len(mesh.vertices) > 0
        assert len(mesh.triangles) > 0

        # Verify density trimming has removed the bottom 1% of vertices
        # The mesh should not be empty, but also not huge (Poisson creates watertight surfaces)
        # Just verify it produces something reasonable
        assert len(mesh.vertices) < 10000  # Should not be excessively large


class TestHeightfieldReconstruction:
    """Tests for height-field surface reconstruction."""

    def test_known_surface(self):
        """Test reconstruction of a known sine surface."""
        pcd = create_sine_surface_cloud(n_points=300)
        config = SurfaceConfig(method="heightfield", grid_resolution=0.01)

        mesh = reconstruct_heightfield(pcd, config)

        # Verify mesh has vertices and triangles
        assert len(mesh.vertices) > 0
        assert len(mesh.triangles) > 0

        # Sample the mesh at a few known XY locations and check Z
        vertices = np.asarray(mesh.vertices)

        # Check that some vertices match the sine function
        # Pick vertices near X=0
        near_zero = np.abs(vertices[:, 0]) < 0.01
        if near_zero.any():
            z_near_zero = vertices[near_zero, 2]
            # Z should be near 1.5 + 0.01*sin(0) = 1.5
            assert np.abs(z_near_zero.mean() - 1.5) < 0.01

    def test_color_transfer(self):
        """Test that colors are interpolated correctly."""
        # Create a cloud with known colors: red on left, blue on right
        n = 100
        x_vals = np.linspace(-0.1, 0.1, n)
        y_vals = np.random.uniform(-0.05, 0.05, n)
        z_vals = np.full(n, 1.5)
        points = np.column_stack([x_vals, y_vals, z_vals])

        # Gradient from red (left) to blue (right)
        t = (x_vals + 0.1) / 0.2  # Normalize to [0, 1]
        colors = np.column_stack([1 - t, np.zeros(n), t])  # Red to blue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        config = SurfaceConfig(method="heightfield", grid_resolution=0.01)
        mesh = reconstruct_heightfield(pcd, config)

        # Verify mesh has colors
        assert len(mesh.vertex_colors) == len(mesh.vertices)

        # Check left side is reddish, right side is bluish
        vertices = np.asarray(mesh.vertices)
        vertex_colors = np.asarray(mesh.vertex_colors)

        left_mask = vertices[:, 0] < -0.05
        right_mask = vertices[:, 0] > 0.05

        if left_mask.any():
            left_colors = vertex_colors[left_mask]
            # Red channel should dominate on the left
            assert left_colors[:, 0].mean() > left_colors[:, 2].mean()

        if right_mask.any():
            right_colors = vertex_colors[right_mask]
            # Blue channel should dominate on the right
            assert right_colors[:, 2].mean() > right_colors[:, 0].mean()

    def test_too_few_points(self):
        """Test that too few points raises ValueError."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.random.rand(2, 3))

        config = SurfaceConfig(method="heightfield")

        with pytest.raises(ValueError, match="at least 4 points"):
            reconstruct_heightfield(pcd, config)

    def test_nan_fill_creates_mesh(self):
        """Test that interpolation works correctly with gaps."""
        # Create two separate clusters of points
        # Note: scipy.griddata with linear method uses Delaunay triangulation
        # and WILL interpolate across gaps - this is expected behavior
        cluster1 = np.random.uniform(-0.1, -0.05, size=(50, 2))
        cluster1_z = np.full(50, 1.5)
        cluster1_points = np.column_stack([cluster1, cluster1_z])

        cluster2 = np.random.uniform(0.05, 0.1, size=(50, 2))
        cluster2_z = np.full(50, 1.5)
        cluster2_points = np.column_stack([cluster2, cluster2_z])

        points = np.vstack([cluster1_points, cluster2_points])
        colors = np.random.uniform(0, 1, size=(len(points), 3))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Use a fine grid
        config = SurfaceConfig(method="heightfield", grid_resolution=0.005)
        mesh = reconstruct_heightfield(pcd, config)

        # Mesh should have vertices from both clusters
        assert len(mesh.vertices) > 0
        assert len(mesh.triangles) > 0

        # Verify mesh has reasonable bounds
        vertices = np.asarray(mesh.vertices)
        assert vertices[:, 0].min() < -0.05  # Left cluster
        assert vertices[:, 0].max() > 0.05   # Right cluster

    def test_empty_point_cloud(self):
        """Test that empty point cloud returns empty mesh."""
        pcd = o3d.geometry.PointCloud()

        config = SurfaceConfig(method="heightfield")

        # Should raise ValueError for < 4 points
        with pytest.raises(ValueError):
            reconstruct_heightfield(pcd, config)


class TestBPAReconstruction:
    """Tests for Ball Pivoting Algorithm surface reconstruction."""

    def test_basic_reconstruction(self):
        """Test that BPA reconstruction produces a valid mesh."""
        pcd = create_flat_plane_cloud(z=1.5, n_points=100)
        config = SurfaceConfig(method="bpa", bpa_radii=[0.01, 0.02, 0.04])

        mesh = reconstruct_bpa(pcd, config)

        # Verify mesh has vertices and triangles
        assert len(mesh.vertices) > 0
        assert len(mesh.triangles) > 0

        # Verify mesh has colors
        assert len(mesh.vertex_colors) == len(mesh.vertices)

        # Verify vertices are near Z = 1.5
        vertices = np.asarray(mesh.vertices)
        z_vals = vertices[:, 2]
        assert np.abs(z_vals.mean() - 1.5) < 0.1

    def test_requires_normals(self):
        """Test that BPA raises ValueError without normals."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
        # No normals set

        config = SurfaceConfig(method="bpa")

        with pytest.raises(ValueError, match="normals"):
            reconstruct_bpa(pcd, config)

    def test_auto_radii(self):
        """Test that BPA auto-estimates radii when bpa_radii=None."""
        pcd = create_flat_plane_cloud(z=1.5, n_points=100)
        config = SurfaceConfig(method="bpa", bpa_radii=None)

        mesh = reconstruct_bpa(pcd, config)

        # Verify mesh has vertices and triangles
        assert len(mesh.vertices) > 0
        assert len(mesh.triangles) > 0

        # Verify mesh has colors
        assert len(mesh.vertex_colors) == len(mesh.vertices)

    def test_explicit_radii(self):
        """Test BPA with explicit radii."""
        pcd = create_flat_plane_cloud(z=1.5, n_points=100)
        config = SurfaceConfig(method="bpa", bpa_radii=[0.01, 0.02, 0.04])

        mesh = reconstruct_bpa(pcd, config)

        # Verify mesh has vertices and triangles
        assert len(mesh.vertices) > 0
        assert len(mesh.triangles) > 0

    def test_dispatch_bpa(self):
        """Test that reconstruct_surface dispatches correctly for method='bpa'."""
        pcd = create_flat_plane_cloud(z=1.5, n_points=100)
        config = SurfaceConfig(method="bpa", bpa_radii=[0.01, 0.02, 0.04])

        mesh = reconstruct_surface(pcd, config)

        # Verify we got a mesh
        assert len(mesh.vertices) > 0
        assert len(mesh.triangles) > 0



class TestReconstructSurface:
    """Tests for the dispatch function."""

    def test_dispatch_poisson(self):
        """Test that 'poisson' dispatches correctly."""
        pcd = create_flat_plane_cloud(n_points=100)
        config = SurfaceConfig(method="poisson", poisson_depth=6)

        mesh = reconstruct_surface(pcd, config)

        # Verify we got a mesh
        assert len(mesh.vertices) > 0
        assert len(mesh.triangles) > 0

    def test_dispatch_heightfield(self):
        """Test that 'heightfield' dispatches correctly."""
        pcd = create_flat_plane_cloud(n_points=100)
        config = SurfaceConfig(method="heightfield", grid_resolution=0.01)

        mesh = reconstruct_surface(pcd, config)

        # Verify we got a mesh
        assert len(mesh.vertices) > 0
        assert len(mesh.triangles) > 0

    def test_unknown_method_raises(self):
        """Test that unknown method raises ValueError."""
        pcd = create_flat_plane_cloud(n_points=100)
        config = SurfaceConfig(method="unknown_method")

        with pytest.raises(ValueError, match="Unknown surface method"):
            reconstruct_surface(pcd, config)


class TestMeshIO:
    """Tests for save_mesh and load_mesh."""

    def test_save_load_roundtrip(self):
        """Test that saving and loading preserves mesh data."""
        pcd = create_flat_plane_cloud(n_points=100)
        config = SurfaceConfig(method="heightfield", grid_resolution=0.01)

        mesh_original = reconstruct_heightfield(pcd, config)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            temp_path = Path(f.name)

        try:
            save_mesh(mesh_original, temp_path)

            # Load back
            mesh_loaded = load_mesh(temp_path)

            # Verify vertex count matches
            assert len(mesh_loaded.vertices) == len(mesh_original.vertices)

            # Verify triangle count matches
            assert len(mesh_loaded.triangles) == len(mesh_original.triangles)

            # Verify vertex positions match
            vertices_orig = np.asarray(mesh_original.vertices)
            vertices_loaded = np.asarray(mesh_loaded.vertices)
            np.testing.assert_allclose(vertices_orig, vertices_loaded, atol=1e-5)

        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()

    def test_normals_preserved(self):
        """Test that vertex normals are preserved in save/load."""
        pcd = create_flat_plane_cloud(n_points=100)
        config = SurfaceConfig(method="heightfield", grid_resolution=0.01)

        mesh_original = reconstruct_heightfield(pcd, config)

        # Verify original has normals
        assert mesh_original.has_vertex_normals()

        # Save and load
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            temp_path = Path(f.name)

        try:
            save_mesh(mesh_original, temp_path)
            mesh_loaded = load_mesh(temp_path)

            # Verify loaded mesh has normals
            assert mesh_loaded.has_vertex_normals()

        finally:
            if temp_path.exists():
                temp_path.unlink()
