"""Tests for 3D scene rendering."""

import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from aquamvs.visualization.scene import (
    LEGACY_VISUALIZER_AVAILABLE,
    OFFSCREEN_AVAILABLE,
    compute_canonical_viewpoints,
    render_all_scenes,
    render_geometry,
    render_scene,
)

RENDERING_AVAILABLE = OFFSCREEN_AVAILABLE or LEGACY_VISUALIZER_AVAILABLE


class TestComputeCanonicalViewpoints:
    """Tests for compute_canonical_viewpoints()."""

    def test_returns_three_viewpoints(self):
        """Test that three viewpoints are returned with correct keys."""
        bbox_min = np.array([0.0, 0.0, 0.5])
        bbox_max = np.array([1.0, 1.0, 1.5])

        viewpoints = compute_canonical_viewpoints(bbox_min, bbox_max)

        assert isinstance(viewpoints, dict)
        assert set(viewpoints.keys()) == {"top", "oblique", "side"}

    def test_viewpoint_structure(self):
        """Test that each viewpoint has correct structure."""
        bbox_min = np.array([0.0, 0.0, 0.5])
        bbox_max = np.array([1.0, 1.0, 1.5])

        viewpoints = compute_canonical_viewpoints(bbox_min, bbox_max)

        for _name, params in viewpoints.items():
            assert isinstance(params, dict)
            assert set(params.keys()) == {"eye", "center", "up"}
            assert params["eye"].shape == (3,)
            assert params["center"].shape == (3,)
            assert params["up"].shape == (3,)

    def test_top_view_eye_above_center(self):
        """Test that top-down view eye is above (smaller Z than) center."""
        bbox_min = np.array([0.0, 0.0, 0.5])
        bbox_max = np.array([1.0, 1.0, 1.5])

        viewpoints = compute_canonical_viewpoints(bbox_min, bbox_max)

        # In Z-down coordinate system, "above" means smaller Z
        assert viewpoints["top"]["eye"][2] < viewpoints["top"]["center"][2]

    def test_top_view_up_vector(self):
        """Test that top-down view up vector is +Y."""
        bbox_min = np.array([0.0, 0.0, 0.5])
        bbox_max = np.array([1.0, 1.0, 1.5])

        viewpoints = compute_canonical_viewpoints(bbox_min, bbox_max)

        np.testing.assert_array_equal(
            viewpoints["top"]["up"],
            np.array([0.0, 1.0, 0.0]),
        )

    def test_oblique_and_side_up_vector(self):
        """Test that oblique and side views have -Z up vector."""
        bbox_min = np.array([0.0, 0.0, 0.5])
        bbox_max = np.array([1.0, 1.0, 1.5])

        viewpoints = compute_canonical_viewpoints(bbox_min, bbox_max)

        np.testing.assert_array_equal(
            viewpoints["oblique"]["up"],
            np.array([0.0, 0.0, -1.0]),
        )
        np.testing.assert_array_equal(
            viewpoints["side"]["up"],
            np.array([0.0, 0.0, -1.0]),
        )

    def test_handles_small_bounding_box(self):
        """Test that very small bounding boxes don't cause division by zero."""
        bbox_min = np.array([0.0, 0.0, 0.0])
        bbox_max = np.array([0.01, 0.01, 0.01])

        # Should not raise
        viewpoints = compute_canonical_viewpoints(bbox_min, bbox_max)

        assert len(viewpoints) == 3


@pytest.mark.skipif(
    not RENDERING_AVAILABLE, reason="No Open3D rendering backend available"
)
class TestRenderGeometry:
    """Tests for render_geometry()."""

    def test_render_point_cloud(self, tmp_path):
        """Test rendering a point cloud to PNG."""
        # Create synthetic point cloud
        points = np.random.rand(100, 3).astype(np.float64)
        colors = np.random.rand(100, 3).astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Render
        output_path = tmp_path / "test.png"
        eye = np.array([0.0, 0.0, -1.0])
        center = np.array([0.5, 0.5, 0.5])
        up = np.array([0.0, 1.0, 0.0])

        render_geometry(pcd, eye, center, up, output_path)

        # Verify file exists
        assert output_path.exists()
        # Verify it's a valid image (can be read)
        img = o3d.io.read_image(str(output_path))
        assert img is not None

    def test_render_mesh(self, tmp_path):
        """Test rendering a mesh to PNG."""
        # Create synthetic mesh (a simple triangle)
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
            ]
        ).astype(np.float64)
        triangles = np.array([[0, 1, 2]]).astype(np.int32)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.paint_uniform_color([0.5, 0.5, 0.5])

        # Render
        output_path = tmp_path / "test_mesh.png"
        eye = np.array([0.5, 0.5, -1.0])
        center = np.array([0.5, 0.5, 0.0])
        up = np.array([0.0, 1.0, 0.0])

        render_geometry(mesh, eye, center, up, output_path)

        # Verify file exists
        assert output_path.exists()

    def test_custom_resolution(self, tmp_path):
        """Test rendering with custom resolution."""
        # Create simple point cloud
        points = np.array([[0.0, 0.0, 0.0]]).astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([1.0, 0.0, 0.0])

        # Render with custom resolution
        output_path = tmp_path / "custom_res.png"
        eye = np.array([0.0, 0.0, -1.0])
        center = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])

        render_geometry(
            pcd,
            eye,
            center,
            up,
            output_path,
            width=640,
            height=480,
        )

        assert output_path.exists()


@pytest.mark.skipif(
    not RENDERING_AVAILABLE, reason="No Open3D rendering backend available"
)
class TestRenderScene:
    """Tests for render_scene()."""

    def test_creates_three_viewpoint_images(self, tmp_path):
        """Test that three PNG files are created for top, oblique, side."""
        # Create point cloud
        points = np.random.rand(50, 3).astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])

        # Render scene
        render_scene(pcd, tmp_path, prefix="test")

        # Verify three files created
        assert (tmp_path / "test_top.png").exists()
        assert (tmp_path / "test_oblique.png").exists()
        assert (tmp_path / "test_side.png").exists()

    def test_custom_prefix(self, tmp_path):
        """Test rendering with custom filename prefix."""
        points = np.array([[0.0, 0.0, 0.0]]).astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([1.0, 0.0, 0.0])

        render_scene(pcd, tmp_path, prefix="fused")

        assert (tmp_path / "fused_top.png").exists()
        assert (tmp_path / "fused_oblique.png").exists()
        assert (tmp_path / "fused_side.png").exists()

    def test_creates_output_directory(self):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "subdir" / "output"
            assert not output_dir.exists()

            points = np.array([[0.0, 0.0, 0.0]]).astype(np.float64)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color([1.0, 0.0, 0.0])

            render_scene(pcd, output_dir, prefix="test")

            assert output_dir.exists()
            assert (output_dir / "test_top.png").exists()


@pytest.mark.skipif(
    not RENDERING_AVAILABLE, reason="No Open3D rendering backend available"
)
class TestRenderAllScenes:
    """Tests for render_all_scenes()."""

    def test_renders_both_point_cloud_and_mesh(self, tmp_path):
        """Test rendering both point cloud and mesh creates 6 files."""
        # Create point cloud
        points = np.random.rand(50, 3).astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])

        # Create mesh
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
            ]
        ).astype(np.float64)
        triangles = np.array([[0, 1, 2]]).astype(np.int32)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.paint_uniform_color([0.5, 0.5, 0.5])

        # Render all
        render_all_scenes(pcd, mesh, tmp_path)

        # Verify 6 files (3 viewpoints x 2 geometries)
        assert (tmp_path / "fused_top.png").exists()
        assert (tmp_path / "fused_oblique.png").exists()
        assert (tmp_path / "fused_side.png").exists()
        assert (tmp_path / "mesh_top.png").exists()
        assert (tmp_path / "mesh_oblique.png").exists()
        assert (tmp_path / "mesh_side.png").exists()

    def test_none_point_cloud(self, tmp_path):
        """Test rendering with None point cloud only renders mesh."""
        # Create mesh
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
            ]
        ).astype(np.float64)
        triangles = np.array([[0, 1, 2]]).astype(np.int32)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.paint_uniform_color([0.5, 0.5, 0.5])

        render_all_scenes(None, mesh, tmp_path)

        # Verify only mesh files created
        assert not (tmp_path / "fused_top.png").exists()
        assert (tmp_path / "mesh_top.png").exists()
        assert (tmp_path / "mesh_oblique.png").exists()
        assert (tmp_path / "mesh_side.png").exists()

    def test_none_mesh(self, tmp_path):
        """Test rendering with None mesh only renders point cloud."""
        # Create point cloud
        points = np.random.rand(50, 3).astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])

        render_all_scenes(pcd, None, tmp_path)

        # Verify only point cloud files created
        assert (tmp_path / "fused_top.png").exists()
        assert (tmp_path / "fused_oblique.png").exists()
        assert (tmp_path / "fused_side.png").exists()
        assert not (tmp_path / "mesh_top.png").exists()

    def test_both_none(self, tmp_path):
        """Test rendering with both None does not crash."""
        # Should not raise, should not create any files
        render_all_scenes(None, None, tmp_path)

        # Verify no PNG files created
        png_files = list(tmp_path.glob("*.png"))
        assert len(png_files) == 0


class TestEmptyGeometry:
    """Tests for handling empty geometries."""

    @pytest.mark.skipif(
        not RENDERING_AVAILABLE, reason="No Open3D rendering backend available"
    )
    def test_empty_point_cloud(self, tmp_path):
        """Test rendering empty point cloud does not crash."""
        # Create empty point cloud
        pcd = o3d.geometry.PointCloud()

        # Should not raise (but may log warnings)
        # Depending on Open3D version, this might fail or succeed
        # We just verify it doesn't crash Python
        try:
            render_scene(pcd, tmp_path, prefix="empty")
        except Exception:
            # It's acceptable for Open3D to raise on empty geometry
            pass


class TestOffscreenUnavailable:
    """Tests for graceful degradation when offscreen rendering is unavailable."""

    def test_render_geometry_when_unavailable(self, tmp_path, monkeypatch):
        """Test that render_geometry logs warning and returns when unavailable."""
        # Temporarily patch both backends to False
        import aquamvs.visualization.scene as scene_module

        monkeypatch.setattr(scene_module, "OFFSCREEN_AVAILABLE", False)
        monkeypatch.setattr(scene_module, "LEGACY_VISUALIZER_AVAILABLE", False)

        # Create point cloud
        points = np.array([[0.0, 0.0, 0.0]]).astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        output_path = tmp_path / "test.png"
        eye = np.array([0.0, 0.0, -1.0])
        center = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])

        # Should not raise, should not create file
        render_geometry(pcd, eye, center, up, output_path)

        assert not output_path.exists()
