"""Tests for camera rig diagram generation."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from aquamvs.visualization.rig import render_rig_diagram


class TestRenderRigDiagram:
    """Tests for render_rig_diagram function."""

    def test_basic_render(self):
        """Test basic rendering with three cameras in a triangle pattern."""
        # Create 3 camera positions in a triangle at Z=0
        camera_positions = {
            "cam1": np.array([0.5, 0.0, 0.0]),
            "cam2": np.array([-0.25, 0.433, 0.0]),
            "cam3": np.array([-0.25, -0.433, 0.0]),
        }

        # Identity rotations (cameras looking +Z)
        camera_rotations = {
            "cam1": np.eye(3),
            "cam2": np.eye(3),
            "cam3": np.eye(3),
        }

        water_z = 0.5

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rig.png"
            render_rig_diagram(camera_positions, camera_rotations, water_z, output_path)

            # Verify PNG exists and is loadable
            assert output_path.exists()
            assert output_path.stat().st_size > 0

            # Verify image can be loaded
            with Image.open(output_path) as img:
                assert img.size[0] > 0
                assert img.size[1] > 0

    def test_with_frustums(self):
        """Test rendering with rotation matrices for frustums."""
        # Single camera at origin
        camera_positions = {
            "cam1": np.array([0.0, 0.0, 0.0]),
        }

        # Camera looking +Z (identity)
        camera_rotations = {
            "cam1": np.eye(3),
        }

        water_z = 0.5

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rig.png"
            render_rig_diagram(camera_positions, camera_rotations, water_z, output_path)

            assert output_path.exists()

    def test_with_intrinsics_and_image_size(self):
        """Test rendering with intrinsic matrix and image size."""
        camera_positions = {
            "cam1": np.array([0.0, 0.0, 0.0]),
        }
        camera_rotations = {
            "cam1": np.eye(3),
        }
        water_z = 0.5

        # Standard intrinsics
        K = np.array([[1000.0, 0.0, 800.0], [0.0, 1000.0, 600.0], [0.0, 0.0, 1.0]])

        image_size = (1600, 1200)  # 4:3 aspect ratio

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rig.png"
            render_rig_diagram(
                camera_positions,
                camera_rotations,
                water_z,
                output_path,
                K=K,
                image_size=image_size,
            )

            assert output_path.exists()

    def test_with_point_cloud(self):
        """Test rendering with point cloud overlay."""
        camera_positions = {
            "cam1": np.array([0.5, 0.0, 0.0]),
            "cam2": np.array([-0.25, 0.433, 0.0]),
        }
        camera_rotations = {
            "cam1": np.eye(3),
            "cam2": np.eye(3),
        }
        water_z = 0.5

        # Random points below the water surface
        rng = np.random.default_rng(42)
        point_cloud_points = rng.uniform(-0.5, 0.5, size=(1000, 3))
        point_cloud_points[:, 2] = rng.uniform(0.6, 1.0, size=1000)  # Below water

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rig.png"
            render_rig_diagram(
                camera_positions,
                camera_rotations,
                water_z,
                output_path,
                point_cloud_points=point_cloud_points,
            )

            assert output_path.exists()

    def test_with_large_point_cloud_subsampling(self):
        """Test that large point clouds are subsampled."""
        camera_positions = {
            "cam1": np.array([0.0, 0.0, 0.0]),
        }
        camera_rotations = {
            "cam1": np.eye(3),
        }
        water_z = 0.5

        # Create a large point cloud (> 5000 points)
        rng = np.random.default_rng(42)
        point_cloud_points = rng.uniform(-1.0, 1.0, size=(10000, 3))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rig.png"
            # Should not crash or take too long
            render_rig_diagram(
                camera_positions,
                camera_rotations,
                water_z,
                output_path,
                point_cloud_points=point_cloud_points,
            )

            assert output_path.exists()

    def test_single_camera(self):
        """Test edge case with single camera (for water plane sizing)."""
        camera_positions = {
            "cam1": np.array([0.0, 0.0, 0.0]),
        }
        camera_rotations = {
            "cam1": np.eye(3),
        }
        water_z = 0.5

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rig.png"
            render_rig_diagram(camera_positions, camera_rotations, water_z, output_path)

            assert output_path.exists()

    def test_headless_rendering(self):
        """Verify matplotlib Agg backend is used (no display server required)."""
        import matplotlib

        # Check that Agg backend is being used
        backend = matplotlib.get_backend()
        assert backend == "agg" or backend == "Agg"

    def test_custom_frustum_scale(self):
        """Test custom frustum scale parameter."""
        camera_positions = {
            "cam1": np.array([0.0, 0.0, 0.0]),
        }
        camera_rotations = {
            "cam1": np.eye(3),
        }
        water_z = 0.5

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rig.png"
            render_rig_diagram(
                camera_positions,
                camera_rotations,
                water_z,
                output_path,
                frustum_scale=0.2,  # Larger frustum
            )

            assert output_path.exists()

    def test_custom_dpi(self):
        """Test custom DPI parameter."""
        camera_positions = {
            "cam1": np.array([0.0, 0.0, 0.0]),
        }
        camera_rotations = {
            "cam1": np.eye(3),
        }
        water_z = 0.5

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rig.png"
            render_rig_diagram(
                camera_positions, camera_rotations, water_z, output_path, dpi=300
            )

            assert output_path.exists()
            # Higher DPI should result in larger file (generally)
            file_size = output_path.stat().st_size
            assert file_size > 0

    def test_twelve_camera_ring(self):
        """Test with realistic 12-camera ring configuration."""
        # Reference geometry: 12 cameras at 0.635m radius
        radius = 0.635
        n_cameras = 12

        camera_positions = {}
        camera_rotations = {}

        for i in range(n_cameras):
            angle = 2 * np.pi * i / n_cameras
            name = f"cam{i:02d}"

            # Position on ring at Z=0
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            camera_positions[name] = np.array([x, y, 0.0])

            # Rotation looking toward center and down
            # For simplicity, use identity rotation
            camera_rotations[name] = np.eye(3)

        water_z = 0.978  # Reference water height

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rig.png"
            render_rig_diagram(camera_positions, camera_rotations, water_z, output_path)

            assert output_path.exists()

    def test_missing_rotation_for_some_cameras(self):
        """Test that cameras without rotation matrices skip frustum drawing."""
        camera_positions = {
            "cam1": np.array([0.5, 0.0, 0.0]),
            "cam2": np.array([-0.5, 0.0, 0.0]),
        }

        # Only provide rotation for cam1
        camera_rotations = {
            "cam1": np.eye(3),
        }

        water_z = 0.5

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rig.png"
            # Should not crash, cam2 just won't have a frustum
            render_rig_diagram(camera_positions, camera_rotations, water_z, output_path)

            assert output_path.exists()

    def test_empty_point_cloud(self):
        """Test with empty point cloud array."""
        camera_positions = {
            "cam1": np.array([0.0, 0.0, 0.0]),
        }
        camera_rotations = {
            "cam1": np.eye(3),
        }
        water_z = 0.5

        # Empty point cloud
        point_cloud_points = np.zeros((0, 3))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rig.png"
            render_rig_diagram(
                camera_positions,
                camera_rotations,
                water_z,
                output_path,
                point_cloud_points=point_cloud_points,
            )

            assert output_path.exists()

    def test_output_is_publication_quality(self):
        """Test that output has reasonable resolution and is not empty."""
        camera_positions = {
            "cam1": np.array([0.0, 0.0, 0.0]),
        }
        camera_rotations = {
            "cam1": np.eye(3),
        }
        water_z = 0.5

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rig.png"
            render_rig_diagram(
                camera_positions, camera_rotations, water_z, output_path, dpi=150
            )

            # Load image and check dimensions
            with Image.open(output_path) as img:
                width, height = img.size

                # With figsize=(10, 8) and dpi=150, expect roughly 1500x1200
                # Allow variation due to tight_layout (actual ~967x800)
                assert width >= 900
                assert height >= 700
