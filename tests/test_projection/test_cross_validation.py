"""Cross-validation tests comparing PyTorch RefractiveProjectionModel against AquaCal NumPy implementation."""

import math

import numpy as np
import pytest
import torch
from aquacal.config.schema import CameraExtrinsics, CameraIntrinsics
from aquacal.core.camera import Camera
from aquacal.core.interface_model import Interface
from aquacal.core.refractive_geometry import refractive_project, trace_ray_air_to_water

from aquamvs.projection import RefractiveProjectionModel


@pytest.fixture
def reference_camera_params():
    """Realistic camera parameters from DESIGN.md reference geometry."""
    # Camera on the ring: radius 0.635m, positioned at angle 0 (positive X)
    # In Z-down world frame, camera is near Z=0, looking down
    # Ring at radius 0.635m: camera at (0.635, 0, 0)
    #
    # Camera looking approximately straight down: R is approximately identity
    # but with a slight tilt toward center. For simplicity, start with
    # a camera looking straight down from (0.635, 0, 0).
    #
    # water_z ~ 0.978m (camera-to-water air gap from DESIGN.md)
    # n_air = 1.0, n_water = 1.333
    # Image resolution: 1600 x 1200
    # Focal length: ~1400 pixels (typical for this setup)
    return {
        "fx": 1400.0,
        "fy": 1400.0,
        "cx": 800.0,
        "cy": 600.0,
        "image_size": (1600, 1200),
        "R": np.eye(3, dtype=np.float64),  # looking straight down
        "t": np.array([-0.635, 0.0, 0.0], dtype=np.float64),  # t = -R @ C
        "water_z": 0.978,
        "normal": np.array([0.0, 0.0, -1.0], dtype=np.float64),
        "n_air": 1.0,
        "n_water": 1.333,
        "dist_coeffs": np.zeros(
            5, dtype=np.float64
        ),  # zero distortion (post-undistortion)
    }


@pytest.fixture
def rotated_camera_params():
    """Camera rotated around Z by 30 degrees (simulating off-axis ring camera)."""
    angle = math.radians(30.0)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Rotation around Z axis
    R = np.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )

    # Camera center at (0.635, 0, 0) rotated by 30 degrees
    # After rotation, camera is at approximately (0.55, 0.318, 0)
    C_world = np.array([0.635 * cos_a, 0.635 * sin_a, 0.0], dtype=np.float64)
    t = -R @ C_world

    return {
        "fx": 1400.0,
        "fy": 1400.0,
        "cx": 800.0,
        "cy": 600.0,
        "image_size": (1600, 1200),
        "R": R,
        "t": t,
        "water_z": 0.978,
        "normal": np.array([0.0, 0.0, -1.0], dtype=np.float64),
        "n_air": 1.0,
        "n_water": 1.333,
        "dist_coeffs": np.zeros(5, dtype=np.float64),
    }


@pytest.fixture
def aquacal_camera_and_interface(reference_camera_params):
    """Construct AquaCal Camera and Interface from reference params."""
    p = reference_camera_params
    K = np.array(
        [[p["fx"], 0, p["cx"]], [0, p["fy"], p["cy"]], [0, 0, 1]], dtype=np.float64
    )

    intrinsics = CameraIntrinsics(
        K=K, dist_coeffs=p["dist_coeffs"], image_size=p["image_size"]
    )
    extrinsics = CameraExtrinsics(R=p["R"], t=p["t"])
    camera = Camera("test_cam", intrinsics, extrinsics)

    interface = Interface(
        normal=p["normal"],
        camera_distances={"test_cam": p["water_z"]},
        n_air=p["n_air"],
        n_water=p["n_water"],
    )
    return camera, interface


@pytest.fixture
def aquacal_rotated_camera_and_interface(rotated_camera_params):
    """Construct AquaCal Camera and Interface from rotated camera params."""
    p = rotated_camera_params
    K = np.array(
        [[p["fx"], 0, p["cx"]], [0, p["fy"], p["cy"]], [0, 0, 1]], dtype=np.float64
    )

    intrinsics = CameraIntrinsics(
        K=K, dist_coeffs=p["dist_coeffs"], image_size=p["image_size"]
    )
    extrinsics = CameraExtrinsics(R=p["R"], t=p["t"])
    camera = Camera("test_cam", intrinsics, extrinsics)

    interface = Interface(
        normal=p["normal"],
        camera_distances={"test_cam": p["water_z"]},
        n_air=p["n_air"],
        n_water=p["n_water"],
    )
    return camera, interface


@pytest.fixture
def aquamvs_model(reference_camera_params, device):
    """Construct PyTorch RefractiveProjectionModel from reference params."""
    p = reference_camera_params
    K = torch.tensor(
        [[p["fx"], 0, p["cx"]], [0, p["fy"], p["cy"]], [0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )
    R = torch.tensor(p["R"], dtype=torch.float32, device=device)
    t = torch.tensor(p["t"], dtype=torch.float32, device=device)
    normal = torch.tensor(p["normal"], dtype=torch.float32, device=device)
    return RefractiveProjectionModel(
        K, R, t, p["water_z"], normal, p["n_air"], p["n_water"]
    )


@pytest.fixture
def aquamvs_rotated_model(rotated_camera_params, device):
    """Construct PyTorch RefractiveProjectionModel from rotated camera params."""
    p = rotated_camera_params
    K = torch.tensor(
        [[p["fx"], 0, p["cx"]], [0, p["fy"], p["cy"]], [0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )
    R = torch.tensor(p["R"], dtype=torch.float32, device=device)
    t = torch.tensor(p["t"], dtype=torch.float32, device=device)
    normal = torch.tensor(p["normal"], dtype=torch.float32, device=device)
    return RefractiveProjectionModel(
        K, R, t, p["water_z"], normal, p["n_air"], p["n_water"]
    )


class TestCastRayCrossValidation:
    """Cross-validation tests for cast_ray against AquaCal's trace_ray_air_to_water."""

    def test_cast_ray_grid(self, aquacal_camera_and_interface, aquamvs_model, device):
        """Test cast_ray on a grid of pixels across the image."""
        camera, interface = aquacal_camera_and_interface

        # Generate 10x10 grid of pixels from (100, 100) to (1500, 1100)
        u_grid = np.linspace(100, 1500, 10)
        v_grid = np.linspace(100, 1100, 10)
        pixels_list = []
        for u in u_grid:
            for v in v_grid:
                pixels_list.append([u, v])

        pixels_np = np.array(pixels_list, dtype=np.float64)
        len(pixels_list)

        # Call AquaCal for each pixel (single-pixel API)
        origins_aquacal = []
        directions_aquacal = []
        valid_indices = []

        for i, pixel_np in enumerate(pixels_np):
            result = trace_ray_air_to_water(camera, interface, pixel_np)
            if result[0] is not None:
                origins_aquacal.append(result[0])
                directions_aquacal.append(result[1])
                valid_indices.append(i)

        # If no valid pixels, skip test
        if len(valid_indices) == 0:
            pytest.skip("No valid pixels in grid")

        # Call AquaMVS (batched)
        pixels_pt = torch.tensor(pixels_np, dtype=torch.float32, device=device)
        origins_pt, directions_pt = aquamvs_model.cast_ray(pixels_pt)

        # Compare valid pixels
        for idx in valid_indices:
            origin_aquacal = torch.tensor(
                origins_aquacal[valid_indices.index(idx)],
                dtype=torch.float32,
                device=device,
            )
            direction_aquacal = torch.tensor(
                directions_aquacal[valid_indices.index(idx)],
                dtype=torch.float32,
                device=device,
            )

            assert torch.allclose(origins_pt[idx], origin_aquacal, atol=1e-5), (
                f"Origin mismatch at pixel {pixels_np[idx]}: PyTorch={origins_pt[idx].cpu().numpy()}, AquaCal={origin_aquacal.cpu().numpy()}"
            )
            assert torch.allclose(directions_pt[idx], direction_aquacal, atol=1e-5), (
                f"Direction mismatch at pixel {pixels_np[idx]}: PyTorch={directions_pt[idx].cpu().numpy()}, AquaCal={direction_aquacal.cpu().numpy()}"
            )

    def test_cast_ray_principal_point(
        self, aquacal_camera_and_interface, aquamvs_model, device
    ):
        """Test cast_ray at the principal point (zero refraction case)."""
        camera, interface = aquacal_camera_and_interface

        # Principal point at (800, 600)
        pixel_np = np.array([800.0, 600.0], dtype=np.float64)
        result_np = trace_ray_air_to_water(camera, interface, pixel_np)

        if result_np[0] is None:
            pytest.skip("AquaCal returned None for principal point")

        pixel_pt = torch.tensor([[800.0, 600.0]], dtype=torch.float32, device=device)
        origins_pt, directions_pt = aquamvs_model.cast_ray(pixel_pt)

        origin_aquacal = torch.tensor(result_np[0], dtype=torch.float32, device=device)
        direction_aquacal = torch.tensor(
            result_np[1], dtype=torch.float32, device=device
        )

        assert torch.allclose(origins_pt[0], origin_aquacal, atol=1e-5)
        assert torch.allclose(directions_pt[0], direction_aquacal, atol=1e-5)

    def test_cast_ray_corners(
        self, aquacal_camera_and_interface, aquamvs_model, device
    ):
        """Test cast_ray at the four image corners (highest incidence angles)."""
        camera, interface = aquacal_camera_and_interface

        # Four corners: (0, 0), (1600, 0), (0, 1200), (1600, 1200)
        corners = np.array(
            [[0.0, 0.0], [1600.0, 0.0], [0.0, 1200.0], [1600.0, 1200.0]],
            dtype=np.float64,
        )

        for corner in corners:
            result_np = trace_ray_air_to_water(camera, interface, corner)

            if result_np[0] is None:
                continue  # Skip invalid corners

            pixel_pt = torch.tensor([corner], dtype=torch.float32, device=device)
            origins_pt, directions_pt = aquamvs_model.cast_ray(pixel_pt)

            origin_aquacal = torch.tensor(
                result_np[0], dtype=torch.float32, device=device
            )
            direction_aquacal = torch.tensor(
                result_np[1], dtype=torch.float32, device=device
            )

            assert torch.allclose(origins_pt[0], origin_aquacal, atol=1e-5), (
                f"Origin mismatch at corner {corner}"
            )
            assert torch.allclose(directions_pt[0], direction_aquacal, atol=1e-5), (
                f"Direction mismatch at corner {corner}"
            )


class TestProjectCrossValidation:
    """Cross-validation tests for project against AquaCal's refractive_project."""

    def test_project_grid(
        self,
        aquacal_camera_and_interface,
        aquamvs_model,
        device,
        reference_camera_params,
    ):
        """Test project on a grid of 3D points underwater."""
        camera, interface = aquacal_camera_and_interface
        p = reference_camera_params

        # Generate grid of 3D points
        # XY grid in [-0.5, 0.5], centered around camera X position (0.635)
        # Depths Z in [water_z + 0.2, water_z + 0.7]
        x_grid = np.linspace(0.135, 1.135, 5)  # 0.635 +/- 0.5
        y_grid = np.linspace(-0.5, 0.5, 5)
        z_grid = np.linspace(p["water_z"] + 0.2, p["water_z"] + 0.7, 4)

        points_list = []
        for x in x_grid:
            for y in y_grid:
                for z in z_grid:
                    points_list.append([x, y, z])

        points_np = np.array(points_list, dtype=np.float64)
        len(points_list)

        # Call AquaCal for each point
        pixels_aquacal = []
        valid_indices = []

        for i, point_np in enumerate(points_np):
            pixel_np = refractive_project(camera, interface, point_np)
            if pixel_np is not None:
                pixels_aquacal.append(pixel_np)
                valid_indices.append(i)

        if len(valid_indices) == 0:
            pytest.skip("No valid points in grid")

        # Call AquaMVS (batched)
        points_pt = torch.tensor(points_np, dtype=torch.float32, device=device)
        pixels_pt, valid_pt = aquamvs_model.project(points_pt)

        # Compare valid points
        for idx in valid_indices:
            pixel_aquacal = torch.tensor(
                pixels_aquacal[valid_indices.index(idx)],
                dtype=torch.float32,
                device=device,
            )

            assert valid_pt[idx].item(), (
                f"PyTorch marked point {points_np[idx]} as invalid, but AquaCal succeeded"
            )
            assert torch.allclose(pixels_pt[idx], pixel_aquacal, atol=1e-5), (
                f"Pixel mismatch for point {points_np[idx]}: PyTorch={pixels_pt[idx].cpu().numpy()}, AquaCal={pixel_aquacal.cpu().numpy()}"
            )

    def test_project_point_below_camera(
        self,
        aquacal_camera_and_interface,
        aquamvs_model,
        device,
        reference_camera_params,
    ):
        """Test projection of point directly below the camera center (nadir)."""
        camera, interface = aquacal_camera_and_interface
        p = reference_camera_params

        # Camera center is at (0.635, 0, 0)
        # Point directly below at (0.635, 0, water_z + 0.5)
        point_np = np.array([0.635, 0.0, p["water_z"] + 0.5], dtype=np.float64)

        pixel_np = refractive_project(camera, interface, point_np)
        if pixel_np is None:
            pytest.skip("AquaCal returned None for nadir point")

        point_pt = torch.tensor([point_np], dtype=torch.float32, device=device)
        pixels_pt, valid_pt = aquamvs_model.project(point_pt)

        assert valid_pt[0].item()

        pixel_aquacal = torch.tensor(pixel_np, dtype=torch.float32, device=device)
        assert torch.allclose(pixels_pt[0], pixel_aquacal, atol=1e-5)

    def test_project_varying_depths(
        self,
        aquacal_camera_and_interface,
        aquamvs_model,
        device,
        reference_camera_params,
    ):
        """Test projection of points at same XY position but varying depths."""
        camera, interface = aquacal_camera_and_interface
        p = reference_camera_params

        # Fixed XY position offset from camera center
        x = 0.635 + 0.2
        y = 0.1

        # Varying depths
        depths = np.linspace(0.2, 0.7, 10)
        points_list = []
        for depth in depths:
            points_list.append([x, y, p["water_z"] + depth])

        points_np = np.array(points_list, dtype=np.float64)

        # Call AquaCal
        pixels_aquacal = []
        valid_indices = []
        for i, point_np in enumerate(points_np):
            pixel_np = refractive_project(camera, interface, point_np)
            if pixel_np is not None:
                pixels_aquacal.append(pixel_np)
                valid_indices.append(i)

        if len(valid_indices) == 0:
            pytest.skip("No valid points at varying depths")

        # Call AquaMVS
        points_pt = torch.tensor(points_np, dtype=torch.float32, device=device)
        pixels_pt, valid_pt = aquamvs_model.project(points_pt)

        # Compare
        for idx in valid_indices:
            pixel_aquacal = torch.tensor(
                pixels_aquacal[valid_indices.index(idx)],
                dtype=torch.float32,
                device=device,
            )
            assert valid_pt[idx].item()
            assert torch.allclose(pixels_pt[idx], pixel_aquacal, atol=1e-5)


class TestRotatedCameraCrossValidation:
    """Cross-validation tests with a rotated camera (non-trivial rotation)."""

    def test_rotated_camera_cast_ray(
        self, aquacal_rotated_camera_and_interface, aquamvs_rotated_model, device
    ):
        """Test cast_ray on a pixel grid with rotated camera."""
        camera, interface = aquacal_rotated_camera_and_interface

        # Generate 8x8 grid
        u_grid = np.linspace(200, 1400, 8)
        v_grid = np.linspace(200, 1000, 8)
        pixels_list = []
        for u in u_grid:
            for v in v_grid:
                pixels_list.append([u, v])

        pixels_np = np.array(pixels_list, dtype=np.float64)

        # Call AquaCal
        origins_aquacal = []
        directions_aquacal = []
        valid_indices = []

        for i, pixel_np in enumerate(pixels_np):
            result = trace_ray_air_to_water(camera, interface, pixel_np)
            if result[0] is not None:
                origins_aquacal.append(result[0])
                directions_aquacal.append(result[1])
                valid_indices.append(i)

        if len(valid_indices) == 0:
            pytest.skip("No valid pixels in rotated camera grid")

        # Call AquaMVS
        pixels_pt = torch.tensor(pixels_np, dtype=torch.float32, device=device)
        origins_pt, directions_pt = aquamvs_rotated_model.cast_ray(pixels_pt)

        # Compare
        for idx in valid_indices:
            origin_aquacal = torch.tensor(
                origins_aquacal[valid_indices.index(idx)],
                dtype=torch.float32,
                device=device,
            )
            direction_aquacal = torch.tensor(
                directions_aquacal[valid_indices.index(idx)],
                dtype=torch.float32,
                device=device,
            )

            assert torch.allclose(origins_pt[idx], origin_aquacal, atol=1e-5)
            assert torch.allclose(directions_pt[idx], direction_aquacal, atol=1e-5)

    def test_rotated_camera_project(
        self,
        aquacal_rotated_camera_and_interface,
        aquamvs_rotated_model,
        device,
        rotated_camera_params,
    ):
        """Test project on a 3D point grid with rotated camera."""
        camera, interface = aquacal_rotated_camera_and_interface
        p = rotated_camera_params

        # Camera center after rotation
        R_inv = p["R"].T
        C = -R_inv @ p["t"]

        # Generate points around camera center
        x_grid = np.linspace(C[0] - 0.3, C[0] + 0.3, 4)
        y_grid = np.linspace(C[1] - 0.3, C[1] + 0.3, 4)
        z_grid = np.linspace(p["water_z"] + 0.3, p["water_z"] + 0.6, 3)

        points_list = []
        for x in x_grid:
            for y in y_grid:
                for z in z_grid:
                    points_list.append([x, y, z])

        points_np = np.array(points_list, dtype=np.float64)

        # Call AquaCal
        pixels_aquacal = []
        valid_indices = []

        for i, point_np in enumerate(points_np):
            pixel_np = refractive_project(camera, interface, point_np)
            if pixel_np is not None:
                pixels_aquacal.append(pixel_np)
                valid_indices.append(i)

        if len(valid_indices) == 0:
            pytest.skip("No valid points in rotated camera grid")

        # Call AquaMVS
        points_pt = torch.tensor(points_np, dtype=torch.float32, device=device)
        pixels_pt, valid_pt = aquamvs_rotated_model.project(points_pt)

        # Compare
        for idx in valid_indices:
            pixel_aquacal = torch.tensor(
                pixels_aquacal[valid_indices.index(idx)],
                dtype=torch.float32,
                device=device,
            )
            assert valid_pt[idx].item()
            assert torch.allclose(pixels_pt[idx], pixel_aquacal, atol=1e-5)


class TestRoundtripCrossValidation:
    """Roundtrip consistency tests for the PyTorch implementation."""

    def test_roundtrip_project_then_cast_ray(
        self, aquamvs_model, device, reference_camera_params
    ):
        """Test roundtrip: project 3D points -> cast_ray -> reconstruct 3D points."""
        p = reference_camera_params

        # Generate grid of 3D points
        x_grid = np.linspace(0.135, 1.135, 5)
        y_grid = np.linspace(-0.5, 0.5, 5)
        z_grid = np.linspace(p["water_z"] + 0.2, p["water_z"] + 0.7, 4)

        points_list = []
        for x in x_grid:
            for y in y_grid:
                for z in z_grid:
                    points_list.append([x, y, z])

        original_points = torch.tensor(points_list, dtype=torch.float32, device=device)

        # Project to pixels
        pixels, valid = aquamvs_model.project(original_points)

        # Only process valid points
        valid_points = original_points[valid]
        valid_pixels = pixels[valid]

        if valid_points.shape[0] == 0:
            pytest.skip("No valid points in roundtrip test")

        # Cast rays from pixels
        origins, directions = aquamvs_model.cast_ray(valid_pixels)

        # Reconstruct 3D points at correct depth
        # depth = (point_z - origin_z) / direction_z
        depths = (valid_points[:, 2] - origins[:, 2]) / directions[:, 2]
        reconstructed_points = origins + depths.unsqueeze(-1) * directions

        # Compare
        assert torch.allclose(reconstructed_points, valid_points, atol=1e-4), (
            "Roundtrip reconstruction failed"
        )
