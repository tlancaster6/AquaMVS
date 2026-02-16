"""Tests for calibration module."""

from unittest.mock import patch

import numpy as np
import torch
from aquacal.config.schema import (
    CalibrationResult,
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
    InterfaceParams,
)

from aquamvs.calibration import (
    CalibrationData,
    CameraData,
    UndistortionData,
    compute_undistortion_maps,
    load_calibration_data,
    undistort_image,
)


def create_synthetic_calibration(
    num_ring_cameras: int = 2,
    num_auxiliary_cameras: int = 1,
    water_z: float = 0.978,
) -> CalibrationResult:
    """Create a synthetic CalibrationResult for testing.

    Args:
        num_ring_cameras: Number of ring cameras (non-auxiliary).
        num_auxiliary_cameras: Number of auxiliary cameras.
        water_z: Z-coordinate of water surface.

    Returns:
        Synthetic CalibrationResult with known values.
    """
    cameras = {}

    # Create ring cameras (non-auxiliary)
    for i in range(num_ring_cameras):
        name = f"ring{i}"
        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        dist_coeffs = np.array([0.1, -0.2, 0.001, 0.002, 0.05])

        # Simple rotation and translation
        R = np.eye(3)
        t = np.array([i * 0.1, 0.0, 0.0])  # shape (3,)

        cameras[name] = CameraCalibration(
            name=name,
            intrinsics=CameraIntrinsics(
                K=K,
                dist_coeffs=dist_coeffs,
                image_size=(640, 480),
                is_fisheye=False,
            ),
            extrinsics=CameraExtrinsics(R=R, t=t),
            water_z=water_z,
            is_auxiliary=False,
        )

    # Create auxiliary cameras (fisheye)
    for i in range(num_auxiliary_cameras):
        name = f"aux{i}"
        K = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])
        dist_coeffs = np.array([0.15, -0.25, 0.03, 0.04])  # fisheye has 4 coeffs

        R = np.eye(3)
        t = np.array([0.0, i * 0.1, 0.0])  # shape (3,)

        cameras[name] = CameraCalibration(
            name=name,
            intrinsics=CameraIntrinsics(
                K=K,
                dist_coeffs=dist_coeffs,
                image_size=(640, 480),
                is_fisheye=True,
            ),
            extrinsics=CameraExtrinsics(R=R, t=t),
            water_z=water_z,
            is_auxiliary=True,
        )

    # Create interface parameters
    interface = InterfaceParams(
        normal=np.array([0.0, 0.0, -1.0]),  # points up from water to air
        n_air=1.0,
        n_water=1.333,
    )

    # Return minimal CalibrationResult (we only need cameras and interface)
    # The other fields (board, diagnostics, metadata) are not used by load_calibration_data
    return CalibrationResult(
        cameras=cameras,
        interface=interface,
        board=None,  # type: ignore
        diagnostics=None,  # type: ignore
        metadata=None,  # type: ignore
    )


class TestCameraData:
    """Tests for CameraData dataclass."""

    def test_creation(self):
        """Test CameraData creation."""
        K = torch.eye(3, dtype=torch.float32)
        dist_coeffs = torch.zeros(5, dtype=torch.float64)
        R = torch.eye(3, dtype=torch.float32)
        t = torch.zeros(3, dtype=torch.float32)

        cam = CameraData(
            name="test_cam",
            K=K,
            dist_coeffs=dist_coeffs,
            R=R,
            t=t,
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        )

        assert cam.name == "test_cam"
        assert cam.K.shape == (3, 3)
        assert cam.dist_coeffs.shape == (5,)
        assert cam.R.shape == (3, 3)
        assert cam.t.shape == (3,)
        assert cam.image_size == (640, 480)
        assert not cam.is_fisheye
        assert not cam.is_auxiliary


class TestCalibrationData:
    """Tests for CalibrationData dataclass."""

    def test_ring_cameras(self):
        """Test ring_cameras property returns sorted non-auxiliary camera names."""
        # Create cameras with names that test sorting
        cam1 = CameraData(
            name="cam_b",
            K=torch.eye(3, dtype=torch.float32),
            dist_coeffs=torch.zeros(5, dtype=torch.float64),
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        )
        cam2 = CameraData(
            name="cam_a",
            K=torch.eye(3, dtype=torch.float32),
            dist_coeffs=torch.zeros(5, dtype=torch.float64),
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        )
        cam3 = CameraData(
            name="aux_cam",
            K=torch.eye(3, dtype=torch.float32),
            dist_coeffs=torch.zeros(4, dtype=torch.float64),
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=True,
            is_auxiliary=True,
        )

        calib = CalibrationData(
            cameras={"cam_b": cam1, "cam_a": cam2, "aux_cam": cam3},
            water_z=0.978,
            interface_normal=torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),
            n_air=1.0,
            n_water=1.333,
        )

        ring = calib.ring_cameras
        assert ring == ["cam_a", "cam_b"]  # sorted alphabetically
        assert "aux_cam" not in ring

    def test_auxiliary_cameras(self):
        """Test auxiliary_cameras property returns sorted auxiliary camera names."""
        cam1 = CameraData(
            name="ring_cam",
            K=torch.eye(3, dtype=torch.float32),
            dist_coeffs=torch.zeros(5, dtype=torch.float64),
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        )
        cam2 = CameraData(
            name="aux_b",
            K=torch.eye(3, dtype=torch.float32),
            dist_coeffs=torch.zeros(4, dtype=torch.float64),
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=True,
            is_auxiliary=True,
        )
        cam3 = CameraData(
            name="aux_a",
            K=torch.eye(3, dtype=torch.float32),
            dist_coeffs=torch.zeros(4, dtype=torch.float64),
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=True,
            is_auxiliary=True,
        )

        calib = CalibrationData(
            cameras={"ring_cam": cam1, "aux_b": cam2, "aux_a": cam3},
            water_z=0.978,
            interface_normal=torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),
            n_air=1.0,
            n_water=1.333,
        )

        aux = calib.auxiliary_cameras
        assert aux == ["aux_a", "aux_b"]  # sorted alphabetically
        assert "ring_cam" not in aux

    def test_camera_positions(self):
        """Test camera_positions computes C = -R^T @ t correctly."""
        # Create camera with known R and t
        # For identity R and t = [1, 2, 3], C should be -[1, 2, 3]
        R = torch.eye(3, dtype=torch.float32)
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        cam = CameraData(
            name="test_cam",
            K=torch.eye(3, dtype=torch.float32),
            dist_coeffs=torch.zeros(5, dtype=torch.float64),
            R=R,
            t=t,
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        )

        calib = CalibrationData(
            cameras={"test_cam": cam},
            water_z=0.978,
            interface_normal=torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),
            n_air=1.0,
            n_water=1.333,
        )

        positions = calib.camera_positions()
        assert "test_cam" in positions
        expected = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32)
        assert torch.allclose(positions["test_cam"], expected, atol=1e-6)

    def test_camera_positions_with_rotation(self):
        """Test camera_positions with non-identity rotation."""
        # 90-degree rotation around Z-axis
        R = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        )
        t = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

        cam = CameraData(
            name="test_cam",
            K=torch.eye(3, dtype=torch.float32),
            dist_coeffs=torch.zeros(5, dtype=torch.float64),
            R=R,
            t=t,
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        )

        calib = CalibrationData(
            cameras={"test_cam": cam},
            water_z=0.978,
            interface_normal=torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),
            n_air=1.0,
            n_water=1.333,
        )

        positions = calib.camera_positions()
        # C = -R^T @ t
        # R^T = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
        # R^T @ [1, 0, 0] = [0, -1, 0]
        # -R^T @ [1, 0, 0] = [0, 1, 0]
        expected = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        assert torch.allclose(positions["test_cam"], expected, atol=1e-6)


class TestLoadCalibrationData:
    """Tests for load_calibration_data function."""

    def test_basic_loading(self):
        """Test basic loading with synthetic calibration data."""
        synthetic_result = create_synthetic_calibration(
            num_ring_cameras=2, num_auxiliary_cameras=1
        )

        with patch(
            "aquamvs.calibration.aquacal_load_calibration",
            return_value=synthetic_result,
        ):
            calib = load_calibration_data("dummy_path.json")

            # Check that we have the right number of cameras
            assert len(calib.cameras) == 3
            assert len(calib.ring_cameras) == 2
            assert len(calib.auxiliary_cameras) == 1

            # Check interface parameters
            assert calib.water_z == 0.978
            assert calib.n_air == 1.0
            assert calib.n_water == 1.333
            assert torch.allclose(
                calib.interface_normal,
                torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),
            )

    def test_tensor_shapes_and_dtypes(self):
        """Test that tensors have correct shapes and dtypes."""
        synthetic_result = create_synthetic_calibration(
            num_ring_cameras=1, num_auxiliary_cameras=1
        )

        with patch(
            "aquamvs.calibration.aquacal_load_calibration",
            return_value=synthetic_result,
        ):
            calib = load_calibration_data("dummy_path.json")

            # Check ring camera
            ring_cam = calib.cameras["ring0"]
            assert ring_cam.K.shape == (3, 3)
            assert ring_cam.K.dtype == torch.float32
            assert ring_cam.dist_coeffs.shape == (5,)  # pinhole
            assert ring_cam.dist_coeffs.dtype == torch.float64
            assert ring_cam.R.shape == (3, 3)
            assert ring_cam.R.dtype == torch.float32
            assert ring_cam.t.shape == (3,)
            assert ring_cam.t.dtype == torch.float32
            assert not ring_cam.is_fisheye
            assert not ring_cam.is_auxiliary

            # Check auxiliary camera (fisheye)
            aux_cam = calib.cameras["aux0"]
            assert aux_cam.K.shape == (3, 3)
            assert aux_cam.K.dtype == torch.float32
            assert aux_cam.dist_coeffs.shape == (4,)  # fisheye
            assert aux_cam.dist_coeffs.dtype == torch.float64
            assert aux_cam.R.shape == (3, 3)
            assert aux_cam.R.dtype == torch.float32
            assert aux_cam.t.shape == (3,)
            assert aux_cam.t.dtype == torch.float32
            assert aux_cam.is_fisheye
            assert aux_cam.is_auxiliary

            # Check interface normal
            assert calib.interface_normal.shape == (3,)
            assert calib.interface_normal.dtype == torch.float32

    def test_device_cpu(self):
        """Test that all tensors are on CPU by default."""
        synthetic_result = create_synthetic_calibration(num_ring_cameras=1)

        with patch(
            "aquamvs.calibration.aquacal_load_calibration",
            return_value=synthetic_result,
        ):
            calib = load_calibration_data("dummy_path.json")

            cam = calib.cameras["ring0"]
            assert cam.K.device.type == "cpu"
            assert cam.dist_coeffs.device.type == "cpu"
            assert cam.R.device.type == "cpu"
            assert cam.t.device.type == "cpu"
            assert calib.interface_normal.device.type == "cpu"

    def test_camera_classification(self):
        """Test that cameras are correctly classified as ring or auxiliary."""
        synthetic_result = create_synthetic_calibration(
            num_ring_cameras=3, num_auxiliary_cameras=2
        )

        with patch(
            "aquamvs.calibration.aquacal_load_calibration",
            return_value=synthetic_result,
        ):
            calib = load_calibration_data("dummy_path.json")

            ring = calib.ring_cameras
            aux = calib.auxiliary_cameras

            assert len(ring) == 3
            assert len(aux) == 2
            assert set(ring) == {"ring0", "ring1", "ring2"}
            assert set(aux) == {"aux0", "aux1"}

            # Verify sorting
            assert ring == sorted(ring)
            assert aux == sorted(aux)

    def test_camera_positions_computation(self):
        """Test that camera positions are computed correctly."""
        synthetic_result = create_synthetic_calibration(num_ring_cameras=2)

        with patch(
            "aquamvs.calibration.aquacal_load_calibration",
            return_value=synthetic_result,
        ):
            calib = load_calibration_data("dummy_path.json")

            positions = calib.camera_positions()

            # We set t = [i * 0.1, 0, 0] with identity R
            # So C = -R^T @ t = -t
            assert torch.allclose(
                positions["ring0"],
                torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                atol=1e-6,
            )
            assert torch.allclose(
                positions["ring1"],
                torch.tensor([-0.1, 0.0, 0.0], dtype=torch.float32),
                atol=1e-6,
            )

    def test_interface_parameters(self):
        """Test that interface parameters are extracted correctly."""
        water_z_test = 1.234
        synthetic_result = create_synthetic_calibration(water_z=water_z_test)

        with patch(
            "aquamvs.calibration.aquacal_load_calibration",
            return_value=synthetic_result,
        ):
            calib = load_calibration_data("dummy_path.json")

            assert calib.water_z == water_z_test
            assert calib.n_air == 1.0
            assert calib.n_water == 1.333
            assert torch.allclose(
                calib.interface_normal,
                torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),
            )

    def test_t_vector_shape_handling(self):
        """Test that both (3,) and (3, 1) shapes for t are handled correctly."""
        # Create calibration with t as (3, 1)
        cameras = {}
        name = "test_cam"
        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        dist_coeffs = np.array([0.1, -0.2, 0.001, 0.002, 0.05])
        R = np.eye(3)
        t_2d = np.array([[1.0], [2.0], [3.0]])  # shape (3, 1)

        cameras[name] = CameraCalibration(
            name=name,
            intrinsics=CameraIntrinsics(
                K=K, dist_coeffs=dist_coeffs, image_size=(640, 480), is_fisheye=False
            ),
            extrinsics=CameraExtrinsics(R=R, t=t_2d),
            water_z=0.978,
            is_auxiliary=False,
        )

        interface = InterfaceParams(
            normal=np.array([0.0, 0.0, -1.0]), n_air=1.0, n_water=1.333
        )

        synthetic_result = CalibrationResult(
            cameras=cameras,
            interface=interface,
            board=None,  # type: ignore
            diagnostics=None,  # type: ignore
            metadata=None,  # type: ignore
        )

        with patch(
            "aquamvs.calibration.aquacal_load_calibration",
            return_value=synthetic_result,
        ):
            calib = load_calibration_data("dummy_path.json")

            cam = calib.cameras["test_cam"]
            # Should be squeezed to (3,)
            assert cam.t.shape == (3,)
            assert torch.allclose(
                cam.t, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
            )

    def test_interface_normal_shape_handling(self):
        """Test that both (3,) and (3, 1) shapes for normal are handled correctly."""
        # Create calibration with normal as (3, 1)
        cameras = {}
        name = "test_cam"
        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        dist_coeffs = np.array([0.1, -0.2, 0.001, 0.002, 0.05])
        R = np.eye(3)
        t = np.array([0.0, 0.0, 0.0])

        cameras[name] = CameraCalibration(
            name=name,
            intrinsics=CameraIntrinsics(
                K=K, dist_coeffs=dist_coeffs, image_size=(640, 480), is_fisheye=False
            ),
            extrinsics=CameraExtrinsics(R=R, t=t),
            water_z=0.978,
            is_auxiliary=False,
        )

        # Normal as (3, 1)
        interface = InterfaceParams(
            normal=np.array([[0.0], [0.0], [-1.0]]), n_air=1.0, n_water=1.333
        )

        synthetic_result = CalibrationResult(
            cameras=cameras,
            interface=interface,
            board=None,  # type: ignore
            diagnostics=None,  # type: ignore
            metadata=None,  # type: ignore
        )

        with patch(
            "aquamvs.calibration.aquacal_load_calibration",
            return_value=synthetic_result,
        ):
            calib = load_calibration_data("dummy_path.json")

            # Should be squeezed to (3,)
            assert calib.interface_normal.shape == (3,)
            assert torch.allclose(
                calib.interface_normal,
                torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),
            )


class TestUndistortionData:
    """Tests for UndistortionData dataclass."""

    def test_creation(self):
        """Test UndistortionData creation."""
        K_new = torch.eye(3, dtype=torch.float32)
        map_x = np.zeros((480, 640), dtype=np.float32)
        map_y = np.zeros((480, 640), dtype=np.float32)

        undist = UndistortionData(K_new=K_new, map_x=map_x, map_y=map_y)

        assert undist.K_new.shape == (3, 3)
        assert undist.K_new.dtype == torch.float32
        assert undist.map_x.shape == (480, 640)
        assert undist.map_x.dtype == np.float32
        assert undist.map_y.shape == (480, 640)
        assert undist.map_y.dtype == np.float32


class TestComputeUndistortionMaps:
    """Tests for compute_undistortion_maps function."""

    def test_pinhole_zero_distortion(self):
        """Test pinhole undistortion with zero distortion."""
        # With zero distortion, K_new should be approximately equal to K
        K = torch.tensor(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        dist_coeffs = torch.zeros(5, dtype=torch.float64)

        cam = CameraData(
            name="test_cam",
            K=K,
            dist_coeffs=dist_coeffs,
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        )

        undist = compute_undistortion_maps(cam)

        # Check shapes
        assert undist.K_new.shape == (3, 3)
        assert undist.K_new.dtype == torch.float32
        assert undist.map_x.shape == (480, 640)  # (H, W)
        assert undist.map_y.shape == (480, 640)
        assert undist.map_x.dtype == np.float32
        assert undist.map_y.dtype == np.float32

        # With zero distortion, K_new should be close to K
        # (may differ slightly due to alpha=0 cropping)
        assert torch.allclose(undist.K_new, K, atol=1.0)

    def test_pinhole_with_distortion(self):
        """Test pinhole undistortion with non-zero distortion."""
        K = torch.tensor(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        dist_coeffs = torch.tensor([0.1, -0.2, 0.001, 0.002, 0.05], dtype=torch.float64)

        cam = CameraData(
            name="test_cam",
            K=K,
            dist_coeffs=dist_coeffs,
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        )

        undist = compute_undistortion_maps(cam)

        # Check shapes
        assert undist.K_new.shape == (3, 3)
        assert undist.map_x.shape == (480, 640)
        assert undist.map_y.shape == (480, 640)

        # K_new should be valid (positive focal lengths)
        assert undist.K_new[0, 0] > 0  # fx
        assert undist.K_new[1, 1] > 0  # fy

    def test_fisheye_undistortion_maps(self):
        """Test fisheye undistortion map generation."""
        K = torch.tensor(
            [[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        dist_coeffs = torch.tensor([0.15, -0.25, 0.03, 0.04], dtype=torch.float64)

        cam = CameraData(
            name="fisheye_cam",
            K=K,
            dist_coeffs=dist_coeffs,
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=True,
            is_auxiliary=True,
        )

        undist = compute_undistortion_maps(cam)

        # Check shapes
        assert undist.K_new.shape == (3, 3)
        assert undist.K_new.dtype == torch.float32
        assert undist.map_x.shape == (480, 640)
        assert undist.map_y.shape == (480, 640)
        assert undist.map_x.dtype == np.float32
        assert undist.map_y.dtype == np.float32

        # K_new should be valid
        assert undist.K_new[0, 0] > 0  # fx
        assert undist.K_new[1, 1] > 0  # fy

    def test_dispatch_correctness(self):
        """Test that is_fisheye flag dispatches to correct path."""
        # Create two cameras with same K and distortion but different lens models
        K = torch.tensor(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

        # Pinhole camera with 5 distortion coefficients
        pinhole_cam = CameraData(
            name="pinhole",
            K=K,
            dist_coeffs=torch.tensor(
                [0.1, -0.2, 0.001, 0.002, 0.05], dtype=torch.float64
            ),
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        )

        # Fisheye camera with 4 distortion coefficients
        fisheye_cam = CameraData(
            name="fisheye",
            K=K,
            dist_coeffs=torch.tensor([0.1, -0.2, 0.001, 0.002], dtype=torch.float64),
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=True,
            is_auxiliary=True,
        )

        pinhole_undist = compute_undistortion_maps(pinhole_cam)
        fisheye_undist = compute_undistortion_maps(fisheye_cam)

        # Both should produce valid results
        assert pinhole_undist.K_new.shape == (3, 3)
        assert fisheye_undist.K_new.shape == (3, 3)

        # Results should differ (different undistortion models)
        assert not torch.allclose(pinhole_undist.K_new, fisheye_undist.K_new)


class TestUndistortImage:
    """Tests for undistort_image function."""

    def test_identity_undistortion(self):
        """Test that zero distortion produces nearly identical output."""
        # Create a simple test image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Camera with zero distortion
        K = torch.tensor(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        dist_coeffs = torch.zeros(5, dtype=torch.float64)

        cam = CameraData(
            name="test_cam",
            K=K,
            dist_coeffs=dist_coeffs,
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        )

        undist = compute_undistortion_maps(cam)
        undistorted = undistort_image(image, undist)

        # Check shape and dtype preservation
        assert undistorted.shape == image.shape
        assert undistorted.dtype == image.dtype

        # With zero distortion, output should be very close to input
        # (small differences due to interpolation)
        diff = np.abs(undistorted.astype(np.float32) - image.astype(np.float32))
        assert np.mean(diff) < 5.0  # Small average difference

    def test_shape_preservation(self):
        """Test that undistort_image preserves shape and dtype."""
        image_shapes = [(480, 640, 3), (720, 1280, 3), (1080, 1920, 3)]

        for height, width, channels in image_shapes:
            image = np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)

            K = torch.tensor(
                [[500.0, 0.0, width / 2], [0.0, 500.0, height / 2], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            )
            dist_coeffs = torch.tensor(
                [0.1, -0.05, 0.001, 0.002, 0.01], dtype=torch.float64
            )

            cam = CameraData(
                name="test_cam",
                K=K,
                dist_coeffs=dist_coeffs,
                R=torch.eye(3, dtype=torch.float32),
                t=torch.zeros(3, dtype=torch.float32),
                image_size=(width, height),
                is_fisheye=False,
                is_auxiliary=False,
            )

            undist = compute_undistortion_maps(cam)
            undistorted = undistort_image(image, undist)

            assert undistorted.shape == image.shape
            assert undistorted.dtype == image.dtype

    def test_undistortion_modifies_image(self):
        """Test that non-zero distortion actually modifies the image."""
        # Create a synthetic pattern image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add a checkerboard pattern
        for i in range(0, 480, 40):
            for j in range(0, 640, 40):
                if (i // 40 + j // 40) % 2 == 0:
                    image[i : i + 40, j : j + 40] = 255

        # Camera with significant distortion
        K = torch.tensor(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        dist_coeffs = torch.tensor([0.3, -0.5, 0.002, 0.001, 0.1], dtype=torch.float64)

        cam = CameraData(
            name="test_cam",
            K=K,
            dist_coeffs=dist_coeffs,
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        )

        undist = compute_undistortion_maps(cam)
        undistorted = undistort_image(image, undist)

        # Image should be modified (not identical)
        assert not np.array_equal(undistorted, image)
