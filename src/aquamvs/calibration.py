"""Bridge module for loading and converting AquaCal calibration data."""

from dataclasses import dataclass
from pathlib import Path

import torch
from aquacal.io.serialization import load_calibration as aquacal_load_calibration


@dataclass
class CameraData:
    """Per-camera calibration data as PyTorch tensors.

    Attributes:
        name: Camera identifier (e.g., "e3v82e0").
        K: Intrinsic matrix, shape (3, 3), float32.
        dist_coeffs: Distortion coefficients, shape (N,), float64.
            Pinhole: N=5 or N=8. Fisheye: N=4.
        R: Rotation matrix (world to camera), shape (3, 3), float32.
        t: Translation vector (world to camera), shape (3,), float32.
        image_size: Image dimensions as (width, height) in pixels.
        is_fisheye: Whether this camera uses the fisheye lens model.
        is_auxiliary: Whether this is an auxiliary camera (center camera).
    """

    name: str
    K: torch.Tensor  # shape (3, 3), float32
    dist_coeffs: torch.Tensor  # shape (N,), float64
    R: torch.Tensor  # shape (3, 3), float32
    t: torch.Tensor  # shape (3,), float32
    image_size: tuple[int, int]  # (width, height)
    is_fisheye: bool
    is_auxiliary: bool


@dataclass
class CalibrationData:
    """Complete calibration data converted to PyTorch tensors.

    Attributes:
        cameras: Per-camera calibration data, keyed by camera name.
        water_z: Z-coordinate of the water surface in world frame (meters).
        interface_normal: Interface normal vector, shape (3,), float32.
        n_air: Refractive index of air.
        n_water: Refractive index of water.
    """

    cameras: dict[str, CameraData]
    water_z: float
    interface_normal: torch.Tensor  # shape (3,), float32
    n_air: float
    n_water: float

    @property
    def ring_cameras(self) -> list[str]:
        """Names of non-auxiliary cameras (sorted for determinism).

        Returns:
            List of camera names where is_auxiliary=False, sorted alphabetically.
        """
        return sorted(
            [name for name, cam in self.cameras.items() if not cam.is_auxiliary]
        )

    @property
    def auxiliary_cameras(self) -> list[str]:
        """Names of auxiliary cameras (sorted for determinism).

        Returns:
            List of camera names where is_auxiliary=True, sorted alphabetically.
        """
        return sorted([name for name, cam in self.cameras.items() if cam.is_auxiliary])

    def camera_positions(self) -> dict[str, torch.Tensor]:
        """World-frame camera centers, computed as C = -R^T @ t.

        Returns:
            Dictionary mapping camera names to their centers in world frame.
            Each center is a tensor of shape (3,), same dtype as R and t.
        """
        positions = {}
        for name, cam in self.cameras.items():
            # C = -R^T @ t
            positions[name] = -cam.R.T @ cam.t
        return positions


def load_calibration_data(calibration_path: str | Path) -> CalibrationData:
    """Load AquaCal calibration and convert to PyTorch tensors.

    Args:
        calibration_path: Path to AquaCal calibration JSON file.

    Returns:
        CalibrationData with all parameters as PyTorch tensors.

    Raises:
        FileNotFoundError: If calibration file does not exist.
        ValueError: If calibration file format is invalid.
    """
    # Load calibration using AquaCal
    result = aquacal_load_calibration(calibration_path)

    # Convert per-camera data
    cameras = {}
    for name, cam_calib in result.cameras.items():
        # Convert intrinsics: K to float32, dist_coeffs to float64
        K = torch.from_numpy(cam_calib.intrinsics.K).to(torch.float32)
        dist_coeffs = torch.from_numpy(
            cam_calib.intrinsics.dist_coeffs
        )  # preserve float64

        # Convert extrinsics: R and t to float32
        R = torch.from_numpy(cam_calib.extrinsics.R).to(torch.float32)
        t_numpy = cam_calib.extrinsics.t
        # Handle both (3,) and (3, 1) shapes from AquaCal
        if t_numpy.ndim == 2:
            t_numpy = t_numpy.squeeze()
        t = torch.from_numpy(t_numpy).to(torch.float32)

        cameras[name] = CameraData(
            name=name,
            K=K,
            dist_coeffs=dist_coeffs,
            R=R,
            t=t,
            image_size=cam_calib.intrinsics.image_size,
            is_fisheye=cam_calib.intrinsics.is_fisheye,
            is_auxiliary=cam_calib.is_auxiliary,
        )

    # Extract interface parameters
    # water_z is stored per-camera but is the same for all after optimization
    water_z = next(iter(result.cameras.values())).interface_distance

    # Convert interface normal to float32 tensor
    interface_normal = torch.from_numpy(result.interface.normal).to(torch.float32)
    # Handle both (3,) and (3, 1) shapes
    if interface_normal.ndim == 2:
        interface_normal = interface_normal.squeeze()

    n_air = result.interface.n_air
    n_water = result.interface.n_water

    return CalibrationData(
        cameras=cameras,
        water_z=water_z,
        interface_normal=interface_normal,
        n_air=n_air,
        n_water=n_water,
    )
