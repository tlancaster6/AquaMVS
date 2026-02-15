"""Dataset loaders for benchmark ground truth generation."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from numpy.typing import NDArray

from ..calibration import (
    compute_undistortion_maps,
    load_calibration_data,
    undistort_image,
)
from .config import BenchmarkDataset
from .synthetic import (
    create_flat_plane_scene,
    create_undulating_scene,
    get_reference_geometry,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetContext:
    """Loaded dataset with ground truth references.

    Attributes:
        mesh: Ground truth surface mesh (None for ChArUco board).
        ground_truth_depths: Per-camera ground truth depth maps (None for ChArUco).
        charuco_corners: Per-camera detected ChArUco corner positions (None for synthetic).
        tolerance_mm: Tolerance for accurate completeness metric (None = skip).
        analytic_fn: Analytic ground truth function for synthetic scenes (None for real data).
    """

    mesh: o3d.geometry.TriangleMesh | None = None
    ground_truth_depths: dict[str, NDArray[np.float64]] | None = None
    charuco_corners: dict[str, NDArray[np.float64]] | None = None
    tolerance_mm: float | None = None
    analytic_fn: Callable | None = None


def load_dataset(dataset: BenchmarkDataset) -> DatasetContext:
    """Load ground truth for a benchmark dataset.

    Args:
        dataset: Dataset configuration.

    Returns:
        DatasetContext with mesh, ground truth depths, or ChArUco corners.

    Raises:
        ValueError: If dataset type is unknown or loading fails.
    """
    if dataset.type == "charuco":
        return _load_charuco_ground_truth(dataset)
    elif dataset.type == "synthetic_plane":
        return _load_synthetic_plane(dataset)
    elif dataset.type == "synthetic_surface":
        return _load_synthetic_surface(dataset)
    else:
        raise ValueError(f"Unknown dataset type: {dataset.type}")


def _load_charuco_ground_truth(dataset: BenchmarkDataset) -> DatasetContext:
    """Load ChArUco board dataset and detect corners on undistorted images.

    Args:
        dataset: ChArUco dataset configuration.

    Returns:
        DatasetContext with detected corner positions.
    """
    # Load dataset path as config directory
    dataset_path = Path(dataset.path)
    if not dataset_path.exists():
        raise ValueError(f"ChArUco dataset path does not exist: {dataset_path}")

    # Load calibration from dataset
    calibration_path = dataset_path / "calibration.json"
    if not calibration_path.exists():
        raise ValueError(f"ChArUco dataset missing calibration.json: {dataset_path}")

    calibration = load_calibration_data(str(calibration_path))

    # Load reference images
    images_dir = dataset_path / "images"
    if not images_dir.exists():
        raise ValueError(f"ChArUco dataset missing images/ directory: {dataset_path}")

    # Undistort images and detect ChArUco corners
    charuco_corners = {}

    # ChArUco board parameters (standard configuration)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard(
        (8, 5),  # Grid size
        0.04,  # Square length (meters)
        0.03,  # Marker length (meters)
        aruco_dict,
    )
    charuco_detector = cv2.aruco.CharucoDetector(board)

    for camera_name, camera in calibration.cameras.items():
        # Load raw image
        image_path = images_dir / f"{camera_name}.png"
        if not image_path.exists():
            logger.warning(f"Missing image for camera {camera_name}, skipping")
            continue

        raw_image = cv2.imread(str(image_path))
        if raw_image is None:
            logger.warning(f"Failed to load image for camera {camera_name}, skipping")
            continue

        # Undistort image (ChArUco detection on undistorted images per research)
        undistortion_maps = compute_undistortion_maps(camera)
        undistorted = undistort_image(raw_image, undistortion_maps)

        # Detect ChArUco corners
        charuco_corners_2d, charuco_ids, marker_corners, marker_ids = (
            charuco_detector.detectBoard(undistorted)
        )

        if charuco_corners_2d is not None and len(charuco_corners_2d) > 0:
            charuco_corners[camera_name] = charuco_corners_2d
            logger.info(
                f"Detected {len(charuco_corners_2d)} ChArUco corners in {camera_name}"
            )
        else:
            logger.warning(f"No ChArUco corners detected in {camera_name}")

    if not charuco_corners:
        raise ValueError(
            f"No ChArUco corners detected in any camera for {dataset.name}"
        )

    return DatasetContext(
        mesh=None,
        ground_truth_depths=None,
        charuco_corners=charuco_corners,
        tolerance_mm=dataset.ground_truth_tolerance_mm,
    )


def _load_synthetic_plane(dataset: BenchmarkDataset) -> DatasetContext:
    """Generate synthetic flat plane scene with ground truth.

    Args:
        dataset: Synthetic plane dataset configuration.

    Returns:
        DatasetContext with mesh and ground truth depth maps.
    """
    # Get reference geometry
    geometry = get_reference_geometry()

    # Compute depth and bounds
    depth_z = geometry["water_z_m"] + 0.2  # 200mm below water surface
    bounds = (-0.25, 0.25, -0.15, 0.15)  # 500mm x 300mm centered

    # Generate flat plane scene
    mesh, analytic_fn = create_flat_plane_scene(
        depth_z=depth_z,
        bounds=bounds,
        resolution=0.005,  # 5mm spacing
    )

    logger.info(
        f"Generated synthetic plane at depth_z={depth_z:.3f}m, "
        f"bounds={bounds}, resolution=0.005m"
    )

    return DatasetContext(
        mesh=mesh,
        ground_truth_depths=None,  # No depth maps for synthetic (rely on mesh + analytic_fn)
        charuco_corners=None,
        tolerance_mm=dataset.ground_truth_tolerance_mm,
        analytic_fn=analytic_fn,
    )


def _load_synthetic_surface(dataset: BenchmarkDataset) -> DatasetContext:
    """Generate synthetic undulating surface scene with ground truth.

    Args:
        dataset: Synthetic surface dataset configuration.

    Returns:
        DatasetContext with mesh and ground truth depth maps.
    """
    # Get reference geometry
    geometry = get_reference_geometry()

    # Compute base depth and bounds
    base_depth_z = geometry["water_z_m"] + 0.2  # 200mm base depth below water
    bounds = (-0.25, 0.25, -0.15, 0.15)  # 500mm x 300mm centered

    # Generate undulating surface scene
    mesh, analytic_fn = create_undulating_scene(
        base_depth_z=base_depth_z,
        amplitude=0.005,  # 5mm wave amplitude
        wavelength=0.05,  # 50mm wavelength
        bounds=bounds,
        resolution=0.005,  # 5mm spacing
    )

    logger.info(
        f"Generated synthetic surface at base_depth_z={base_depth_z:.3f}m, "
        f"amplitude=0.005m, wavelength=0.05m, bounds={bounds}, resolution=0.005m"
    )

    return DatasetContext(
        mesh=mesh,
        ground_truth_depths=None,  # No depth maps for synthetic (rely on mesh + analytic_fn)
        charuco_corners=None,
        tolerance_mm=dataset.ground_truth_tolerance_mm,
        analytic_fn=analytic_fn,
    )


def load_charuco_ground_truth(
    dataset_path: Path,
    tolerance_mm: float | None = None,
) -> DatasetContext:
    """Convenience function for loading ChArUco ground truth.

    Args:
        dataset_path: Path to ChArUco dataset directory.
        tolerance_mm: Optional tolerance for accurate completeness metric.

    Returns:
        DatasetContext with detected corner positions.
    """
    dataset = BenchmarkDataset(
        name="charuco",
        type="charuco",
        path=str(dataset_path),
        ground_truth_tolerance_mm=tolerance_mm,
    )
    return load_dataset(dataset)


__all__ = [
    "DatasetContext",
    "load_dataset",
    "load_charuco_ground_truth",
]
