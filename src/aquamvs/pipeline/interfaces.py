"""Protocol interfaces for pipeline abstraction."""

import logging
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from ..calibration import CameraData

logger = logging.getLogger(__name__)


@runtime_checkable
class FrameSource(Protocol):
    """Protocol for frame iteration over multi-camera video or image sequences.

    Abstracts both VideoSet and ImageDirectorySet to allow uniform iteration
    over synchronized frames from multiple cameras.
    """

    def iterate_frames(
        self, start: int, stop: int | None, step: int
    ) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
        """Iterate over synchronized frames from all cameras.

        Args:
            start: Starting frame index.
            stop: Ending frame index (exclusive), or None for all remaining frames.
            step: Frame step size (1 = every frame, 2 = every other frame, etc.).

        Yields:
            Tuple of (frame_idx, images) where images is a dict mapping camera
            name to raw BGR image (H, W, 3) uint8 array. May contain None values
            for cameras that failed to read.
        """
        ...

    def __enter__(self):
        """Enter context manager."""
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        ...


@runtime_checkable
class CalibrationProvider(Protocol):
    """Protocol for providing calibration data to the pipeline.

    Defines the interface for accessing camera calibration parameters and
    refractive geometry. The existing CalibrationData class already satisfies
    this protocol structurally — no modifications needed.
    """

    @property
    def cameras(self) -> dict[str, CameraData]:
        """Per-camera calibration data, keyed by camera name."""
        ...

    @property
    def water_z(self) -> float:
        """Z-coordinate of the water surface in world frame (meters)."""
        ...

    @property
    def n_water(self) -> float:
        """Refractive index of water."""
        ...

    @property
    def n_air(self) -> float:
        """Refractive index of air."""
        ...

    @property
    def interface_normal(self) -> torch.Tensor:
        """Interface normal vector, shape (3,), float32."""
        ...

    @property
    def ring_cameras(self) -> list[str]:
        """Names of non-auxiliary cameras (sorted for determinism)."""
        ...

    @property
    def auxiliary_cameras(self) -> list[str]:
        """Names of auxiliary cameras (sorted for determinism)."""
        ...

    def camera_positions(self) -> dict[str, torch.Tensor]:
        """World-frame camera centers, computed as C = -R^T @ t.

        Returns:
            Dictionary mapping camera names to their centers in world frame.
            Each center is a tensor of shape (3,), float32.
        """
        ...


def ensure_refractive_params(provider: CalibrationProvider) -> CalibrationProvider:
    """Ensure calibration has valid refractive parameters.

    Checks if water_z, n_water, and n_air are non-trivial (not zero or 1.0).
    If missing or set to trivial values, logs a descriptive warning and returns
    a provider that uses n_air = n_water = 1.0 (refraction-naive fallback).

    This allows the pipeline to operate on calibrations that lack refractive
    data (e.g., air-only setups or pinhole-only calibrations) without errors.

    Args:
        provider: Original calibration provider.

    Returns:
        CalibrationProvider with valid refractive parameters. Either the
        original provider (if params are valid) or a thin wrapper that
        overrides n_air and n_water to 1.0.
    """
    # Check for missing or trivial refractive parameters
    has_water_z = provider.water_z is not None and provider.water_z != 0.0
    has_n_water = provider.n_water is not None and provider.n_water != 1.0
    has_n_air = provider.n_air is not None and provider.n_air != 1.0

    if has_water_z and has_n_water and has_n_air:
        # All parameters present and non-trivial
        return provider

    # Missing or trivial parameters — warn and fallback
    logger.warning(
        "Calibration missing refractive parameters (water_z=%s, n_water=%s, n_air=%s). "
        "Falling back to refraction-naive mode (n_air=n_water=1.0). "
        "This is suitable for air-only setups or testing, but will produce incorrect "
        "results for underwater geometry.",
        provider.water_z,
        provider.n_water,
        provider.n_air,
    )

    # Create a simple wrapper that overrides n_air and n_water
    class RefractionNaiveProvider:
        """Wrapper that forces n_air = n_water = 1.0 for refraction-naive mode."""

        def __init__(self, original: CalibrationProvider):
            self._original = original

        @property
        def cameras(self) -> dict[str, CameraData]:
            return self._original.cameras

        @property
        def water_z(self) -> float:
            return self._original.water_z

        @property
        def n_water(self) -> float:
            return 1.0

        @property
        def n_air(self) -> float:
            return 1.0

        @property
        def interface_normal(self) -> torch.Tensor:
            return self._original.interface_normal

        @property
        def ring_cameras(self) -> list[str]:
            return self._original.ring_cameras

        @property
        def auxiliary_cameras(self) -> list[str]:
            return self._original.auxiliary_cameras

        def camera_positions(self) -> dict[str, torch.Tensor]:
            return self._original.camera_positions()

    return RefractionNaiveProvider(provider)
