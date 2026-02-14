"""Pipeline context dataclass for precomputed data."""

from dataclasses import dataclass

import numpy as np

from ..calibration import CalibrationData, UndistortionData
from ..config import PipelineConfig
from ..projection.protocol import ProjectionModel


@dataclass
class PipelineContext:
    """Precomputed data that is constant across all frames.

    Created once by setup_pipeline() and reused for every frame.
    """

    config: PipelineConfig
    calibration: CalibrationData
    undistortion_maps: dict[str, UndistortionData]
    projection_models: dict[str, ProjectionModel]
    pairs: dict[str, list[str]]
    ring_cameras: list[str]
    auxiliary_cameras: list[str]
    device: str
    masks: dict[str, np.ndarray]
