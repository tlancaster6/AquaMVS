"""Multi-view stereo reconstruction of underwater surfaces with refractive modeling."""

from .calibration import (
    CalibrationData,
    CameraData,
    UndistortionData,
    compute_undistortion_maps,
    load_calibration_data,
    undistort_image,
)
from .config import (
    DenseStereoConfig,
    DeviceConfig,
    EvaluationConfig,
    FeatureExtractionConfig,
    FrameSamplingConfig,
    FusionConfig,
    MatchingConfig,
    PairSelectionConfig,
    PipelineConfig,
    SurfaceConfig,
)
from .fusion import (
    filter_all_depth_maps,
    filter_depth_map,
)
from .triangulation import (
    compute_depth_ranges,
    load_sparse_cloud,
    save_sparse_cloud,
    triangulate_all_pairs,
    triangulate_pair,
    triangulate_rays,
)

__version__ = "0.1.0"

__all__ = [
    "PipelineConfig",
    "FrameSamplingConfig",
    "FeatureExtractionConfig",
    "PairSelectionConfig",
    "MatchingConfig",
    "DenseStereoConfig",
    "FusionConfig",
    "SurfaceConfig",
    "EvaluationConfig",
    "DeviceConfig",
    "CalibrationData",
    "CameraData",
    "UndistortionData",
    "load_calibration_data",
    "compute_undistortion_maps",
    "undistort_image",
    "triangulate_rays",
    "triangulate_pair",
    "triangulate_all_pairs",
    "compute_depth_ranges",
    "save_sparse_cloud",
    "load_sparse_cloud",
    "filter_depth_map",
    "filter_all_depth_maps",
]
