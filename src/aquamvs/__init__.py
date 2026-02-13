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
    backproject_depth_map,
    filter_all_depth_maps,
    filter_depth_map,
    fuse_depth_maps,
    load_point_cloud,
    save_point_cloud,
)
from .surface import (
    load_mesh,
    reconstruct_heightfield,
    reconstruct_poisson,
    reconstruct_surface,
    save_mesh,
)
from .triangulation import (
    compute_depth_ranges,
    filter_sparse_cloud,
    load_sparse_cloud,
    save_sparse_cloud,
    triangulate_all_pairs,
    triangulate_pair,
    triangulate_rays,
)
from .evaluation import (
    cloud_to_cloud_distance,
    height_map_difference,
    icp_align,
    reprojection_error,
)
from .pipeline import (
    PipelineContext,
    process_frame,
    run_pipeline,
    setup_pipeline,
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
    "filter_sparse_cloud",
    "compute_depth_ranges",
    "save_sparse_cloud",
    "load_sparse_cloud",
    "filter_depth_map",
    "filter_all_depth_maps",
    "backproject_depth_map",
    "fuse_depth_maps",
    "save_point_cloud",
    "load_point_cloud",
    "reconstruct_poisson",
    "reconstruct_heightfield",
    "reconstruct_surface",
    "save_mesh",
    "load_mesh",
    "icp_align",
    "cloud_to_cloud_distance",
    "height_map_difference",
    "reprojection_error",
    "PipelineContext",
    "setup_pipeline",
    "process_frame",
    "run_pipeline",
]
