"""Configuration management for AquaMVS pipeline."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ColorNormConfig:
    """Configuration for cross-camera color normalization.

    When enabled, undistorted images are normalized across cameras before
    any color sampling. This corrects white balance and exposure differences
    between cameras.

    Attributes:
        enabled: Enable cross-camera color normalization.
        method: Normalization method.
            "gain": Per-channel multiplicative gain to match cross-camera mean.
            "histogram": Per-channel histogram matching to the cross-camera
                aggregate histogram.
    """

    enabled: bool = False
    method: str = "gain"


@dataclass
class FrameSamplingConfig:
    """Configuration for frame sampling from video sequences.

    Attributes:
        start: First frame index to process.
        stop: Last frame index to process (None = end of video).
        step: Frame step interval (e.g., 100 = every 100th frame).
    """

    start: int = 0
    stop: int | None = None
    step: int = 1


@dataclass
class FeatureExtractionConfig:
    """Configuration for feature extraction.

    Attributes:
        extractor_type: Feature extractor backend ("superpoint", "aliked", or "disk").
        max_keypoints: Maximum number of keypoints to extract per image.
        detection_threshold: Detection confidence threshold.
        clahe_enabled: Apply CLAHE preprocessing before feature detection.
        clahe_clip_limit: Contrast limit for CLAHE (higher = more enhancement).
    """

    extractor_type: str = "superpoint"
    max_keypoints: int = 2048
    detection_threshold: float = 0.005
    clahe_enabled: bool = False
    clahe_clip_limit: float = 2.0


@dataclass
class PairSelectionConfig:
    """Configuration for selecting source cameras for stereo matching.

    Attributes:
        num_neighbors: Number of nearest ring cameras to use as sources.
        include_center: Whether to include the center (auxiliary) camera as a source.
    """

    num_neighbors: int = 4
    include_center: bool = True


@dataclass
class MatchingConfig:
    """Configuration for LightGlue feature matching.

    Attributes:
        filter_threshold: Match confidence threshold for filtering.
    """

    filter_threshold: float = 0.1


@dataclass
class DenseMatchingConfig:
    """Configuration for RoMa v2 dense matching.

    Attributes:
        certainty_threshold: Minimum overlap certainty for correspondence extraction.
        max_correspondences: Maximum number of correspondences to keep per pair.
    """

    certainty_threshold: float = 0.5
    max_correspondences: int = 100000


@dataclass
class DenseStereoConfig:
    """Configuration for plane-sweep dense stereo.

    Attributes:
        num_depths: Number of depth hypotheses in plane sweep.
        cost_function: Photometric cost function ("ncc" or "ssim").
        window_size: Local window size for cost computation (pixels, must be odd).
        depth_margin: Margin added to sparse depth range [d_min, d_max] (meters).
    """

    num_depths: int = 128
    cost_function: str = "ncc"
    window_size: int = 11
    depth_margin: float = 0.05


@dataclass
class FusionConfig:
    """Configuration for multi-view depth map fusion.

    Attributes:
        min_consistent_views: Minimum number of views that must agree for a point to survive.
        depth_tolerance: Maximum depth disagreement for consistency (meters).
            Used by plane-sweep geometric consistency filtering.
        roma_depth_tolerance: Maximum depth disagreement for RoMa pairwise depth
            aggregation (meters). RoMa operates at ~560px warp resolution, so
            triangulated depths have coarser quantization than plane-sweep.
            Defaults to 0.02m (4x the plane-sweep tolerance). (B.16)
        voxel_size: Voxel grid cell size for deduplication (meters).
        min_confidence: Minimum confidence threshold to consider a depth pixel.
    """

    min_consistent_views: int = 3
    depth_tolerance: float = 0.005
    roma_depth_tolerance: float = 0.02
    voxel_size: float = 0.001
    min_confidence: float = 0.1


@dataclass
class SurfaceConfig:
    """Configuration for surface reconstruction from point clouds.

    Attributes:
        method: Surface reconstruction method ("poisson", "heightfield", or "bpa").
        poisson_depth: Octree depth for Poisson reconstruction.
        grid_resolution: Grid cell size for height-field interpolation (meters).
        bpa_radii: List of ball radii for Ball Pivoting Algorithm (meters),
            or None to auto-estimate from point spacing.
    """

    method: str = "poisson"
    poisson_depth: int = 9
    grid_resolution: float = 0.002
    bpa_radii: list[float] | None = None


@dataclass
class EvaluationConfig:
    """Configuration for reconstruction evaluation metrics.

    Attributes:
        icp_max_distance: ICP correspondence distance threshold (meters).
    """

    icp_max_distance: float = 0.01


@dataclass
class DeviceConfig:
    """Configuration for PyTorch device selection.

    Attributes:
        device: PyTorch device string ("cpu" or "cuda").
    """

    device: str = "cpu"


@dataclass
class OutputConfig:
    """Configuration for output artifact persistence.

    Attributes:
        save_features: Save features and matches (.pt files). Off by default.
        save_depth_maps: Save per-camera depth + confidence maps (.npz). On by default.
        save_point_cloud: Save fused point cloud (.ply). On by default.
        save_mesh: Save surface mesh (.ply). On by default.
        keep_intermediates: Keep depth maps after fusion. If False, depth maps
            are deleted after successful fusion to save space. On by default.
    """

    save_features: bool = False
    save_depth_maps: bool = True
    save_point_cloud: bool = True
    save_mesh: bool = True
    keep_intermediates: bool = True


VALID_COLOR_NORM_METHODS = ["gain", "histogram"]
VALID_VIZ_STAGES = ["depth", "features", "scene", "rig", "summary"]
VALID_EXTRACTORS = ["superpoint", "aliked", "disk"]
VALID_MATCHERS = ["lightglue", "roma"]


@dataclass
class VizConfig:
    """Configuration for visualization output.

    When enabled=False, no visualization is generated (zero overhead).
    When enabled=True, only the stages listed in `stages` are rendered.

    Attributes:
        enabled: Master switch for all visualization.
        stages: List of visualization stages to run. Valid values:
            "depth", "features", "scene", "rig", "summary".
            Empty list with enabled=True means all stages.
    """

    enabled: bool = False
    stages: list[str] = field(default_factory=list)


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark sweep.

    The runner computes the cross product of extractors x clahe to
    produce the sweep matrix. Adding new extractors means appending
    to the list -- no code change.

    Attributes:
        extractors: List of extractor backends to sweep.
        clahe: List of CLAHE on/off settings to sweep.
    """

    extractors: list[str] = field(
        default_factory=lambda: ["superpoint", "aliked", "disk"]
    )
    clahe: list[bool] = field(default_factory=lambda: [True, False])


@dataclass
class PipelineConfig:
    """Top-level configuration for the AquaMVS reconstruction pipeline.

    Attributes:
        calibration_path: Path to AquaCal calibration JSON file.
        output_dir: Root output directory for reconstruction results.
        camera_video_map: Mapping from camera name to video file path.
        mask_dir: Optional directory containing per-camera ROI mask PNGs.
            If None, no masking is applied. Masks suppress features and depth
            outside the valid region.
        pipeline_mode: Pipeline execution mode ("sparse" or "full").
            "sparse" mode stops after sparse triangulation and produces point cloud + mesh.
            "full" mode runs the complete pipeline including dense stereo and fusion.
        frame_sampling: Frame sampling configuration.
        feature_extraction: Feature extraction configuration.
        pair_selection: Camera pair selection configuration.
        matching: Feature matching configuration.
        dense_stereo: Dense stereo configuration.
        fusion: Multi-view fusion configuration.
        surface: Surface reconstruction configuration.
        evaluation: Evaluation configuration.
        device: Device configuration.
        output: Output artifact persistence configuration.
        visualization: Visualization output configuration.
    """

    # Session fields (no sensible defaults)
    calibration_path: str = ""
    output_dir: str = ""
    camera_video_map: dict[str, str] = field(default_factory=dict)
    mask_dir: str | None = None
    pipeline_mode: str = "full"
    matcher_type: str = "lightglue"

    # Stage configurations (all have defaults)
    color_norm: ColorNormConfig = field(default_factory=ColorNormConfig)
    frame_sampling: FrameSamplingConfig = field(default_factory=FrameSamplingConfig)
    feature_extraction: FeatureExtractionConfig = field(
        default_factory=FeatureExtractionConfig
    )
    pair_selection: PairSelectionConfig = field(default_factory=PairSelectionConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    dense_matching: DenseMatchingConfig = field(default_factory=DenseMatchingConfig)
    dense_stereo: DenseStereoConfig = field(default_factory=DenseStereoConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    visualization: VizConfig = field(default_factory=VizConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        # Validate color normalization method
        if self.color_norm.method not in VALID_COLOR_NORM_METHODS:
            raise ValueError(
                f"Invalid color_norm method: {self.color_norm.method!r}. "
                f"Valid methods: {VALID_COLOR_NORM_METHODS}"
            )

        # Validate pipeline mode
        if self.pipeline_mode not in ["sparse", "full"]:
            raise ValueError(
                f"Invalid pipeline_mode: {self.pipeline_mode!r}. "
                "Must be 'sparse' or 'full'."
            )

        # Validate cost function
        if self.dense_stereo.cost_function not in ["ncc", "ssim"]:
            raise ValueError(
                f"Invalid cost_function: {self.dense_stereo.cost_function}. "
                "Must be 'ncc' or 'ssim'."
            )

        # Validate surface method
        if self.surface.method not in ["poisson", "heightfield", "bpa"]:
            raise ValueError(
                f"Invalid surface method: {self.surface.method}. "
                "Must be 'poisson', 'heightfield', or 'bpa'."
            )

        # Validate window size (must be odd and positive)
        if self.dense_stereo.window_size <= 0 or self.dense_stereo.window_size % 2 == 0:
            raise ValueError(
                f"Invalid window_size: {self.dense_stereo.window_size}. "
                "Must be positive and odd."
            )

        # Validate device
        if self.device.device not in ["cpu", "cuda"]:
            raise ValueError(
                f"Invalid device: {self.device.device}. Must be 'cpu' or 'cuda'."
            )

        # Validate extractor type
        if self.feature_extraction.extractor_type not in VALID_EXTRACTORS:
            raise ValueError(
                f"Invalid extractor_type: {self.feature_extraction.extractor_type!r}. "
                f"Valid types: {VALID_EXTRACTORS}"
            )

        # Validate matcher type
        if self.matcher_type not in VALID_MATCHERS:
            raise ValueError(
                f"Invalid matcher_type: {self.matcher_type!r}. "
                f"Valid types: {VALID_MATCHERS}"
            )

        # Validate viz stages
        for stage in self.visualization.stages:
            if stage not in VALID_VIZ_STAGES:
                raise ValueError(
                    f"Invalid visualization stage: {stage!r}. "
                    f"Valid stages: {VALID_VIZ_STAGES}"
                )

        # Validate benchmark extractors
        for extractor in self.benchmark.extractors:
            if extractor not in VALID_EXTRACTORS:
                raise ValueError(
                    f"Invalid benchmark extractor: {extractor!r}. "
                    f"Valid extractors: {VALID_EXTRACTORS}"
                )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load configuration from a YAML file.

        Missing fields use their default values. Loaded values are merged over defaults.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Loaded configuration with defaults filled in.

        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If the file is not valid YAML.
        """
        path = Path(path)
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        # Helper to instantiate dataclass from dict, merging over defaults
        def _build_dataclass(cls_type: type, data_dict: dict[str, Any] | None):
            if data_dict is None:
                return cls_type()
            return cls_type(**data_dict)

        # Extract sub-config dicts
        color_norm = data.pop("color_norm", None)
        frame_sampling = data.pop("frame_sampling", None)
        feature_extraction = data.pop("feature_extraction", None)
        pair_selection = data.pop("pair_selection", None)
        matching = data.pop("matching", None)
        dense_matching = data.pop("dense_matching", None)
        dense_stereo = data.pop("dense_stereo", None)
        fusion = data.pop("fusion", None)
        surface = data.pop("surface", None)
        evaluation = data.pop("evaluation", None)
        device = data.pop("device", None)
        output = data.pop("output", None)
        visualization = data.pop("visualization", None)
        benchmark = data.pop("benchmark", None)

        # Build sub-configs
        config = cls(
            calibration_path=data.get("calibration_path", ""),
            output_dir=data.get("output_dir", ""),
            camera_video_map=data.get("camera_video_map", {}),
            mask_dir=data.get("mask_dir", None),
            pipeline_mode=data.get("pipeline_mode", "full"),
            matcher_type=data.get("matcher_type", "lightglue"),
            color_norm=_build_dataclass(ColorNormConfig, color_norm),
            frame_sampling=_build_dataclass(FrameSamplingConfig, frame_sampling),
            feature_extraction=_build_dataclass(
                FeatureExtractionConfig, feature_extraction
            ),
            pair_selection=_build_dataclass(PairSelectionConfig, pair_selection),
            matching=_build_dataclass(MatchingConfig, matching),
            dense_matching=_build_dataclass(DenseMatchingConfig, dense_matching),
            dense_stereo=_build_dataclass(DenseStereoConfig, dense_stereo),
            fusion=_build_dataclass(FusionConfig, fusion),
            surface=_build_dataclass(SurfaceConfig, surface),
            evaluation=_build_dataclass(EvaluationConfig, evaluation),
            device=_build_dataclass(DeviceConfig, device),
            output=_build_dataclass(OutputConfig, output),
            visualization=_build_dataclass(VizConfig, visualization),
            benchmark=_build_dataclass(BenchmarkConfig, benchmark),
        )

        # Validate the loaded config
        config.validate()

        return config

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file.

        All fields including defaults are written for explicitness.

        Args:
            path: Path to output YAML file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict using dataclasses.asdict (handles nested dataclasses)
        data = asdict(self)

        with open(path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
