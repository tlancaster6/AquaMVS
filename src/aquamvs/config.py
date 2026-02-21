"""Configuration management for AquaMVS pipeline."""

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)

# Keys that were removed from the config schema (outputs are now always saved).
# Old configs containing these keys will produce a warning and have them stripped.
REMOVED_KEYS = {"save_depth_maps", "save_point_cloud", "save_mesh"}

# Valid values for enum fields
VALID_COLOR_NORM_METHODS = ["gain", "histogram"]
VALID_VIZ_STAGES = ["depth", "features", "scene", "rig", "summary"]
VALID_EXTRACTORS = ["superpoint", "aliked", "disk"]
VALID_MATCHERS = ["lightglue", "roma"]


class QualityPreset(str, Enum):
    """Quality presets for reconstruction pipeline.

    Each preset provides a different speed/accuracy tradeoff:
    - FAST: Fastest processing, lower quality (fewer depths, smaller windows, larger voxels)
    - BALANCED: Good balance of speed and quality (default settings)
    - QUALITY: Highest quality, slower processing (more depths, larger windows, smaller voxels)
    """

    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"


PRESET_CONFIGS = {
    QualityPreset.FAST: {
        "num_depths": 64,
        "window_size": 7,
        "depth_batch_size": 8,
        "max_keypoints": 1024,
        "voxel_size": 0.002,
        "poisson_depth": 8,
    },
    QualityPreset.BALANCED: {
        "num_depths": 128,
        "window_size": 11,
        "depth_batch_size": 4,
        "max_keypoints": 2048,
        "voxel_size": 0.001,
        "poisson_depth": 9,
    },
    QualityPreset.QUALITY: {
        "num_depths": 256,
        "window_size": 15,
        "depth_batch_size": 1,
        "max_keypoints": 4096,
        "voxel_size": 0.0005,
        "poisson_depth": 10,
    },
}


class PreprocessingConfig(BaseModel):
    """Configuration for preprocessing (color normalization + frame sampling).

    Consolidates ColorNormConfig and FrameSamplingConfig.

    Attributes:
        color_norm_enabled: Enable cross-camera color normalization.
        color_norm_method: Normalization method ("gain" or "histogram").
        frame_start: First frame index to process.
        frame_stop: Last frame index to process (None = end of video).
        frame_step: Frame step interval (e.g., 100 = every 100th frame).
    """

    model_config = ConfigDict(extra="allow")

    # Color normalization
    color_norm_enabled: bool = False
    color_norm_method: Literal["gain", "histogram"] = "gain"

    # Frame sampling
    frame_start: int = 0
    frame_stop: int | None = None
    frame_step: int = 1

    @model_validator(mode="after")
    def warn_extra_fields(self) -> "PreprocessingConfig":
        """Warn about unknown configuration keys."""
        if self.__pydantic_extra__:
            logger.warning(
                "Unknown config keys in PreprocessingConfig (ignored): %s",
                list(self.__pydantic_extra__.keys()),
            )
        return self


class SparseMatchingConfig(BaseModel):
    """Configuration for sparse matching (extraction + pair selection + matching).

    Consolidates FeatureExtractionConfig, PairSelectionConfig, and MatchingConfig.

    Attributes:
        extractor_type: Feature extractor backend.
        max_keypoints: Maximum number of keypoints to extract per image.
        detection_threshold: Detection confidence threshold.
        clahe_enabled: Apply CLAHE preprocessing before feature detection.
        clahe_clip_limit: Contrast limit for CLAHE (higher = more enhancement).
        num_neighbors: Number of nearest ring cameras to use as sources.
        include_center: Whether to include the center (auxiliary) camera as a source.
        filter_threshold: Match confidence threshold for filtering.
    """

    model_config = ConfigDict(extra="allow")

    # Feature extraction
    extractor_type: Literal["superpoint", "aliked", "disk"] = "superpoint"
    max_keypoints: int = 2048
    detection_threshold: float = 0.005
    clahe_enabled: bool = False
    clahe_clip_limit: float = 2.0

    # Pair selection
    num_neighbors: int = 4
    include_center: bool = True

    # Matching
    filter_threshold: float = 0.1

    @model_validator(mode="after")
    def warn_extra_fields(self) -> "SparseMatchingConfig":
        """Warn about unknown configuration keys."""
        if self.__pydantic_extra__:
            logger.warning(
                "Unknown config keys in SparseMatchingConfig (ignored): %s",
                list(self.__pydantic_extra__.keys()),
            )
        return self


class DenseMatchingConfig(BaseModel):
    """Configuration for RoMa v2 dense matching.

    Attributes:
        certainty_threshold: Minimum overlap certainty for correspondence extraction.
        max_correspondences: Maximum number of correspondences to keep per pair.
    """

    model_config = ConfigDict(extra="allow")

    certainty_threshold: float = 0.5
    max_correspondences: int = 100000

    @model_validator(mode="after")
    def warn_extra_fields(self) -> "DenseMatchingConfig":
        """Warn about unknown configuration keys."""
        if self.__pydantic_extra__:
            logger.warning(
                "Unknown config keys in DenseMatchingConfig (ignored): %s",
                list(self.__pydantic_extra__.keys()),
            )
        return self


class ReconstructionConfig(BaseModel):
    """Configuration for reconstruction (stereo + fusion + surface + outliers).

    Consolidates DenseStereoConfig, FusionConfig, SurfaceConfig, and OutlierRemovalConfig.

    Attributes:
        num_depths: Number of depth hypotheses in plane sweep.
        cost_function: Photometric cost function.
        window_size: Local window size for cost computation (pixels, must be odd).
        depth_margin: Margin added to sparse depth range [d_min, d_max] (meters).
        depth_batch_size: Number of depth planes to process per batch in plane sweep (1 = no batching).
        min_consistent_views: Minimum number of views that must agree for a point to survive.
        depth_tolerance: Maximum depth disagreement for consistency (meters).
        roma_depth_tolerance: Maximum depth disagreement for RoMa pairwise depth aggregation (meters).
        voxel_size: Voxel grid cell size for deduplication (meters).
        min_confidence: Minimum confidence threshold to consider a depth pixel.
        surface_method: Surface reconstruction method.
        poisson_depth: Octree depth for Poisson reconstruction.
        grid_resolution: Grid cell size for height-field interpolation (meters).
        bpa_radii: List of ball radii for Ball Pivoting Algorithm (meters), or None to auto-estimate.
        target_faces: Target triangle count for mesh simplification (None = no simplification).
        outlier_removal_enabled: Enable statistical outlier removal.
        outlier_nb_neighbors: Number of neighbors for mean distance calculation.
        outlier_std_ratio: Standard deviation ratio threshold.
    """

    model_config = ConfigDict(extra="allow")

    # Dense stereo
    num_depths: int = 128
    cost_function: Literal["ncc", "ssim"] = "ncc"
    window_size: int = 11
    depth_margin: float = 0.05
    depth_batch_size: int = 4

    # Fusion
    min_consistent_views: int = 3
    depth_tolerance: float = 0.005
    roma_depth_tolerance: float = 0.02
    voxel_size: float = 0.001
    min_confidence: float = 0.1

    # Surface
    surface_method: Literal["poisson", "heightfield", "bpa"] = "poisson"
    poisson_depth: int = 9
    grid_resolution: float = 0.002
    bpa_radii: list[float] | None = None
    target_faces: int | None = None

    # Outlier removal
    outlier_removal_enabled: bool = True
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0

    @field_validator("window_size")
    @classmethod
    def validate_window_size(cls, v: int) -> int:
        """Validate that window_size is positive and odd."""
        if v <= 0 or v % 2 == 0:
            raise ValueError(f"window_size must be positive and odd, got {v}")
        return v

    @model_validator(mode="after")
    def warn_extra_fields(self) -> "ReconstructionConfig":
        """Warn about unknown configuration keys."""
        if self.__pydantic_extra__:
            logger.warning(
                "Unknown config keys in ReconstructionConfig (ignored): %s",
                list(self.__pydantic_extra__.keys()),
            )
        return self


class RuntimeConfig(BaseModel):
    """Configuration for runtime settings (device + output + viz + benchmark + evaluation).

    Consolidates DeviceConfig, OutputConfig, VizConfig, BenchmarkConfig, and EvaluationConfig.

    Depth maps, point clouds, and meshes are always saved (no toggle).

    Attributes:
        device: PyTorch device string.
        save_features: Save features and matches (.pt files).
        keep_intermediates: Keep depth maps after fusion.
        save_consistency_maps: Save consistency maps as colormapped PNG + NPZ.
        viz_enabled: Master switch for all visualization.
        viz_stages: List of visualization stages to run.
        benchmark_extractors: List of extractor backends to sweep.
        benchmark_clahe: List of CLAHE on/off settings to sweep.
        icp_max_distance: ICP correspondence distance threshold (meters).
        quiet: Suppress progress output.
    """

    model_config = ConfigDict(extra="allow")

    # Device
    device: Literal["cpu", "cuda"] = "cpu"

    # Output
    save_features: bool = False
    keep_intermediates: bool = True
    save_consistency_maps: bool = False

    # Visualization
    viz_enabled: bool = False
    viz_stages: list[str] = Field(default_factory=list)

    # Benchmark
    benchmark_extractors: list[str] = Field(
        default_factory=lambda: ["superpoint", "aliked", "disk"]
    )
    benchmark_clahe: list[bool] = Field(default_factory=lambda: [True, False])

    # Evaluation
    icp_max_distance: float = 0.01

    # Progress
    quiet: bool = False

    @field_validator("viz_stages")
    @classmethod
    def validate_viz_stages(cls, v: list[str]) -> list[str]:
        """Validate that all viz_stages are valid."""
        for stage in v:
            if stage not in VALID_VIZ_STAGES:
                raise ValueError(
                    f"Invalid visualization stage: {stage!r}. "
                    f"Valid stages: {VALID_VIZ_STAGES}"
                )
        return v

    @field_validator("benchmark_extractors")
    @classmethod
    def validate_benchmark_extractors(cls, v: list[str]) -> list[str]:
        """Validate that all benchmark_extractors are valid."""
        for extractor in v:
            if extractor not in VALID_EXTRACTORS:
                raise ValueError(
                    f"Invalid benchmark extractor: {extractor!r}. "
                    f"Valid extractors: {VALID_EXTRACTORS}"
                )
        return v

    @model_validator(mode="after")
    def warn_extra_fields(self) -> "RuntimeConfig":
        """Warn about unknown configuration keys."""
        if self.__pydantic_extra__:
            logger.warning(
                "Unknown config keys in RuntimeConfig (ignored): %s",
                list(self.__pydantic_extra__.keys()),
            )
        return self


class PipelineConfig(BaseModel):
    """Top-level configuration for the AquaMVS reconstruction pipeline.

    Attributes:
        calibration_path: Path to AquaCal calibration JSON file.
        output_dir: Root output directory for reconstruction results.
        camera_input_map: Mapping from camera name to input path (video file or image directory).
        mask_dir: Optional directory containing per-camera ROI mask PNGs.
        pipeline_mode: Pipeline execution mode ("sparse" or "full").
        matcher_type: Matcher backend ("lightglue" or "roma").
        quality_preset: Optional quality preset (fast/balanced/quality) to apply default values.
        preprocessing: Preprocessing configuration.
        sparse_matching: Sparse matching configuration.
        dense_matching: Dense matching configuration.
        reconstruction: Reconstruction configuration.
        runtime: Runtime configuration.
    """

    model_config = ConfigDict(extra="allow")

    # Required fields (no sensible defaults)
    calibration_path: str = ""
    output_dir: str = ""
    camera_input_map: dict[str, str] = Field(default_factory=dict)

    # Optional with defaults
    mask_dir: str | None = None
    pipeline_mode: Literal["sparse", "full"] = "full"
    matcher_type: Literal["lightglue", "roma"] = "lightglue"
    quality_preset: QualityPreset | None = None

    # Sub-configs
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    sparse_matching: SparseMatchingConfig = Field(default_factory=SparseMatchingConfig)
    dense_matching: DenseMatchingConfig = Field(default_factory=DenseMatchingConfig)
    reconstruction: ReconstructionConfig = Field(default_factory=ReconstructionConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    def apply_preset(self, preset: QualityPreset) -> "PipelineConfig":
        """Apply a quality preset to this configuration.

        Only applies preset values to parameters that are still at their defaults.
        User-specified values take precedence and are not overridden.

        Args:
            preset: Quality preset to apply.

        Returns:
            Self for method chaining.
        """
        if preset not in PRESET_CONFIGS:
            logger.warning(f"Unknown quality preset: {preset}")
            return self

        preset_values = PRESET_CONFIGS[preset]

        # Get default configs for comparison
        default_reconstruction = ReconstructionConfig()
        default_sparse = SparseMatchingConfig()

        # Apply reconstruction params if they're at defaults
        for key in [
            "num_depths",
            "window_size",
            "depth_batch_size",
            "voxel_size",
            "poisson_depth",
        ]:
            if key in preset_values:
                current_value = getattr(self.reconstruction, key)
                default_value = getattr(default_reconstruction, key)
                if current_value == default_value:
                    setattr(self.reconstruction, key, preset_values[key])

        # Apply sparse matching params if they're at defaults
        if (
            "max_keypoints" in preset_values
            and self.sparse_matching.max_keypoints == default_sparse.max_keypoints
        ):
            self.sparse_matching.max_keypoints = preset_values["max_keypoints"]

        return self

    @model_validator(mode="after")
    def auto_apply_preset(self) -> "PipelineConfig":
        """Warn when quality_preset is present in config (no longer applied at runtime).

        Presets are now baked in at init time via ``aquamvs init --preset <name>``.
        Loading a config with quality_preset set does NOT silently override
        user-specified values.
        """
        if self.quality_preset is not None:
            logger.warning(
                "quality_preset '%s' in config is deprecated and will NOT be applied "
                "at runtime. Re-generate your config with "
                "'aquamvs init --preset %s' to bake preset values in explicitly.",
                self.quality_preset.value,
                self.quality_preset.value,
            )
        return self

    @model_validator(mode="after")
    def check_cross_stage_constraints(self) -> "PipelineConfig":
        """Validate cross-stage constraints and warn about extra fields."""
        # Warn about RoMa with low certainty threshold
        if (
            self.matcher_type == "roma"
            and self.dense_matching.certainty_threshold < 0.1
        ):
            logger.warning(
                "matcher_type=roma with certainty_threshold=%.2f (< 0.1) may produce "
                "unreliable results. Consider increasing certainty_threshold.",
                self.dense_matching.certainty_threshold,
            )

        # Force save_features when features viz is active (viz pass needs them on disk)
        if (
            self.runtime.viz_enabled
            and (not self.runtime.viz_stages or "features" in self.runtime.viz_stages)
            and not self.runtime.save_features
        ):
            logger.info("Enabling save_features (required by features visualization)")
            self.runtime.save_features = True

        # Warn about RoMa + sparse (wasteful — pays full dense matching cost
        # but discards most of the warp field to extract sparse correspondences)
        if self.matcher_type == "roma" and self.pipeline_mode == "sparse":
            logger.warning(
                "matcher_type=roma with pipeline_mode=sparse is not recommended. "
                "RoMa produces dense warps; sparse mode discards most of that "
                "output to extract keypoints. Use pipeline_mode=full to leverage "
                "the full warp field, or switch to matcher_type=lightglue for "
                "efficient sparse matching."
            )

        # Warn about unknown top-level keys
        if self.__pydantic_extra__:
            logger.warning(
                "Unknown config keys in PipelineConfig (ignored): %s",
                list(self.__pydantic_extra__.keys()),
            )

        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load configuration from a YAML file.

        Missing fields use their default values. Loaded values are merged over defaults.
        Supports backward compatibility with old flat structure.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Loaded configuration with defaults filled in.

        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If the file is not valid YAML.
            ValueError: If validation fails (with all errors collected).
        """
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        # Backward compatibility: remap old flat structure to new nested structure
        data = cls._migrate_legacy_config(data)

        # Log info about missing sections (using defaults)
        cls._log_default_sections(data)

        try:
            config = cls.model_validate(data)
        except ValidationError as e:
            # Format validation errors with YAML paths
            formatted_errors = format_validation_errors(e)
            raise ValueError(
                f"Configuration validation failed:\n{formatted_errors}"
            ) from None

        return config

    @staticmethod
    def _migrate_legacy_config(data: dict[str, Any]) -> dict[str, Any]:
        """Migrate legacy flat config structure to new nested structure.

        Args:
            data: Configuration dictionary loaded from YAML.

        Returns:
            Migrated configuration dictionary.
        """
        # Strip removed keys (outputs are now always saved; these flags no longer exist).
        for key in list(data.keys()):
            if key in REMOVED_KEYS:
                logger.warning(
                    "Config key '%s' has been removed and will be ignored. "
                    "All outputs are now always saved. Remove this key from your config.",
                    key,
                )
                del data[key]
        # Also check inside 'runtime' subdict (old structure placed them there).
        if "runtime" in data and isinstance(data["runtime"], dict):
            for key in list(data["runtime"].keys()):
                if key in REMOVED_KEYS:
                    logger.warning(
                        "Config key 'runtime.%s' has been removed and will be ignored. "
                        "All outputs are now always saved.",
                        key,
                    )
                    del data["runtime"][key]

        # Check for old-style keys and migrate them
        migrations = {
            # Preprocessing
            "color_norm": "preprocessing",
            "frame_sampling": "preprocessing",
            # Sparse matching
            "feature_extraction": "sparse_matching",
            "pair_selection": "sparse_matching",
            "matching": "sparse_matching",
            # Reconstruction
            "dense_stereo": "reconstruction",
            "fusion": "reconstruction",
            "surface": "reconstruction",
            "outlier_removal": "reconstruction",
            # Runtime
            "device": "runtime",
            "output": "runtime",
            "visualization": "runtime",
            "benchmark": "runtime",
            "evaluation": "runtime",
        }

        migrated = data.copy()

        for old_key, new_section in migrations.items():
            if old_key in migrated:
                logger.info(
                    "Migrating legacy config key '%s' to new structure", old_key
                )

                # Get or create the target section
                if new_section not in migrated:
                    migrated[new_section] = {}

                # Merge the old section into the new section
                old_data = migrated.pop(old_key)
                if isinstance(old_data, dict):
                    # Handle field name mappings
                    if old_key == "color_norm":
                        # Map color_norm.enabled -> preprocessing.color_norm_enabled
                        if "enabled" in old_data:
                            migrated[new_section]["color_norm_enabled"] = old_data[
                                "enabled"
                            ]
                        if "method" in old_data:
                            migrated[new_section]["color_norm_method"] = old_data[
                                "method"
                            ]
                    elif old_key == "frame_sampling":
                        # Map frame_sampling fields directly
                        migrated[new_section].update(
                            {f"frame_{k}": v for k, v in old_data.items()}
                        )
                    elif old_key == "outlier_removal":
                        # Map outlier_removal fields with prefix
                        migrated[new_section]["outlier_removal_enabled"] = old_data.get(
                            "enabled", True
                        )
                        migrated[new_section]["outlier_nb_neighbors"] = old_data.get(
                            "nb_neighbors", 20
                        )
                        migrated[new_section]["outlier_std_ratio"] = old_data.get(
                            "std_ratio", 2.0
                        )
                    elif old_key == "visualization":
                        # Map visualization fields with prefix
                        if "enabled" in old_data:
                            migrated[new_section]["viz_enabled"] = old_data["enabled"]
                        if "stages" in old_data:
                            migrated[new_section]["viz_stages"] = old_data["stages"]
                    elif old_key == "benchmark":
                        # Map benchmark fields with prefix
                        if "extractors" in old_data:
                            migrated[new_section]["benchmark_extractors"] = old_data[
                                "extractors"
                            ]
                        if "clahe" in old_data:
                            migrated[new_section]["benchmark_clahe"] = old_data["clahe"]
                    elif old_key == "device":
                        # Device is a nested config in old structure
                        if isinstance(old_data, dict) and "device" in old_data:
                            migrated[new_section]["device"] = old_data["device"]
                        else:
                            # Or it might be a direct string
                            migrated[new_section]["device"] = old_data
                    else:
                        # Direct merge for other sections
                        migrated[new_section].update(old_data)

        return migrated

    @staticmethod
    def _log_default_sections(data: dict[str, Any]) -> None:
        """Log INFO messages about sections using defaults.

        Args:
            data: Configuration dictionary.
        """
        sections = [
            "preprocessing",
            "sparse_matching",
            "dense_matching",
            "reconstruction",
            "runtime",
        ]

        for section in sections:
            if section not in data:
                logger.info("Using default: %s (all defaults)", section)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file.

        All fields including defaults are written for explicitness.

        Args:
            path: Path to output YAML file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict using Pydantic v2 model_dump
        data = self.model_dump(mode="json")

        with open(path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    def validate(self) -> None:
        """Validate configuration values.

        This method exists for backward compatibility. Pydantic now validates
        automatically during construction.

        Raises:
            ValidationError: If any configuration value is invalid.
        """
        # Pydantic validates on construction, so just trigger re-validation
        self.model_validate(self.model_dump())


def format_validation_errors(error: ValidationError) -> str:
    """Format Pydantic validation errors with YAML-style paths.

    Args:
        error: Pydantic ValidationError.

    Returns:
        Formatted error string with YAML paths and messages.
    """
    lines = []
    for err in error.errors():
        # Build YAML path from location tuple
        loc = err["loc"]
        path_parts = []
        for part in loc:
            if isinstance(part, int):
                # Array index
                path_parts[-1] = f"{path_parts[-1]}[{part}]"
            else:
                # Field name
                path_parts.append(str(part))

        path = ".".join(path_parts)
        msg = err["msg"]
        lines.append(f"  {path}: {msg}")

    return "\n".join(lines)


# Backward-compatible aliases (deprecated, will be removed in v0.3)
ColorNormConfig = PreprocessingConfig  # Partial — users should migrate
FrameSamplingConfig = PreprocessingConfig
FeatureExtractionConfig = SparseMatchingConfig
PairSelectionConfig = SparseMatchingConfig
MatchingConfig = SparseMatchingConfig
DenseStereoConfig = ReconstructionConfig
FusionConfig = ReconstructionConfig
SurfaceConfig = ReconstructionConfig
OutlierRemovalConfig = ReconstructionConfig
DeviceConfig = RuntimeConfig
OutputConfig = RuntimeConfig
VizConfig = RuntimeConfig
BenchmarkConfig = RuntimeConfig
EvaluationConfig = RuntimeConfig
