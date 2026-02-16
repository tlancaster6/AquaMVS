"""Tests for configuration system."""

import logging

import pytest
from pydantic import ValidationError

from aquamvs.config import (
    BenchmarkConfig,
    ColorNormConfig,
    DenseMatchingConfig,
    DenseStereoConfig,
    DeviceConfig,
    EvaluationConfig,
    FeatureExtractionConfig,
    FrameSamplingConfig,
    FusionConfig,
    MatchingConfig,
    PairSelectionConfig,
    PipelineConfig,
    PreprocessingConfig,
    ReconstructionConfig,
    RuntimeConfig,
    SparseMatchingConfig,
    SurfaceConfig,
)


class TestPreprocessingConfig:
    """Tests for PreprocessingConfig."""

    def test_defaults(self):
        """Test default values."""
        config = PreprocessingConfig()
        # Color norm
        assert config.color_norm_enabled is False
        assert config.color_norm_method == "gain"
        # Frame sampling
        assert config.frame_start == 0
        assert config.frame_stop is None
        assert config.frame_step == 1

    def test_custom_values(self):
        """Test custom values."""
        config = PreprocessingConfig(
            color_norm_enabled=True,
            color_norm_method="histogram",
            frame_start=100,
            frame_stop=500,
            frame_step=10,
        )
        assert config.color_norm_enabled is True
        assert config.color_norm_method == "histogram"
        assert config.frame_start == 100
        assert config.frame_stop == 500
        assert config.frame_step == 10

    def test_invalid_color_norm_method_raises(self):
        """Test that invalid color_norm_method raises ValidationError."""
        with pytest.raises(
            ValidationError, match="Input should be 'gain' or 'histogram'"
        ):
            PreprocessingConfig(color_norm_method="invalid")


class TestSparseMatchingConfig:
    """Tests for SparseMatchingConfig."""

    def test_defaults(self):
        """Test default values."""
        config = SparseMatchingConfig()
        # Feature extraction
        assert config.extractor_type == "superpoint"
        assert config.max_keypoints == 2048
        assert config.detection_threshold == 0.005
        assert config.clahe_enabled is False
        assert config.clahe_clip_limit == 2.0
        # Pair selection
        assert config.num_neighbors == 4
        assert config.include_center is True
        # Matching
        assert config.filter_threshold == 0.1

    def test_custom_values(self):
        """Test custom values."""
        config = SparseMatchingConfig(
            extractor_type="aliked",
            max_keypoints=1024,
            detection_threshold=0.01,
            clahe_enabled=True,
            clahe_clip_limit=4.0,
            num_neighbors=6,
            include_center=False,
            filter_threshold=0.2,
        )
        assert config.extractor_type == "aliked"
        assert config.max_keypoints == 1024
        assert config.detection_threshold == 0.01
        assert config.clahe_enabled is True
        assert config.clahe_clip_limit == 4.0
        assert config.num_neighbors == 6
        assert config.include_center is False
        assert config.filter_threshold == 0.2

    def test_invalid_extractor_type_raises(self):
        """Test that invalid extractor_type raises ValidationError."""
        with pytest.raises(ValidationError, match="Input should be"):
            SparseMatchingConfig(extractor_type="invalid")


class TestDenseMatchingConfig:
    """Tests for DenseMatchingConfig."""

    def test_defaults(self):
        """Test default values."""
        config = DenseMatchingConfig()
        assert config.certainty_threshold == 0.5
        assert config.max_correspondences == 100000

    def test_custom_values(self):
        """Test custom values."""
        config = DenseMatchingConfig(certainty_threshold=0.7, max_correspondences=5000)
        assert config.certainty_threshold == 0.7
        assert config.max_correspondences == 5000


class TestReconstructionConfig:
    """Tests for ReconstructionConfig."""

    def test_defaults(self):
        """Test default values."""
        config = ReconstructionConfig()
        # Dense stereo
        assert config.num_depths == 128
        assert config.cost_function == "ncc"
        assert config.window_size == 11
        assert config.depth_margin == 0.05
        # Fusion
        assert config.min_consistent_views == 3
        assert config.depth_tolerance == 0.005
        assert config.roma_depth_tolerance == 0.02
        assert config.voxel_size == 0.001
        assert config.min_confidence == 0.1
        # Surface
        assert config.surface_method == "poisson"
        assert config.poisson_depth == 9
        assert config.grid_resolution == 0.002
        assert config.bpa_radii is None
        assert config.target_faces is None
        # Outlier removal
        assert config.outlier_removal_enabled is True
        assert config.outlier_nb_neighbors == 20
        assert config.outlier_std_ratio == 2.0

    def test_custom_values(self):
        """Test custom values."""
        config = ReconstructionConfig(
            num_depths=256,
            cost_function="ssim",
            window_size=7,
            depth_margin=0.1,
            min_consistent_views=4,
            depth_tolerance=0.01,
            voxel_size=0.002,
            min_confidence=0.7,
            surface_method="heightfield",
            poisson_depth=10,
            grid_resolution=0.001,
            outlier_removal_enabled=False,
            outlier_nb_neighbors=10,
            outlier_std_ratio=3.0,
        )
        assert config.num_depths == 256
        assert config.cost_function == "ssim"
        assert config.window_size == 7
        assert config.depth_margin == 0.1
        assert config.min_consistent_views == 4
        assert config.depth_tolerance == 0.01
        assert config.voxel_size == 0.002
        assert config.min_confidence == 0.7
        assert config.surface_method == "heightfield"
        assert config.poisson_depth == 10
        assert config.grid_resolution == 0.001
        assert config.outlier_removal_enabled is False
        assert config.outlier_nb_neighbors == 10
        assert config.outlier_std_ratio == 3.0

    def test_invalid_window_size_even(self):
        """Test that even window_size raises ValidationError."""
        with pytest.raises(
            ValidationError, match="window_size must be positive and odd"
        ):
            ReconstructionConfig(window_size=10)

    def test_invalid_window_size_negative(self):
        """Test that negative window_size raises ValidationError."""
        with pytest.raises(
            ValidationError, match="window_size must be positive and odd"
        ):
            ReconstructionConfig(window_size=-5)

    def test_invalid_surface_method_raises(self):
        """Test that invalid surface_method raises ValidationError."""
        with pytest.raises(ValidationError, match="Input should be"):
            ReconstructionConfig(surface_method="invalid")

    def test_invalid_cost_function_raises(self):
        """Test that invalid cost_function raises ValidationError."""
        with pytest.raises(ValidationError, match="Input should be"):
            ReconstructionConfig(cost_function="invalid")


class TestRuntimeConfig:
    """Tests for RuntimeConfig."""

    def test_defaults(self):
        """Test default values."""
        config = RuntimeConfig()
        # Device
        assert config.device == "cpu"
        # Output
        assert config.save_features is False
        assert config.save_depth_maps is True
        assert config.save_point_cloud is True
        assert config.save_mesh is True
        assert config.keep_intermediates is True
        assert config.save_consistency_maps is False
        # Visualization
        assert config.viz_enabled is False
        assert config.viz_stages == []
        # Benchmark
        assert config.benchmark_extractors == ["superpoint", "aliked", "disk"]
        assert config.benchmark_clahe == [True, False]
        # Evaluation
        assert config.icp_max_distance == 0.01
        # Progress
        assert config.quiet is False

    def test_custom_values(self):
        """Test custom values."""
        config = RuntimeConfig(
            device="cuda",
            save_features=True,
            save_depth_maps=False,
            viz_enabled=True,
            viz_stages=["depth", "scene"],
            benchmark_extractors=["superpoint"],
            benchmark_clahe=[False],
            icp_max_distance=0.02,
            quiet=True,
        )
        assert config.device == "cuda"
        assert config.save_features is True
        assert config.save_depth_maps is False
        assert config.viz_enabled is True
        assert config.viz_stages == ["depth", "scene"]
        assert config.benchmark_extractors == ["superpoint"]
        assert config.benchmark_clahe == [False]
        assert config.icp_max_distance == 0.02
        assert config.quiet is True

    def test_invalid_device_raises(self):
        """Test that invalid device raises ValidationError."""
        with pytest.raises(ValidationError, match="Input should be"):
            RuntimeConfig(device="invalid")

    def test_invalid_viz_stage_raises(self):
        """Test that invalid viz_stage raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid visualization stage"):
            RuntimeConfig(viz_stages=["invalid"])

    def test_invalid_benchmark_extractor_raises(self):
        """Test that invalid benchmark_extractor raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid benchmark extractor"):
            RuntimeConfig(benchmark_extractors=["invalid"])

    def test_valid_viz_stages(self):
        """Test that valid viz_stages don't raise."""
        config = RuntimeConfig(
            viz_stages=["depth", "features", "scene", "rig", "summary"]
        )
        assert config.viz_stages == ["depth", "features", "scene", "rig", "summary"]

    def test_quiet_default(self):
        """Test that quiet defaults to False."""
        config = RuntimeConfig()
        assert config.quiet is False


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_defaults(self):
        """Test default values."""
        config = PipelineConfig()
        assert config.calibration_path == ""
        assert config.output_dir == ""
        assert config.camera_input_map == {}
        assert config.mask_dir is None
        assert config.pipeline_mode == "full"
        assert config.matcher_type == "lightglue"
        # Sub-configs should be initialized
        assert isinstance(config.preprocessing, PreprocessingConfig)
        assert isinstance(config.sparse_matching, SparseMatchingConfig)
        assert isinstance(config.dense_matching, DenseMatchingConfig)
        assert isinstance(config.reconstruction, ReconstructionConfig)
        assert isinstance(config.runtime, RuntimeConfig)

    def test_custom_values(self):
        """Test custom values."""
        config = PipelineConfig(
            calibration_path="/path/to/calibration.json",
            output_dir="/path/to/output",
            camera_input_map={"cam1": "/video1.mp4"},
            mask_dir="/path/to/masks",
            pipeline_mode="sparse",
            matcher_type="roma",
        )
        assert config.calibration_path == "/path/to/calibration.json"
        assert config.output_dir == "/path/to/output"
        assert config.camera_input_map == {"cam1": "/video1.mp4"}
        assert config.mask_dir == "/path/to/masks"
        assert config.pipeline_mode == "sparse"
        assert config.matcher_type == "roma"

    def test_invalid_matcher_type_raises(self):
        """Test that invalid matcher_type raises ValidationError."""
        with pytest.raises(ValidationError, match="Input should be"):
            PipelineConfig(matcher_type="invalid")

    def test_invalid_pipeline_mode_raises(self):
        """Test that invalid pipeline_mode raises ValidationError."""
        with pytest.raises(ValidationError, match="Input should be"):
            PipelineConfig(pipeline_mode="invalid")

    def test_sub_config_defaults(self):
        """Test that all sub-configs have correct defaults."""
        config = PipelineConfig()
        assert config.preprocessing.color_norm_enabled is False
        assert config.sparse_matching.extractor_type == "superpoint"
        assert config.dense_matching.certainty_threshold == 0.5
        assert config.reconstruction.num_depths == 128
        assert config.runtime.device == "cpu"


class TestValidationErrorCollection:
    """Tests for validation error collection."""

    def test_multiple_errors_collected(self):
        """Test that multiple validation errors are collected."""
        with pytest.raises(ValidationError) as exc_info:
            PipelineConfig(
                matcher_type="invalid",
                pipeline_mode="invalid",
                preprocessing={"color_norm_method": "invalid"},
            )

        error = exc_info.value
        errors = error.errors()
        # Should have at least 3 errors
        assert len(errors) >= 3

        # Check that all error locations are captured
        locs = [err["loc"] for err in errors]
        assert ("matcher_type",) in locs
        assert ("pipeline_mode",) in locs
        assert ("preprocessing", "color_norm_method") in locs

    def test_error_messages_contain_yaml_paths(self):
        """Test that error messages contain YAML-style paths."""
        from aquamvs.config import format_validation_errors

        with pytest.raises(ValidationError) as exc_info:
            ReconstructionConfig(
                window_size=10, cost_function="invalid", surface_method="invalid"
            )

        formatted = format_validation_errors(exc_info.value)
        # Should contain dot-separated paths
        assert "window_size:" in formatted or "cost_function:" in formatted


class TestExtraFieldWarning:
    """Tests for unknown field warnings."""

    def test_extra_fields_produce_warning(self, caplog):
        """Test that unknown keys produce a warning."""
        with caplog.at_level(logging.WARNING):
            PreprocessingConfig(unknown_field="value")

        # Check that warning was logged
        assert any("Unknown config keys" in record.message for record in caplog.records)
        assert any("unknown_field" in record.message for record in caplog.records)

    def test_extra_fields_in_pipeline_config(self, caplog):
        """Test that unknown keys in PipelineConfig produce a warning."""
        with caplog.at_level(logging.WARNING):
            PipelineConfig(unknown_top_level_field="value")

        assert any("Unknown config keys" in record.message for record in caplog.records)


class TestYAMLRoundTrip:
    """Tests for YAML serialization and deserialization."""

    def test_round_trip_full_config(self, tmp_path):
        """Test that save and load preserves all values."""
        config_path = tmp_path / "config.yaml"

        # Create a config with custom values
        original = PipelineConfig(
            calibration_path="/path/to/calibration.json",
            output_dir="/path/to/output",
            camera_input_map={"cam1": "/video1.mp4", "cam2": "/video2.mp4"},
            pipeline_mode="sparse",
            matcher_type="roma",
            preprocessing=PreprocessingConfig(
                color_norm_enabled=True,
                color_norm_method="histogram",
                frame_start=100,
                frame_stop=500,
                frame_step=10,
            ),
            sparse_matching=SparseMatchingConfig(
                extractor_type="aliked", max_keypoints=1024
            ),
            reconstruction=ReconstructionConfig(num_depths=256, cost_function="ssim"),
            runtime=RuntimeConfig(device="cuda", viz_enabled=True),
        )

        # Save and load
        original.to_yaml(config_path)
        loaded = PipelineConfig.from_yaml(config_path)

        # Check all fields match
        assert loaded.calibration_path == original.calibration_path
        assert loaded.output_dir == original.output_dir
        assert loaded.camera_input_map == original.camera_input_map
        assert loaded.pipeline_mode == original.pipeline_mode
        assert loaded.matcher_type == original.matcher_type
        assert (
            loaded.preprocessing.color_norm_enabled
            == original.preprocessing.color_norm_enabled
        )
        assert (
            loaded.sparse_matching.extractor_type
            == original.sparse_matching.extractor_type
        )
        assert loaded.reconstruction.num_depths == original.reconstruction.num_depths
        assert loaded.runtime.device == original.runtime.device

    def test_round_trip_defaults_only(self, tmp_path):
        """Test round-trip with only session fields set."""
        config_path = tmp_path / "config.yaml"

        original = PipelineConfig(
            calibration_path="/path/to/calibration.json",
            output_dir="/path/to/output",
            camera_input_map={"cam1": "/video1.mp4"},
        )

        original.to_yaml(config_path)
        loaded = PipelineConfig.from_yaml(config_path)

        assert loaded.calibration_path == original.calibration_path
        assert loaded.output_dir == original.output_dir
        assert loaded.camera_input_map == original.camera_input_map
        assert loaded.preprocessing.frame_start == 0

    def test_partial_yaml_merges_over_defaults(self, tmp_path):
        """Test that loading partial YAML merges over defaults."""
        config_path = tmp_path / "config.yaml"

        yaml_content = """
calibration_path: /path/to/calibration.json
output_dir: /path/to/output
camera_input_map:
  cam1: /video1.mp4

reconstruction:
  num_depths: 256
  cost_function: ssim

runtime:
  device: cuda
"""

        config_path.write_text(yaml_content)
        loaded = PipelineConfig.from_yaml(config_path)

        # Session fields
        assert loaded.calibration_path == "/path/to/calibration.json"
        assert loaded.output_dir == "/path/to/output"
        assert loaded.camera_input_map == {"cam1": "/video1.mp4"}

        # Partially specified reconstruction (others should be default)
        assert loaded.reconstruction.num_depths == 256
        assert loaded.reconstruction.cost_function == "ssim"
        assert loaded.reconstruction.window_size == 11  # default
        assert loaded.reconstruction.depth_margin == 0.05  # default

        # Specified device
        assert loaded.runtime.device == "cuda"

        # Unspecified configs should be all defaults
        assert loaded.preprocessing.frame_start == 0
        assert loaded.sparse_matching.extractor_type == "superpoint"

    def test_empty_yaml_loads_as_defaults(self, tmp_path):
        """Test that loading an empty YAML file gives all defaults."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("")

        loaded = PipelineConfig.from_yaml(config_path)
        assert loaded.calibration_path == ""
        assert loaded.output_dir == ""
        assert loaded.camera_input_map == {}
        assert loaded.reconstruction.num_depths == 128

    def test_yaml_handles_none_values(self, tmp_path):
        """Test that None values are correctly serialized/deserialized."""
        config_path = tmp_path / "config.yaml"

        config = PipelineConfig(
            calibration_path="/path/to/calibration.json",
            output_dir="/path/to/output",
            camera_input_map={"cam1": "/video1.mp4"},
            preprocessing=PreprocessingConfig(frame_stop=None),
        )

        config.to_yaml(config_path)

        # Load and verify
        loaded = PipelineConfig.from_yaml(config_path)
        assert loaded.preprocessing.frame_stop is None


class TestBackwardCompatibility:
    """Tests for backward compatibility with old YAML structure."""

    def test_old_flat_structure_loads(self, tmp_path, caplog):
        """Test that old flat YAML structure still loads."""
        config_path = tmp_path / "config.yaml"

        yaml_content = """
calibration_path: /path/to/calibration.json
output_dir: /path/to/output
camera_input_map:
  cam1: /video1.mp4

color_norm:
  enabled: true
  method: histogram

frame_sampling:
  start: 100
  stop: 500
  step: 10

feature_extraction:
  extractor_type: aliked
  max_keypoints: 1024

dense_stereo:
  num_depths: 256
  cost_function: ssim

device:
  device: cuda
"""

        config_path.write_text(yaml_content)

        with caplog.at_level(logging.INFO):
            loaded = PipelineConfig.from_yaml(config_path)

        # Check that migration messages were logged
        assert any(
            "Migrating legacy config key" in record.message for record in caplog.records
        )

        # Check that values were migrated correctly
        assert loaded.preprocessing.color_norm_enabled is True
        assert loaded.preprocessing.color_norm_method == "histogram"
        assert loaded.preprocessing.frame_start == 100
        assert loaded.preprocessing.frame_stop == 500
        assert loaded.preprocessing.frame_step == 10
        assert loaded.sparse_matching.extractor_type == "aliked"
        assert loaded.sparse_matching.max_keypoints == 1024
        assert loaded.reconstruction.num_depths == 256
        assert loaded.reconstruction.cost_function == "ssim"
        assert loaded.runtime.device == "cuda"

    def test_new_nested_structure_loads(self, tmp_path):
        """Test that new nested YAML structure loads."""
        config_path = tmp_path / "config.yaml"

        yaml_content = """
calibration_path: /path/to/calibration.json
output_dir: /path/to/output
camera_input_map:
  cam1: /video1.mp4

preprocessing:
  color_norm_enabled: true
  color_norm_method: histogram
  frame_start: 100

sparse_matching:
  extractor_type: aliked

reconstruction:
  num_depths: 256

runtime:
  device: cuda
"""

        config_path.write_text(yaml_content)
        loaded = PipelineConfig.from_yaml(config_path)

        assert loaded.preprocessing.color_norm_enabled is True
        assert loaded.preprocessing.frame_start == 100
        assert loaded.sparse_matching.extractor_type == "aliked"
        assert loaded.reconstruction.num_depths == 256
        assert loaded.runtime.device == "cuda"

    def test_old_class_imports_still_resolve(self):
        """Test that old class name imports still work."""
        # These should all resolve to the new classes
        assert DenseStereoConfig == ReconstructionConfig
        assert FusionConfig == ReconstructionConfig
        assert SurfaceConfig == ReconstructionConfig
        assert FeatureExtractionConfig == SparseMatchingConfig
        assert PairSelectionConfig == SparseMatchingConfig
        assert MatchingConfig == SparseMatchingConfig
        assert ColorNormConfig == PreprocessingConfig
        assert FrameSamplingConfig == PreprocessingConfig
        assert DeviceConfig == RuntimeConfig
        assert BenchmarkConfig == RuntimeConfig
        assert EvaluationConfig == RuntimeConfig


class TestDefaultLogging:
    """Tests for default section logging."""

    def test_missing_sections_logged(self, tmp_path, caplog):
        """Test that missing sections are logged."""
        config_path = tmp_path / "config.yaml"

        yaml_content = """
calibration_path: /path/to/calibration.json
output_dir: /path/to/output
camera_input_map:
  cam1: /video1.mp4
"""

        config_path.write_text(yaml_content)

        with caplog.at_level(logging.INFO):
            PipelineConfig.from_yaml(config_path)

        # Check that default logging happened
        assert any(
            "Using default: preprocessing" in record.message
            for record in caplog.records
        )
        assert any(
            "Using default: sparse_matching" in record.message
            for record in caplog.records
        )


class TestImports:
    """Test that config classes can be imported."""

    def test_import_new_config_classes(self):
        """Test that new config classes can be imported from aquamvs.config."""
        from aquamvs.config import (
            DenseMatchingConfig,
            PipelineConfig,
            PreprocessingConfig,
            ReconstructionConfig,
            RuntimeConfig,
            SparseMatchingConfig,
        )

        assert PreprocessingConfig is not None
        assert SparseMatchingConfig is not None
        assert DenseMatchingConfig is not None
        assert ReconstructionConfig is not None
        assert RuntimeConfig is not None
        assert PipelineConfig is not None

    def test_import_from_package(self):
        """Test that configs can be imported from the aquamvs package."""
        from aquamvs import PipelineConfig as PipelineConfig2
        from aquamvs import PreprocessingConfig as PreprocessingConfig2

        assert PipelineConfig2 is not None
        assert PreprocessingConfig2 is not None

    def test_old_aliases_import(self):
        """Test that old aliases still import."""
        from aquamvs.config import (
            BenchmarkConfig,
            ColorNormConfig,
            DenseStereoConfig,
            DeviceConfig,
            EvaluationConfig,
            FeatureExtractionConfig,
            FrameSamplingConfig,
            FusionConfig,
            MatchingConfig,
            PairSelectionConfig,
            SurfaceConfig,
        )

        # These should all exist (even if they're aliases)
        assert ColorNormConfig is not None
        assert FrameSamplingConfig is not None
        assert FeatureExtractionConfig is not None
        assert PairSelectionConfig is not None
        assert MatchingConfig is not None
        assert DenseStereoConfig is not None
        assert FusionConfig is not None
        assert SurfaceConfig is not None
        assert DeviceConfig is not None
        assert BenchmarkConfig is not None
        assert EvaluationConfig is not None


class TestQualityPresets:
    """Tests for quality preset system."""

    def test_quality_preset_enum_values(self):
        """Test that QualityPreset enum has expected values."""
        from aquamvs.config import QualityPreset

        assert QualityPreset.FAST == "fast"
        assert QualityPreset.BALANCED == "balanced"
        assert QualityPreset.QUALITY == "quality"

    def test_fast_preset_sets_expected_values(self):
        """Test that fast preset sets expected parameter values."""
        from aquamvs.config import PipelineConfig, QualityPreset

        config = PipelineConfig(quality_preset=QualityPreset.FAST)

        # Dense stereo
        assert config.reconstruction.num_depths == 64
        assert config.reconstruction.window_size == 7
        assert config.reconstruction.depth_batch_size == 8

        # Sparse matching
        assert config.sparse_matching.max_keypoints == 1024

        # Fusion
        assert config.reconstruction.voxel_size == 0.002

        # Surface
        assert config.reconstruction.poisson_depth == 8

    def test_balanced_preset_sets_expected_values(self):
        """Test that balanced preset sets expected parameter values."""
        from aquamvs.config import PipelineConfig, QualityPreset

        config = PipelineConfig(quality_preset=QualityPreset.BALANCED)

        # Dense stereo
        assert config.reconstruction.num_depths == 128
        assert config.reconstruction.window_size == 11
        assert config.reconstruction.depth_batch_size == 4

        # Sparse matching
        assert config.sparse_matching.max_keypoints == 2048

        # Fusion
        assert config.reconstruction.voxel_size == 0.001

        # Surface
        assert config.reconstruction.poisson_depth == 9

    def test_quality_preset_sets_expected_values(self):
        """Test that quality preset sets expected parameter values."""
        from aquamvs.config import PipelineConfig, QualityPreset

        config = PipelineConfig(quality_preset=QualityPreset.QUALITY)

        # Dense stereo
        assert config.reconstruction.num_depths == 256
        assert config.reconstruction.window_size == 15
        assert config.reconstruction.depth_batch_size == 1

        # Sparse matching
        assert config.sparse_matching.max_keypoints == 4096

        # Fusion
        assert config.reconstruction.voxel_size == 0.0005

        # Surface
        assert config.reconstruction.poisson_depth == 10

    def test_explicit_values_override_preset(self):
        """Test that explicitly set values are NOT overridden by preset."""
        from aquamvs.config import (
            PipelineConfig,
            QualityPreset,
            ReconstructionConfig,
            SparseMatchingConfig,
        )

        config = PipelineConfig(
            quality_preset=QualityPreset.FAST,
            reconstruction=ReconstructionConfig(num_depths=512),
            sparse_matching=SparseMatchingConfig(max_keypoints=8192),
        )

        # Explicit values should be preserved
        assert config.reconstruction.num_depths == 512
        assert config.sparse_matching.max_keypoints == 8192

        # Other preset values should still apply
        assert config.reconstruction.window_size == 7
        assert config.reconstruction.depth_batch_size == 8

    def test_partial_override_preset(self):
        """Test that partially overridden configs preserve user values."""
        from aquamvs.config import PipelineConfig, QualityPreset, ReconstructionConfig

        config = PipelineConfig(
            quality_preset=QualityPreset.QUALITY,
            reconstruction=ReconstructionConfig(num_depths=200),
        )

        # User-specified value preserved
        assert config.reconstruction.num_depths == 200

        # Other preset values applied
        assert config.reconstruction.window_size == 15
        assert config.reconstruction.voxel_size == 0.0005
        assert config.sparse_matching.max_keypoints == 4096

    def test_preset_round_trip_yaml(self, tmp_path):
        """Test that preset round-trips correctly through YAML."""
        from aquamvs.config import PipelineConfig, QualityPreset

        config_path = tmp_path / "config.yaml"

        original = PipelineConfig(
            calibration_path="/path/to/calibration.json",
            output_dir="/path/to/output",
            camera_input_map={"cam1": "/video1.mp4"},
            quality_preset=QualityPreset.FAST,
        )

        # Save and load
        original.to_yaml(config_path)
        loaded = PipelineConfig.from_yaml(config_path)

        # Preset should be preserved
        assert loaded.quality_preset == QualityPreset.FAST

        # Preset values should be applied
        assert loaded.reconstruction.num_depths == 64
        assert loaded.reconstruction.window_size == 7
        assert loaded.sparse_matching.max_keypoints == 1024

    def test_preset_from_string(self):
        """Test that preset can be specified as string in construction."""
        from aquamvs.config import PipelineConfig, QualityPreset

        config = PipelineConfig(quality_preset="balanced")

        assert config.quality_preset == QualityPreset.BALANCED
        assert config.reconstruction.num_depths == 128

    def test_no_preset_uses_defaults(self):
        """Test that no preset uses default values."""
        from aquamvs.config import PipelineConfig

        config = PipelineConfig()

        assert config.quality_preset is None
        assert config.reconstruction.num_depths == 128  # Default value
        assert config.reconstruction.window_size == 11  # Default value
        assert config.reconstruction.depth_batch_size == 4  # Default value
        assert config.sparse_matching.max_keypoints == 2048  # Default value
