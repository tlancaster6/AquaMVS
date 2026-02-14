"""Tests for configuration system."""

import tempfile
from pathlib import Path

import pytest
import yaml

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
    OutputConfig,
    PairSelectionConfig,
    PipelineConfig,
    SurfaceConfig,
    VizConfig,
)


class TestColorNormConfig:
    """Tests for ColorNormConfig."""

    def test_defaults(self):
        """Test default values."""
        config = ColorNormConfig()
        assert config.enabled is False
        assert config.method == "gain"

    def test_custom_values(self):
        """Test custom values."""
        config = ColorNormConfig(enabled=True, method="histogram")
        assert config.enabled is True
        assert config.method == "histogram"

    def test_invalid_method_raises(self):
        """Test that invalid method is caught during validation."""
        config = ColorNormConfig(method="invalid")
        # Method validation happens in PipelineConfig.validate()
        pipeline = PipelineConfig()
        pipeline.color_norm = config
        with pytest.raises(ValueError, match="Invalid color_norm method"):
            pipeline.validate()


class TestFrameSamplingConfig:
    """Tests for FrameSamplingConfig."""

    def test_defaults(self):
        """Test default values."""
        config = FrameSamplingConfig()
        assert config.start == 0
        assert config.stop is None
        assert config.step == 1

    def test_custom_values(self):
        """Test custom values."""
        config = FrameSamplingConfig(start=100, stop=500, step=10)
        assert config.start == 100
        assert config.stop == 500
        assert config.step == 10


class TestFeatureExtractionConfig:
    """Tests for FeatureExtractionConfig."""

    def test_defaults(self):
        """Test default values."""
        config = FeatureExtractionConfig()
        assert config.extractor_type == "superpoint"
        assert config.max_keypoints == 2048
        assert config.detection_threshold == 0.005

    def test_custom_values(self):
        """Test custom values."""
        config = FeatureExtractionConfig(
            extractor_type="aliked", max_keypoints=1024, detection_threshold=0.01
        )
        assert config.extractor_type == "aliked"
        assert config.max_keypoints == 1024
        assert config.detection_threshold == 0.01

    def test_extractor_type_default(self):
        """Test that extractor_type defaults to superpoint."""
        config = FeatureExtractionConfig()
        assert config.extractor_type == "superpoint"

    def test_clahe_defaults(self):
        """Test that CLAHE defaults are correct."""
        config = FeatureExtractionConfig()
        assert config.clahe_enabled is False
        assert config.clahe_clip_limit == 2.0


class TestPairSelectionConfig:
    """Tests for PairSelectionConfig."""

    def test_defaults(self):
        """Test default values."""
        config = PairSelectionConfig()
        assert config.num_neighbors == 4
        assert config.include_center is True

    def test_custom_values(self):
        """Test custom values."""
        config = PairSelectionConfig(num_neighbors=6, include_center=False)
        assert config.num_neighbors == 6
        assert config.include_center is False


class TestMatchingConfig:
    """Tests for MatchingConfig."""

    def test_defaults(self):
        """Test default values."""
        config = MatchingConfig()
        assert config.filter_threshold == 0.1

    def test_custom_values(self):
        """Test custom values."""
        config = MatchingConfig(filter_threshold=0.2)
        assert config.filter_threshold == 0.2


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


class TestDenseStereoConfig:
    """Tests for DenseStereoConfig."""

    def test_defaults(self):
        """Test default values."""
        config = DenseStereoConfig()
        assert config.num_depths == 128
        assert config.cost_function == "ncc"
        assert config.window_size == 11
        assert config.depth_margin == 0.05

    def test_custom_values(self):
        """Test custom values."""
        config = DenseStereoConfig(
            num_depths=256, cost_function="ssim", window_size=7, depth_margin=0.1
        )
        assert config.num_depths == 256
        assert config.cost_function == "ssim"
        assert config.window_size == 7
        assert config.depth_margin == 0.1


class TestFusionConfig:
    """Tests for FusionConfig."""

    def test_defaults(self):
        """Test default values."""
        config = FusionConfig()
        assert config.min_consistent_views == 3
        assert config.depth_tolerance == 0.005
        assert config.voxel_size == 0.001
        assert config.min_confidence == 0.1

    def test_custom_values(self):
        """Test custom values."""
        config = FusionConfig(
            min_consistent_views=4,
            depth_tolerance=0.01,
            voxel_size=0.002,
            min_confidence=0.7,
        )
        assert config.min_consistent_views == 4
        assert config.depth_tolerance == 0.01
        assert config.voxel_size == 0.002
        assert config.min_confidence == 0.7


class TestSurfaceConfig:
    """Tests for SurfaceConfig."""

    def test_defaults(self):
        """Test default values."""
        config = SurfaceConfig()
        assert config.method == "poisson"
        assert config.poisson_depth == 9
        assert config.grid_resolution == 0.002

    def test_custom_values(self):
        """Test custom values."""
        config = SurfaceConfig(
            method="heightfield", poisson_depth=10, grid_resolution=0.001
        )
        assert config.method == "heightfield"
        assert config.poisson_depth == 10
        assert config.grid_resolution == 0.001


class TestEvaluationConfig:
    """Tests for EvaluationConfig."""

    def test_defaults(self):
        """Test default values."""
        config = EvaluationConfig()
        assert config.icp_max_distance == 0.01

    def test_custom_values(self):
        """Test custom values."""
        config = EvaluationConfig(icp_max_distance=0.02)
        assert config.icp_max_distance == 0.02


class TestDeviceConfig:
    """Tests for DeviceConfig."""

    def test_defaults(self):
        """Test default values."""
        config = DeviceConfig()
        assert config.device == "cpu"

    def test_custom_values(self):
        """Test custom values."""
        config = DeviceConfig(device="cuda")
        assert config.device == "cuda"


class TestOutputConfig:
    """Tests for OutputConfig."""

    def test_defaults(self):
        """Test default values."""
        config = OutputConfig()
        assert config.save_features is False
        assert config.save_depth_maps is True
        assert config.save_point_cloud is True
        assert config.save_mesh is True
        assert config.keep_intermediates is True

    def test_custom_values(self):
        """Test custom values."""
        config = OutputConfig(
            save_features=True,
            save_depth_maps=False,
            save_point_cloud=False,
            save_mesh=False,
            keep_intermediates=False,
        )
        assert config.save_features is True
        assert config.save_depth_maps is False
        assert config.save_point_cloud is False
        assert config.save_mesh is False
        assert config.keep_intermediates is False


class TestVizConfig:
    """Tests for VizConfig."""

    def test_defaults(self):
        """Test default values."""
        config = VizConfig()
        assert config.enabled is False
        assert config.stages == []

    def test_custom_values(self):
        """Test custom values."""
        config = VizConfig(enabled=True, stages=["depth", "scene"])
        assert config.enabled is True
        assert config.stages == ["depth", "scene"]


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""

    def test_defaults(self):
        """Test default values."""
        config = BenchmarkConfig()
        assert config.extractors == ["superpoint", "aliked", "disk"]
        assert config.clahe == [True, False]

    def test_custom_values(self):
        """Test custom values."""
        config = BenchmarkConfig(
            extractors=["superpoint", "aliked"],
            clahe=[False],
        )
        assert config.extractors == ["superpoint", "aliked"]
        assert config.clahe == [False]


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_defaults_only_session_fields(self):
        """Test that PipelineConfig with only session fields has all defaults."""
        config = PipelineConfig(
            calibration_path="/path/to/calibration.json",
            output_dir="/path/to/output",
            camera_video_map={"cam1": "/path/to/video1.mp4"},
        )

        # Session fields
        assert config.calibration_path == "/path/to/calibration.json"
        assert config.output_dir == "/path/to/output"
        assert config.camera_video_map == {"cam1": "/path/to/video1.mp4"}

        # All stage configs should have defaults
        assert config.frame_sampling == FrameSamplingConfig()
        assert config.feature_extraction == FeatureExtractionConfig()
        assert config.pair_selection == PairSelectionConfig()
        assert config.matching == MatchingConfig()
        assert config.dense_stereo == DenseStereoConfig()
        assert config.fusion == FusionConfig()
        assert config.surface == SurfaceConfig()
        assert config.evaluation == EvaluationConfig()
        assert config.device == DeviceConfig()
        assert config.output == OutputConfig()
        assert config.visualization == VizConfig()

    def test_empty_config(self):
        """Test creating an empty config (all defaults)."""
        config = PipelineConfig()
        assert config.calibration_path == ""
        assert config.output_dir == ""
        assert config.camera_video_map == {}
        assert config.pipeline_mode == "full"

    def test_validation_valid_config(self):
        """Test validation passes for valid config."""
        config = PipelineConfig()
        config.validate()  # Should not raise

    def test_validation_invalid_cost_function(self):
        """Test validation catches invalid cost function."""
        config = PipelineConfig()
        config.dense_stereo.cost_function = "invalid"
        with pytest.raises(ValueError, match="Invalid cost_function"):
            config.validate()

    def test_validation_invalid_surface_method(self):
        """Test validation catches invalid surface method."""
        config = PipelineConfig()
        config.surface.method = "invalid"
        with pytest.raises(ValueError, match="Invalid surface method"):
            config.validate()

    def test_validation_invalid_window_size_even(self):
        """Test validation catches even window size."""
        config = PipelineConfig()
        config.dense_stereo.window_size = 10
        with pytest.raises(ValueError, match="Invalid window_size"):
            config.validate()

    def test_validation_invalid_window_size_negative(self):
        """Test validation catches negative window size."""
        config = PipelineConfig()
        config.dense_stereo.window_size = -5
        with pytest.raises(ValueError, match="Invalid window_size"):
            config.validate()

    def test_validation_invalid_device(self):
        """Test validation catches invalid device."""
        config = PipelineConfig()
        config.device.device = "invalid"
        with pytest.raises(ValueError, match="Invalid device"):
            config.validate()

    def test_validation_invalid_viz_stage(self):
        """Test validation catches invalid visualization stage."""
        config = PipelineConfig()
        config.visualization.stages = ["invalid"]
        with pytest.raises(ValueError, match="Invalid visualization stage"):
            config.validate()

    def test_validation_valid_viz_stages(self):
        """Test validation passes for valid visualization stages."""
        config = PipelineConfig()
        config.visualization.stages = ["depth", "scene"]
        config.validate()  # Should not raise

    def test_validation_valid_pipeline_mode_sparse(self):
        """Test validation passes for pipeline_mode='sparse'."""
        config = PipelineConfig()
        config.pipeline_mode = "sparse"
        config.validate()  # Should not raise

    def test_validation_valid_pipeline_mode_full(self):
        """Test validation passes for pipeline_mode='full'."""
        config = PipelineConfig()
        config.pipeline_mode = "full"
        config.validate()  # Should not raise

    def test_validation_invalid_pipeline_mode(self):
        """Test validation catches invalid pipeline_mode."""
        config = PipelineConfig()
        config.pipeline_mode = "dense"
        with pytest.raises(ValueError, match="Invalid pipeline_mode"):
            config.validate()

    def test_validation_valid_extractor_types(self):
        """Test validation passes for valid extractor types."""
        for extractor_type in ["superpoint", "aliked", "disk"]:
            config = PipelineConfig()
            config.feature_extraction.extractor_type = extractor_type
            config.validate()  # Should not raise

    def test_validation_invalid_extractor_type(self):
        """Test validation catches invalid extractor_type."""
        config = PipelineConfig()
        config.feature_extraction.extractor_type = "invalid"
        with pytest.raises(ValueError, match="Invalid extractor_type"):
            config.validate()

    def test_validation_invalid_benchmark_extractor(self):
        """Test validation catches invalid benchmark extractor."""
        config = PipelineConfig()
        config.benchmark.extractors = ["superpoint", "invalid"]
        with pytest.raises(ValueError, match="Invalid benchmark extractor"):
            config.validate()

    def test_validation_valid_benchmark_extractors(self):
        """Test validation passes for valid benchmark extractors."""
        config = PipelineConfig()
        config.benchmark.extractors = ["superpoint", "aliked", "disk"]
        config.validate()  # Should not raise

    def test_validation_valid_matcher_types(self):
        """Test validation passes for valid matcher types."""
        for matcher_type in ["lightglue", "roma"]:
            config = PipelineConfig()
            config.matcher_type = matcher_type
            config.validate()  # Should not raise

    def test_validation_invalid_matcher_type(self):
        """Test validation catches invalid matcher_type."""
        config = PipelineConfig()
        config.matcher_type = "invalid"
        with pytest.raises(ValueError, match="Invalid matcher_type"):
            config.validate()

    def test_matcher_type_default(self):
        """Test that matcher_type defaults to lightglue."""
        config = PipelineConfig()
        assert config.matcher_type == "lightglue"


class TestYAMLRoundTrip:
    """Tests for YAML serialization and deserialization."""

    def test_round_trip_full_config(self):
        """Test that save and load preserves all values."""
        # Create a config with custom values
        original = PipelineConfig(
            calibration_path="/path/to/calibration.json",
            output_dir="/path/to/output",
            camera_video_map={"cam1": "/video1.mp4", "cam2": "/video2.mp4"},
            frame_sampling=FrameSamplingConfig(start=100, stop=500, step=10),
            feature_extraction=FeatureExtractionConfig(
                max_keypoints=1024, detection_threshold=0.01
            ),
            pair_selection=PairSelectionConfig(num_neighbors=6, include_center=False),
            matching=MatchingConfig(filter_threshold=0.2),
            dense_stereo=DenseStereoConfig(
                num_depths=256, cost_function="ssim", window_size=7, depth_margin=0.1
            ),
            fusion=FusionConfig(
                min_consistent_views=4,
                depth_tolerance=0.01,
                voxel_size=0.002,
                min_confidence=0.7,
            ),
            surface=SurfaceConfig(
                method="heightfield", poisson_depth=10, grid_resolution=0.001
            ),
            evaluation=EvaluationConfig(icp_max_distance=0.02),
            device=DeviceConfig(device="cuda"),
            output=OutputConfig(
                save_features=True,
                save_depth_maps=False,
                save_point_cloud=True,
                save_mesh=False,
                keep_intermediates=False,
            ),
            visualization=VizConfig(enabled=True, stages=["depth", "scene"]),
        )

        # Save and load
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            original.to_yaml(temp_path)
            loaded = PipelineConfig.from_yaml(temp_path)

            # Check all fields match
            assert loaded.calibration_path == original.calibration_path
            assert loaded.output_dir == original.output_dir
            assert loaded.camera_video_map == original.camera_video_map
            assert loaded.frame_sampling == original.frame_sampling
            assert loaded.feature_extraction == original.feature_extraction
            assert loaded.pair_selection == original.pair_selection
            assert loaded.matching == original.matching
            assert loaded.dense_stereo == original.dense_stereo
            assert loaded.fusion == original.fusion
            assert loaded.surface == original.surface
            assert loaded.evaluation == original.evaluation
            assert loaded.device == original.device
            assert loaded.output == original.output
            assert loaded.visualization == original.visualization
        finally:
            temp_path.unlink()

    def test_round_trip_defaults_only(self):
        """Test round-trip with only session fields set."""
        original = PipelineConfig(
            calibration_path="/path/to/calibration.json",
            output_dir="/path/to/output",
            camera_video_map={"cam1": "/video1.mp4"},
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            original.to_yaml(temp_path)
            loaded = PipelineConfig.from_yaml(temp_path)

            # All fields should match
            assert loaded.calibration_path == original.calibration_path
            assert loaded.output_dir == original.output_dir
            assert loaded.camera_video_map == original.camera_video_map
            assert loaded.frame_sampling == original.frame_sampling
            assert loaded.dense_stereo == original.dense_stereo
        finally:
            temp_path.unlink()

    def test_partial_yaml_merges_over_defaults(self):
        """Test that loading partial YAML merges over defaults."""
        yaml_content = """
calibration_path: /path/to/calibration.json
output_dir: /path/to/output
camera_video_map:
  cam1: /video1.mp4

dense_stereo:
  num_depths: 256
  cost_function: ssim

device:
  device: cuda
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            loaded = PipelineConfig.from_yaml(temp_path)

            # Session fields
            assert loaded.calibration_path == "/path/to/calibration.json"
            assert loaded.output_dir == "/path/to/output"
            assert loaded.camera_video_map == {"cam1": "/video1.mp4"}

            # Partially specified dense_stereo (others should be default)
            assert loaded.dense_stereo.num_depths == 256
            assert loaded.dense_stereo.cost_function == "ssim"
            assert loaded.dense_stereo.window_size == 11  # default
            assert loaded.dense_stereo.depth_margin == 0.05  # default

            # Specified device
            assert loaded.device.device == "cuda"

            # Unspecified configs should be all defaults
            assert loaded.frame_sampling == FrameSamplingConfig()
            assert loaded.fusion == FusionConfig()
        finally:
            temp_path.unlink()

    def test_yaml_output_is_human_readable(self):
        """Test that YAML output is clean and readable."""
        config = PipelineConfig(
            calibration_path="/path/to/calibration.json",
            output_dir="/path/to/output",
            camera_video_map={"cam1": "/video1.mp4"},
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            config.to_yaml(temp_path)

            # Read the raw YAML
            with open(temp_path) as f:
                content = f.read()

            # Check no Python object tags
            assert "!!" not in content
            assert "python" not in content.lower()

            # Verify it's valid YAML
            data = yaml.safe_load(content)
            assert data is not None
            assert "calibration_path" in data
            assert "dense_stereo" in data

            # Check nested structure
            assert isinstance(data["dense_stereo"], dict)
            assert "num_depths" in data["dense_stereo"]
        finally:
            temp_path.unlink()

    def test_yaml_handles_none_values(self):
        """Test that None values are correctly serialized/deserialized."""
        config = PipelineConfig(
            calibration_path="/path/to/calibration.json",
            output_dir="/path/to/output",
            camera_video_map={"cam1": "/video1.mp4"},
            frame_sampling=FrameSamplingConfig(start=0, stop=None, step=1),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            config.to_yaml(temp_path)

            # Check YAML contains null
            with open(temp_path) as f:
                content = f.read()
            assert "null" in content or "~" in content  # YAML null representations

            # Load and verify
            loaded = PipelineConfig.from_yaml(temp_path)
            assert loaded.frame_sampling.stop is None
        finally:
            temp_path.unlink()

    def test_empty_yaml_loads_as_defaults(self):
        """Test that loading an empty YAML file gives all defaults."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            temp_path = Path(f.name)

        try:
            loaded = PipelineConfig.from_yaml(temp_path)
            assert loaded.calibration_path == ""
            assert loaded.output_dir == ""
            assert loaded.camera_video_map == {}
            assert loaded.dense_stereo == DenseStereoConfig()
        finally:
            temp_path.unlink()

    def test_backward_compatibility_missing_output_and_viz(self):
        """Test that YAML without output/visualization fields uses defaults."""
        yaml_content = """
calibration_path: /path/to/calibration.json
output_dir: /path/to/output
camera_video_map:
  cam1: /video1.mp4

dense_stereo:
  num_depths: 256
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            loaded = PipelineConfig.from_yaml(temp_path)

            # Session fields should be loaded
            assert loaded.calibration_path == "/path/to/calibration.json"
            assert loaded.output_dir == "/path/to/output"

            # Missing output/visualization should use defaults
            assert loaded.output == OutputConfig()
            assert loaded.visualization == VizConfig()
        finally:
            temp_path.unlink()

    def test_backward_compatibility_missing_pipeline_mode(self):
        """Test that YAML without pipeline_mode defaults to 'full'."""
        yaml_content = """
calibration_path: /path/to/calibration.json
output_dir: /path/to/output
camera_video_map:
  cam1: /video1.mp4
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            loaded = PipelineConfig.from_yaml(temp_path)

            # Missing pipeline_mode should default to 'full'
            assert loaded.pipeline_mode == "full"
        finally:
            temp_path.unlink()

    def test_yaml_round_trip_with_pipeline_mode(self):
        """Test that pipeline_mode survives YAML round-trip."""
        original = PipelineConfig(
            calibration_path="/path/to/calibration.json",
            output_dir="/path/to/output",
            camera_video_map={"cam1": "/video1.mp4"},
            pipeline_mode="sparse",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            original.to_yaml(temp_path)
            loaded = PipelineConfig.from_yaml(temp_path)

            assert loaded.pipeline_mode == "sparse"
        finally:
            temp_path.unlink()

    def test_extractor_type_yaml_roundtrip(self):
        """Test that extractor_type survives YAML round-trip."""
        original = PipelineConfig(
            calibration_path="/path/to/calibration.json",
            output_dir="/path/to/output",
            camera_video_map={"cam1": "/video1.mp4"},
            feature_extraction=FeatureExtractionConfig(extractor_type="aliked"),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            original.to_yaml(temp_path)
            loaded = PipelineConfig.from_yaml(temp_path)

            assert loaded.feature_extraction.extractor_type == "aliked"
        finally:
            temp_path.unlink()

    def test_clahe_yaml_roundtrip(self):
        """Test that CLAHE config survives YAML round-trip."""
        original = PipelineConfig(
            calibration_path="/path/to/calibration.json",
            output_dir="/path/to/output",
            camera_video_map={"cam1": "/video1.mp4"},
            feature_extraction=FeatureExtractionConfig(
                clahe_enabled=True, clahe_clip_limit=4.0
            ),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            original.to_yaml(temp_path)
            loaded = PipelineConfig.from_yaml(temp_path)

            assert loaded.feature_extraction.clahe_enabled is True
            assert loaded.feature_extraction.clahe_clip_limit == 4.0
        finally:
            temp_path.unlink()

    def test_benchmark_config_yaml_roundtrip(self):
        """Test that benchmark config survives YAML round-trip."""
        original = PipelineConfig(
            calibration_path="/path/to/calibration.json",
            output_dir="/path/to/output",
            camera_video_map={"cam1": "/video1.mp4"},
            benchmark=BenchmarkConfig(
                extractors=["superpoint", "aliked"],
                clahe=[False],
            ),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            original.to_yaml(temp_path)
            loaded = PipelineConfig.from_yaml(temp_path)

            assert loaded.benchmark.extractors == ["superpoint", "aliked"]
            assert loaded.benchmark.clahe == [False]
        finally:
            temp_path.unlink()

    def test_backward_compat_no_benchmark(self):
        """Test that YAML without benchmark section loads with defaults."""
        yaml_content = """
calibration_path: /path/to/calibration.json
output_dir: /path/to/output
camera_video_map:
  cam1: /video1.mp4
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            loaded = PipelineConfig.from_yaml(temp_path)

            # Missing benchmark should use defaults
            assert loaded.benchmark == BenchmarkConfig()
            assert loaded.benchmark.extractors == ["superpoint", "aliked", "disk"]
            assert loaded.benchmark.clahe == [True, False]
        finally:
            temp_path.unlink()

    def test_matcher_type_yaml_roundtrip(self):
        """Test that matcher_type survives YAML round-trip."""
        original = PipelineConfig(
            calibration_path="/path/to/calibration.json",
            output_dir="/path/to/output",
            camera_video_map={"cam1": "/video1.mp4"},
            matcher_type="roma",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            original.to_yaml(temp_path)
            loaded = PipelineConfig.from_yaml(temp_path)

            assert loaded.matcher_type == "roma"
        finally:
            temp_path.unlink()

    def test_backward_compat_no_matcher_type(self):
        """Test that YAML without matcher_type defaults to lightglue."""
        yaml_content = """
calibration_path: /path/to/calibration.json
output_dir: /path/to/output
camera_video_map:
  cam1: /video1.mp4
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            loaded = PipelineConfig.from_yaml(temp_path)

            # Missing matcher_type should default to lightglue
            assert loaded.matcher_type == "lightglue"
        finally:
            temp_path.unlink()

    def test_dense_matching_config_yaml_roundtrip(self):
        """Test that dense_matching config survives YAML round-trip."""
        original = PipelineConfig(
            calibration_path="/path/to/calibration.json",
            output_dir="/path/to/output",
            camera_video_map={"cam1": "/video1.mp4"},
            dense_matching=DenseMatchingConfig(
                certainty_threshold=0.7, max_correspondences=5000
            ),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            original.to_yaml(temp_path)
            loaded = PipelineConfig.from_yaml(temp_path)

            assert loaded.dense_matching.certainty_threshold == 0.7
            assert loaded.dense_matching.max_correspondences == 5000
        finally:
            temp_path.unlink()

    def test_backward_compat_no_dense_matching(self):
        """Test that YAML without dense_matching section loads with defaults."""
        yaml_content = """
calibration_path: /path/to/calibration.json
output_dir: /path/to/output
camera_video_map:
  cam1: /video1.mp4
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            loaded = PipelineConfig.from_yaml(temp_path)

            # Missing dense_matching should use defaults
            assert loaded.dense_matching == DenseMatchingConfig()
            assert loaded.dense_matching.certainty_threshold == 0.5
            assert loaded.dense_matching.max_correspondences == 100000
        finally:
            temp_path.unlink()

    def test_color_norm_yaml_roundtrip(self):
        """Test that color_norm config survives YAML round-trip."""
        original = PipelineConfig(
            calibration_path="/path/to/calibration.json",
            output_dir="/path/to/output",
            camera_video_map={"cam1": "/video1.mp4"},
            color_norm=ColorNormConfig(enabled=True, method="histogram"),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            original.to_yaml(temp_path)
            loaded = PipelineConfig.from_yaml(temp_path)

            assert loaded.color_norm.enabled is True
            assert loaded.color_norm.method == "histogram"
        finally:
            temp_path.unlink()

    def test_backward_compat_no_color_norm(self):
        """Test that YAML without color_norm section loads with defaults."""
        yaml_content = """
calibration_path: /path/to/calibration.json
output_dir: /path/to/output
camera_video_map:
  cam1: /video1.mp4
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            loaded = PipelineConfig.from_yaml(temp_path)

            # Missing color_norm should use defaults
            assert loaded.color_norm == ColorNormConfig()
            assert loaded.color_norm.enabled is False
            assert loaded.color_norm.method == "gain"
        finally:
            temp_path.unlink()


class TestImports:
    """Test that all config classes can be imported individually."""

    def test_import_individual_configs(self):
        """Test that sub-configs can be imported individually."""
        # This test verifies the imports work (already done at top of file)
        assert ColorNormConfig is not None
        assert FrameSamplingConfig is not None
        assert FeatureExtractionConfig is not None
        assert PairSelectionConfig is not None
        assert MatchingConfig is not None
        assert DenseStereoConfig is not None
        assert FusionConfig is not None
        assert SurfaceConfig is not None
        assert EvaluationConfig is not None
        assert DeviceConfig is not None
        assert OutputConfig is not None
        assert VizConfig is not None
        assert PipelineConfig is not None

    def test_import_from_package(self):
        """Test that configs can be imported from the package."""
        from aquamvs import (
            DenseStereoConfig as DenseStereoConfig2,
        )
        from aquamvs import (
            PipelineConfig as PipelineConfig2,
        )

        assert DenseStereoConfig2 is DenseStereoConfig
        assert PipelineConfig2 is PipelineConfig
