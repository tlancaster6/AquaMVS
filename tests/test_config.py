"""Tests for configuration system."""

import tempfile
from pathlib import Path

import pytest
import yaml

from aquamvs.config import (
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
        assert config.max_keypoints == 2048
        assert config.detection_threshold == 0.005

    def test_custom_values(self):
        """Test custom values."""
        config = FeatureExtractionConfig(max_keypoints=1024, detection_threshold=0.01)
        assert config.max_keypoints == 1024
        assert config.detection_threshold == 0.01


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
        assert config.min_confidence == 0.5

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

    def test_empty_config(self):
        """Test creating an empty config (all defaults)."""
        config = PipelineConfig()
        assert config.calibration_path == ""
        assert config.output_dir == ""
        assert config.camera_video_map == {}

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
            with open(temp_path, "r") as f:
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
            with open(temp_path, "r") as f:
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


class TestImports:
    """Test that all config classes can be imported individually."""

    def test_import_individual_configs(self):
        """Test that sub-configs can be imported individually."""
        # This test verifies the imports work (already done at top of file)
        assert FrameSamplingConfig is not None
        assert FeatureExtractionConfig is not None
        assert PairSelectionConfig is not None
        assert MatchingConfig is not None
        assert DenseStereoConfig is not None
        assert FusionConfig is not None
        assert SurfaceConfig is not None
        assert EvaluationConfig is not None
        assert DeviceConfig is not None
        assert PipelineConfig is not None

    def test_import_from_package(self):
        """Test that configs can be imported from the package."""
        from aquamvs import (
            DenseStereoConfig as DenseStereoConfig2,
            PipelineConfig as PipelineConfig2,
        )

        assert DenseStereoConfig2 is DenseStereoConfig
        assert PipelineConfig2 is PipelineConfig
