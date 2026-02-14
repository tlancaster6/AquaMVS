"""Tests for CLI init and run commands."""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from aquamvs.cli import init_config, main, run_command
from aquamvs.config import PipelineConfig


@pytest.fixture
def calibration_json(tmp_path: Path) -> Path:
    """Create a minimal calibration JSON file for testing.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path to the calibration JSON file.
    """
    cal_path = tmp_path / "calibration.json"
    data = {
        "version": "1.0",
        "cameras": {
            "e3v82e0": {},
            "e3v831b": {},
            "e3v8ab7": {},
        },
    }
    with open(cal_path, "w") as f:
        json.dump(data, f)
    return cal_path


def test_init_happy_path(tmp_path: Path, calibration_json: Path):
    """Test init with all videos matching calibration cameras."""
    # Create video directory with matching videos
    video_dir = tmp_path / "videos"
    video_dir.mkdir()

    (video_dir / "e3v82e0-session1.mp4").touch()
    (video_dir / "e3v831b-session1.mp4").touch()
    (video_dir / "e3v8ab7-session1.mp4").touch()

    # Run init
    config_path = tmp_path / "config.yaml"
    output_dir = str(tmp_path / "output")

    config = init_config(
        video_dir=video_dir,
        pattern=r"^([a-z0-9]+)-",
        calibration_path=calibration_json,
        output_dir=output_dir,
        config_path=config_path,
    )

    # Verify config was saved
    assert config_path.exists()

    # Verify config content
    assert config.calibration_path == str(calibration_json)
    assert config.output_dir == output_dir
    assert len(config.camera_video_map) == 3
    assert "e3v82e0" in config.camera_video_map
    assert "e3v831b" in config.camera_video_map
    assert "e3v8ab7" in config.camera_video_map

    # Verify paths are correct
    assert config.camera_video_map["e3v82e0"].endswith("e3v82e0-session1.mp4")

    # Verify we can reload it
    loaded_config = PipelineConfig.from_yaml(config_path)
    assert loaded_config.camera_video_map == config.camera_video_map


def test_init_partial_match(tmp_path: Path, calibration_json: Path):
    """Test init with only some videos matching calibration."""
    # Create video directory with partial matches
    video_dir = tmp_path / "videos"
    video_dir.mkdir()

    (video_dir / "e3v82e0-session1.mp4").touch()  # matches
    (video_dir / "e3v831b-session1.mp4").touch()  # matches
    (video_dir / "unknown-session1.mp4").touch()  # doesn't match calibration

    # Run init
    config_path = tmp_path / "config.yaml"
    output_dir = str(tmp_path / "output")

    config = init_config(
        video_dir=video_dir,
        pattern=r"^([a-z0-9]+)-",
        calibration_path=calibration_json,
        output_dir=output_dir,
        config_path=config_path,
    )

    # Only matched cameras should be in config
    assert len(config.camera_video_map) == 2
    assert "e3v82e0" in config.camera_video_map
    assert "e3v831b" in config.camera_video_map
    assert "unknown" not in config.camera_video_map

    # e3v8ab7 is in calibration but has no video - should be reported but not error
    assert "e3v8ab7" not in config.camera_video_map


def test_init_no_matches(tmp_path: Path, calibration_json: Path):
    """Test init exits with error when no cameras match."""
    # Create video directory with no matching videos
    video_dir = tmp_path / "videos"
    video_dir.mkdir()

    (video_dir / "unknown1-session1.mp4").touch()
    (video_dir / "unknown2-session1.mp4").touch()

    config_path = tmp_path / "config.yaml"
    output_dir = str(tmp_path / "output")

    # Should exit with error
    with pytest.raises(SystemExit) as exc_info:
        init_config(
            video_dir=video_dir,
            pattern=r"^([a-z0-9]+)-",
            calibration_path=calibration_json,
            output_dir=output_dir,
            config_path=config_path,
        )

    assert exc_info.value.code == 1
    # Config file should not be created
    assert not config_path.exists()


def test_init_no_capture_group(tmp_path: Path, calibration_json: Path):
    """Test init exits with error when regex has no capture group."""
    # Create video directory
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "e3v82e0-session1.mp4").touch()

    config_path = tmp_path / "config.yaml"
    output_dir = str(tmp_path / "output")

    # Pattern with no capture group
    with pytest.raises(SystemExit) as exc_info:
        init_config(
            video_dir=video_dir,
            pattern=r"^[a-z0-9]+-",  # No parentheses = no capture group
            calibration_path=calibration_json,
            output_dir=output_dir,
            config_path=config_path,
        )

    assert exc_info.value.code == 1
    assert not config_path.exists()


def test_init_non_video_files_ignored(tmp_path: Path, calibration_json: Path):
    """Test init ignores non-video files in the directory."""
    # Create video directory with mix of files
    video_dir = tmp_path / "videos"
    video_dir.mkdir()

    (video_dir / "e3v82e0-session1.mp4").touch()
    (video_dir / "e3v831b-session1.avi").touch()
    (video_dir / "notes.txt").touch()  # Should be ignored
    (video_dir / "metadata.json").touch()  # Should be ignored
    (video_dir / "README.md").touch()  # Should be ignored

    config_path = tmp_path / "config.yaml"
    output_dir = str(tmp_path / "output")

    config = init_config(
        video_dir=video_dir,
        pattern=r"^([a-z0-9]+)-",
        calibration_path=calibration_json,
        output_dir=output_dir,
        config_path=config_path,
    )

    # Only video files should be processed
    assert len(config.camera_video_map) == 2
    assert "e3v82e0" in config.camera_video_map
    assert "e3v831b" in config.camera_video_map


def test_init_case_insensitive_extensions(tmp_path: Path, calibration_json: Path):
    """Test init handles video extensions case-insensitively."""
    # Create video directory with uppercase extensions
    video_dir = tmp_path / "videos"
    video_dir.mkdir()

    (video_dir / "e3v82e0-session1.MP4").touch()
    (video_dir / "e3v831b-session1.AVI").touch()
    (video_dir / "e3v8ab7-session1.MKV").touch()

    config_path = tmp_path / "config.yaml"
    output_dir = str(tmp_path / "output")

    config = init_config(
        video_dir=video_dir,
        pattern=r"^([a-z0-9]+)-",
        calibration_path=calibration_json,
        output_dir=output_dir,
        config_path=config_path,
    )

    # All should be recognized
    assert len(config.camera_video_map) == 3


def test_init_no_regex_match(tmp_path: Path, calibration_json: Path):
    """Test init when video filenames don't match the regex pattern."""
    # Create video directory with non-matching filenames
    video_dir = tmp_path / "videos"
    video_dir.mkdir()

    (video_dir / "e3v82e0-session1.mp4").touch()  # matches
    (video_dir / "no_dash_e3v831b.mp4").touch()  # doesn't match pattern

    config_path = tmp_path / "config.yaml"
    output_dir = str(tmp_path / "output")

    config = init_config(
        video_dir=video_dir,
        pattern=r"^([a-z0-9]+)-",
        calibration_path=calibration_json,
        output_dir=output_dir,
        config_path=config_path,
    )

    # Only the one that matched the pattern should be included
    assert len(config.camera_video_map) == 1
    assert "e3v82e0" in config.camera_video_map


def test_init_missing_video_dir(tmp_path: Path, calibration_json: Path):
    """Test init exits with error when video directory doesn't exist."""
    video_dir = tmp_path / "nonexistent"
    config_path = tmp_path / "config.yaml"
    output_dir = str(tmp_path / "output")

    with pytest.raises(SystemExit) as exc_info:
        init_config(
            video_dir=video_dir,
            pattern=r"^([a-z0-9]+)-",
            calibration_path=calibration_json,
            output_dir=output_dir,
            config_path=config_path,
        )

    assert exc_info.value.code == 1


def test_init_missing_calibration(tmp_path: Path):
    """Test init exits with error when calibration file doesn't exist."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "e3v82e0-session1.mp4").touch()

    calibration_path = tmp_path / "nonexistent.json"
    config_path = tmp_path / "config.yaml"
    output_dir = str(tmp_path / "output")

    with pytest.raises(SystemExit) as exc_info:
        init_config(
            video_dir=video_dir,
            pattern=r"^([a-z0-9]+)-",
            calibration_path=calibration_path,
            output_dir=output_dir,
            config_path=config_path,
        )

    assert exc_info.value.code == 1


def test_init_invalid_calibration_json(tmp_path: Path):
    """Test init exits with error when calibration JSON is malformed."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "e3v82e0-session1.mp4").touch()

    # Create invalid JSON
    calibration_path = tmp_path / "bad.json"
    with open(calibration_path, "w") as f:
        f.write("{invalid json")

    config_path = tmp_path / "config.yaml"
    output_dir = str(tmp_path / "output")

    with pytest.raises(SystemExit) as exc_info:
        init_config(
            video_dir=video_dir,
            pattern=r"^([a-z0-9]+)-",
            calibration_path=calibration_path,
            output_dir=output_dir,
            config_path=config_path,
        )

    assert exc_info.value.code == 1


def test_init_missing_cameras_key(tmp_path: Path):
    """Test init exits with error when calibration JSON has no 'cameras' key."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "e3v82e0-session1.mp4").touch()

    # Create JSON without 'cameras' key
    calibration_path = tmp_path / "bad.json"
    with open(calibration_path, "w") as f:
        json.dump({"version": "1.0"}, f)

    config_path = tmp_path / "config.yaml"
    output_dir = str(tmp_path / "output")

    with pytest.raises(SystemExit) as exc_info:
        init_config(
            video_dir=video_dir,
            pattern=r"^([a-z0-9]+)-",
            calibration_path=calibration_path,
            output_dir=output_dir,
            config_path=config_path,
        )

    assert exc_info.value.code == 1


# ============================================================================
# Tests for `run` command
# ============================================================================


def test_run_missing_config(tmp_path: Path):
    """Test run command exits with error when config file doesn't exist."""
    config_path = tmp_path / "nonexistent.yaml"

    with pytest.raises(SystemExit) as exc_info:
        run_command(config_path)

    assert exc_info.value.code == 1


def test_run_valid_config_mocked_pipeline(tmp_path: Path):
    """Test run command with valid config calls run_pipeline."""
    # Create a minimal valid config YAML
    config_path = tmp_path / "config.yaml"
    config_data = {
        "calibration_path": str(tmp_path / "calibration.json"),
        "output_dir": str(tmp_path / "output"),
        "camera_video_map": {
            "cam1": str(tmp_path / "cam1.mp4"),
        },
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    # Mock run_pipeline
    with patch("aquamvs.pipeline.run_pipeline") as mock_run_pipeline:
        run_command(config_path)

        # Verify run_pipeline was called once with a PipelineConfig instance
        assert mock_run_pipeline.call_count == 1
        args, _ = mock_run_pipeline.call_args
        assert isinstance(args[0], PipelineConfig)
        assert args[0].calibration_path == config_data["calibration_path"]
        assert args[0].output_dir == config_data["output_dir"]


def test_run_device_override(tmp_path: Path):
    """Test run command --device override."""
    # Create a config with device: cpu
    config_path = tmp_path / "config.yaml"
    config_data = {
        "calibration_path": str(tmp_path / "calibration.json"),
        "output_dir": str(tmp_path / "output"),
        "camera_video_map": {
            "cam1": str(tmp_path / "cam1.mp4"),
        },
        "device": {
            "device": "cpu",
        },
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    # Call with device override
    with patch("aquamvs.pipeline.run_pipeline") as mock_run_pipeline:
        run_command(config_path, device="cuda")

        # Verify the config passed to run_pipeline has device == "cuda"
        assert mock_run_pipeline.call_count == 1
        args, _ = mock_run_pipeline.call_args
        assert args[0].device.device == "cuda"


def test_run_verbose_flag(tmp_path: Path, caplog):
    """Test run command --verbose sets logging to DEBUG."""
    # Create a minimal valid config
    config_path = tmp_path / "config.yaml"
    config_data = {
        "calibration_path": str(tmp_path / "calibration.json"),
        "output_dir": str(tmp_path / "output"),
        "camera_video_map": {
            "cam1": str(tmp_path / "cam1.mp4"),
        },
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    # Mock run_pipeline
    with patch("aquamvs.pipeline.run_pipeline"):
        # Clear any previous logging config
        logging.root.handlers = []

        run_command(config_path, verbose=True)

        # Verify root logger is set to DEBUG
        assert logging.getLogger().level == logging.DEBUG


def test_run_verbose_false_sets_info(tmp_path: Path):
    """Test run command without --verbose sets logging to INFO."""
    # Create a minimal valid config
    config_path = tmp_path / "config.yaml"
    config_data = {
        "calibration_path": str(tmp_path / "calibration.json"),
        "output_dir": str(tmp_path / "output"),
        "camera_video_map": {
            "cam1": str(tmp_path / "cam1.mp4"),
        },
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    # Mock run_pipeline
    with patch("aquamvs.pipeline.run_pipeline"):
        # Clear any previous logging config
        logging.root.handlers = []

        run_command(config_path, verbose=False)

        # Verify root logger is set to INFO
        assert logging.getLogger().level == logging.INFO


def test_run_invalid_config_validation_error(tmp_path: Path):
    """Test run command with invalid config exits with error."""
    # Create a config with an invalid cost function
    config_path = tmp_path / "config.yaml"
    config_data = {
        "calibration_path": str(tmp_path / "calibration.json"),
        "output_dir": str(tmp_path / "output"),
        "camera_video_map": {
            "cam1": str(tmp_path / "cam1.mp4"),
        },
        "dense_stereo": {
            "cost_function": "invalid_function",
        },
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    # Should exit with error
    with pytest.raises(SystemExit) as exc_info:
        run_command(config_path)

    assert exc_info.value.code == 1


def test_run_malformed_yaml(tmp_path: Path):
    """Test run command with malformed YAML exits with error."""
    # Create a malformed YAML file
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        f.write("{invalid: yaml: content:")

    # Should exit with error
    with pytest.raises(SystemExit) as exc_info:
        run_command(config_path)

    assert exc_info.value.code == 1


def test_main_run_argument_parsing(tmp_path: Path):
    """Test main() correctly parses run subcommand arguments."""
    # Create a minimal valid config
    config_path = tmp_path / "config.yaml"
    config_data = {
        "calibration_path": str(tmp_path / "calibration.json"),
        "output_dir": str(tmp_path / "output"),
        "camera_video_map": {
            "cam1": str(tmp_path / "cam1.mp4"),
        },
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    # Test basic run command
    with patch("sys.argv", ["aquamvs", "run", str(config_path)]):
        with patch("aquamvs.cli.run_command") as mock_run_command:
            main()
            mock_run_command.assert_called_once_with(
                config_path=config_path,
                verbose=False,
                device=None,
            )

    # Test run with --verbose
    with patch("sys.argv", ["aquamvs", "run", "-v", str(config_path)]):
        with patch("aquamvs.cli.run_command") as mock_run_command:
            main()
            mock_run_command.assert_called_once_with(
                config_path=config_path,
                verbose=True,
                device=None,
            )

    # Test run with --device
    with patch("sys.argv", ["aquamvs", "run", "--device", "cuda", str(config_path)]):
        with patch("aquamvs.cli.run_command") as mock_run_command:
            main()
            mock_run_command.assert_called_once_with(
                config_path=config_path,
                verbose=False,
                device="cuda",
            )


# ============================================================================
# Tests for `export-refs` command
# ============================================================================


def test_export_refs_creates_images(tmp_path: Path, calibration_json: Path):
    """Test export-refs creates undistorted images for all cameras."""
    from unittest.mock import patch

    import numpy as np

    # Create config
    config_path = tmp_path / "config.yaml"
    output_dir = str(tmp_path / "output")
    config_data = {
        "calibration_path": str(calibration_json),
        "output_dir": output_dir,
        "camera_video_map": {
            "e3v82e0": str(tmp_path / "e3v82e0.mp4"),
            "e3v831b": str(tmp_path / "e3v831b.mp4"),
        },
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    # Mock VideoSet
    mock_images = {
        "e3v82e0": np.zeros((480, 640, 3), dtype=np.uint8),
        "e3v831b": np.zeros((480, 640, 3), dtype=np.uint8),
    }

    mock_video_set = MagicMock()
    mock_video_set.__enter__.return_value = mock_video_set
    mock_video_set.iterate_frames.return_value = [(0, mock_images)]

    # Mock calibration loading and undistortion
    mock_cal_data = MagicMock()
    mock_cam = MagicMock()
    mock_cam.image_size = (640, 480)
    mock_cal_data.cameras = {
        "e3v82e0": mock_cam,
        "e3v831b": mock_cam,
    }

    mock_undist_data = MagicMock()

    with (
        patch("aquamvs.io.detect_input_type", return_value="video"),
        patch("aquamvs.cli.VideoSet", return_value=mock_video_set),
        patch("aquamvs.cli.load_calibration_data", return_value=mock_cal_data),
        patch("aquamvs.cli.compute_undistortion_maps", return_value=mock_undist_data),
        patch("aquamvs.cli.undistort_image", side_effect=lambda img, _: img),
        patch("aquamvs.cli.cv2.imwrite") as mock_imwrite,
    ):
        from aquamvs.cli import export_refs_command

        export_refs_command(config_path, frame=0)

        # Verify images were written
        assert mock_imwrite.call_count == 2

        # Check that the output directory was created and images saved
        calls = mock_imwrite.call_args_list
        saved_paths = [call[0][0] for call in calls]

        assert any("e3v82e0.png" in path for path in saved_paths)
        assert any("e3v831b.png" in path for path in saved_paths)


def test_export_refs_includes_center_camera(tmp_path: Path):
    """Test export-refs includes auxiliary (center) cameras."""
    from unittest.mock import patch

    import numpy as np

    # Create calibration with ring and auxiliary cameras
    cal_path = tmp_path / "calibration.json"
    data = {
        "version": "1.0",
        "cameras": {
            "e3v82e0": {"is_auxiliary": False},  # Ring camera
            "center": {"is_auxiliary": True},  # Auxiliary camera
        },
    }
    with open(cal_path, "w") as f:
        json.dump(data, f)

    # Create config
    config_path = tmp_path / "config.yaml"
    output_dir = str(tmp_path / "output")
    config_data = {
        "calibration_path": str(cal_path),
        "output_dir": output_dir,
        "camera_video_map": {
            "e3v82e0": str(tmp_path / "e3v82e0.mp4"),
            "center": str(tmp_path / "center.mp4"),
        },
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    # Mock VideoSet
    mock_images = {
        "e3v82e0": np.zeros((480, 640, 3), dtype=np.uint8),
        "center": np.zeros((480, 640, 3), dtype=np.uint8),
    }

    mock_video_set = MagicMock()
    mock_video_set.__enter__.return_value = mock_video_set
    mock_video_set.iterate_frames.return_value = [(0, mock_images)]

    # Mock calibration
    mock_cal_data = MagicMock()
    mock_cam = MagicMock()
    mock_cam.image_size = (640, 480)
    mock_cal_data.cameras = {
        "e3v82e0": mock_cam,
        "center": mock_cam,
    }

    mock_undist_data = MagicMock()

    with (
        patch("aquamvs.io.detect_input_type", return_value="video"),
        patch("aquamvs.cli.VideoSet", return_value=mock_video_set),
        patch("aquamvs.cli.load_calibration_data", return_value=mock_cal_data),
        patch("aquamvs.cli.compute_undistortion_maps", return_value=mock_undist_data),
        patch("aquamvs.cli.undistort_image", side_effect=lambda img, _: img),
        patch("aquamvs.cli.cv2.imwrite") as mock_imwrite,
    ):
        from aquamvs.cli import export_refs_command

        export_refs_command(config_path, frame=0)

        # Verify both cameras were exported
        assert mock_imwrite.call_count == 2

        calls = mock_imwrite.call_args_list
        saved_paths = [call[0][0] for call in calls]

        assert any("e3v82e0.png" in path for path in saved_paths)
        assert any("center.png" in path for path in saved_paths)


def test_export_refs_argument_parsing():
    """Test main() correctly parses export-refs subcommand arguments."""
    from unittest.mock import patch

    # Test basic export-refs command (default frame=0)
    with patch("sys.argv", ["aquamvs", "export-refs", "config.yaml"]):
        with patch("aquamvs.cli.export_refs_command") as mock_export:
            from aquamvs.cli import main

            main()
            mock_export.assert_called_once_with(
                config_path=Path("config.yaml"),
                frame=0,
            )

    # Test export-refs with --frame argument
    with patch("sys.argv", ["aquamvs", "export-refs", "config.yaml", "--frame", "5"]):
        with patch("aquamvs.cli.export_refs_command") as mock_export:
            from aquamvs.cli import main

            main()
            mock_export.assert_called_once_with(
                config_path=Path("config.yaml"),
                frame=5,
            )


# ============================================================================
# Tests for `benchmark` command
# ============================================================================


def test_benchmark_argument_parsing():
    """Test main() correctly parses benchmark subcommand arguments."""
    from unittest.mock import patch

    # Test basic benchmark command (default frame=0)
    with patch("sys.argv", ["aquamvs", "benchmark", "config.yaml"]):
        with patch("aquamvs.cli.benchmark_command") as mock_benchmark:
            from aquamvs.cli import main

            main()
            mock_benchmark.assert_called_once_with(
                config_path=Path("config.yaml"),
                frame=0,
            )

    # Test benchmark with --frame argument
    with patch("sys.argv", ["aquamvs", "benchmark", "config.yaml", "--frame", "5"]):
        with patch("aquamvs.cli.benchmark_command") as mock_benchmark:
            from aquamvs.cli import main

            main()
            mock_benchmark.assert_called_once_with(
                config_path=Path("config.yaml"),
                frame=5,
            )


def test_benchmark_default_frame():
    """Test benchmark command defaults to frame=0."""
    from unittest.mock import patch

    with patch("sys.argv", ["aquamvs", "benchmark", "config.yaml"]):
        with patch("aquamvs.cli.benchmark_command") as mock_benchmark:
            from aquamvs.cli import main

            main()
            # Verify frame defaults to 0
            assert mock_benchmark.call_args[1]["frame"] == 0
