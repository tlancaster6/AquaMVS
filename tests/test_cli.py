"""Tests for CLI init command."""

import json
from pathlib import Path

import pytest

from aquamvs.cli import init_config
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
