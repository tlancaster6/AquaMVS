"""Command-line interface for AquaMVS pipeline."""

import argparse
import json
import re
import sys
from pathlib import Path

from aquamvs.config import PipelineConfig

# Video file extensions to scan for
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}


def init_config(
    video_dir: Path,
    pattern: str,
    calibration_path: Path,
    output_dir: str,
    config_path: Path,
) -> PipelineConfig:
    """Generate a PipelineConfig from a video directory and calibration file.

    Scans the video directory for video files, extracts camera names via regex,
    cross-references against calibration camera names, and generates a config YAML.

    Args:
        video_dir: Directory containing video files.
        pattern: Regex pattern to extract camera name from filename.
            The first capture group is used as the camera name.
        calibration_path: Path to AquaCal calibration JSON file.
        output_dir: Output directory for reconstruction results.
        config_path: Path where the generated config YAML will be saved.

    Returns:
        The generated PipelineConfig.

    Raises:
        SystemExit: If no cameras are matched or if the regex pattern has no capture group.
    """
    # 1. Scan video directory
    if not video_dir.exists():
        print(f"Error: Video directory does not exist: {video_dir}", file=sys.stderr)
        sys.exit(1)

    video_files = [
        f
        for f in video_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not video_files:
        print(f"Error: No video files found in {video_dir}", file=sys.stderr)
        sys.exit(1)

    # 2. Extract camera names from video filenames
    try:
        regex = re.compile(pattern)
    except re.error as e:
        print(f"Error: Invalid regex pattern '{pattern}': {e}", file=sys.stderr)
        sys.exit(1)

    # Check that pattern has at least one capture group
    if regex.groups < 1:
        print(
            f"Error: Regex pattern '{pattern}' has no capture groups. "
            "The first capture group should extract the camera name.",
            file=sys.stderr,
        )
        sys.exit(1)

    video_camera_map: dict[str, Path] = {}
    unmatched_videos: list[Path] = []

    for video_file in video_files:
        match = regex.match(video_file.name)
        if match:
            camera_name = match.group(1)
            video_camera_map[camera_name] = video_file
        else:
            unmatched_videos.append(video_file)

    # 3. Load calibration camera names
    if not calibration_path.exists():
        print(
            f"Error: Calibration file does not exist: {calibration_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        with open(calibration_path, "r") as f:
            calibration_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in calibration file: {e}", file=sys.stderr)
        sys.exit(1)

    if "cameras" not in calibration_data:
        print("Error: Calibration file missing 'cameras' key", file=sys.stderr)
        sys.exit(1)

    calibration_cameras = set(calibration_data["cameras"].keys())

    # 4. Cross-reference: find intersection
    matched_cameras = set(video_camera_map.keys()) & calibration_cameras
    videos_without_calibration = set(video_camera_map.keys()) - calibration_cameras
    cameras_without_video = calibration_cameras - set(video_camera_map.keys())

    # 5. Report
    print(f"\n{'='*70}")
    print("Configuration Initialization Summary")
    print(f"{'='*70}\n")

    if matched_cameras:
        print(f"[OK] Matched {len(matched_cameras)} camera(s):")
        for camera in sorted(matched_cameras):
            print(f"  {camera:15s} -> {video_camera_map[camera].name}")
        print()
    else:
        print("[ERROR] No cameras matched!\n")

    if unmatched_videos:
        print(f"[WARN] {len(unmatched_videos)} video(s) with no regex match:")
        for video in sorted(unmatched_videos):
            print(f"  {video.name}")
        print()

    if videos_without_calibration:
        print(
            f"[WARN] {len(videos_without_calibration)} video(s) with no calibration entry:"
        )
        for camera in sorted(videos_without_calibration):
            print(f"  {camera:15s} -> {video_camera_map[camera].name}")
        print()

    if cameras_without_video:
        print(
            f"[WARN] {len(cameras_without_video)} calibration camera(s) with no video:"
        )
        for camera in sorted(cameras_without_video):
            print(f"  {camera}")
        print()

    if not matched_cameras:
        print(
            "Error: No cameras matched between videos and calibration.", file=sys.stderr
        )
        print("Check your regex pattern and calibration file.", file=sys.stderr)
        sys.exit(1)

    # 6. Build config
    camera_video_map = {
        camera: str(video_camera_map[camera]) for camera in matched_cameras
    }

    config = PipelineConfig(
        calibration_path=str(calibration_path),
        output_dir=output_dir,
        camera_video_map=camera_video_map,
    )

    # 7. Save
    config.to_yaml(config_path)
    print(f"[OK] Configuration saved to: {config_path}")
    print(f"{'='*70}\n")

    return config


def main() -> None:
    """Main entry point for the AquaMVS CLI."""
    parser = argparse.ArgumentParser(
        prog="aquamvs",
        description="Multi-view stereo reconstruction of underwater surfaces with refraction modeling.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init subcommand
    init_parser = subparsers.add_parser(
        "init",
        help="Generate config from video directory",
    )
    init_parser.add_argument(
        "--video-dir",
        type=Path,
        required=True,
        help="Directory containing video files",
    )
    init_parser.add_argument(
        "--pattern",
        type=str,
        required=True,
        help="Regex pattern to extract camera name from filename (first capture group)",
    )
    init_parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="Path to AquaCal calibration JSON file",
    )
    init_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for reconstruction results",
    )
    init_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to output config YAML file (default: config.yaml)",
    )

    # run subcommand (placeholder for P.22)
    run_parser = subparsers.add_parser(
        "run",
        help="Run reconstruction pipeline",
    )

    args = parser.parse_args()

    # Dispatch
    if args.command == "init":
        init_config(
            video_dir=args.video_dir,
            pattern=args.pattern,
            calibration_path=args.calibration,
            output_dir=args.output_dir,
            config_path=args.config,
        )
    elif args.command == "run":
        print("Error: 'run' command not yet implemented", file=sys.stderr)
        sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)
