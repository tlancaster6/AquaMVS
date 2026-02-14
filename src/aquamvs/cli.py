"""Command-line interface for AquaMVS pipeline."""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import cv2
from aquacal.io.video import VideoSet

from aquamvs.calibration import (
    compute_undistortion_maps,
    load_calibration_data,
    undistort_image,
)
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
        with open(calibration_path) as f:
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
    print(f"\n{'=' * 70}")
    print("Configuration Initialization Summary")
    print(f"{'=' * 70}\n")

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
    print(f"{'=' * 70}\n")

    return config


def export_refs_command(
    config_path: Path,
    frame: int = 0,
) -> None:
    """Export undistorted reference images for ROI mask drawing.

    Reads one frame from the configured videos, undistorts all camera images,
    and saves them as PNGs in {output_dir}/reference_images/. These images
    serve as templates for drawing ROI masks in an external image editor.

    Args:
        config_path: Path to the pipeline config YAML file.
        frame: Frame index to export (default: 0).
    """
    # 1. Load config
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        config = PipelineConfig.from_yaml(config_path)
    except Exception as e:
        print(f"Error: Failed to load config: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Load calibration
    try:
        calibration = load_calibration_data(config.calibration_path)
    except Exception as e:
        print(f"Error: Failed to load calibration: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Compute undistortion maps for all cameras with video
    video_cameras = set(config.camera_video_map.keys())
    undistortion_maps = {}
    for name in video_cameras:
        if name not in calibration.cameras:
            print(
                f"Warning: Camera {name} in config but not in calibration, skipping",
                file=sys.stderr,
            )
            continue
        cam = calibration.cameras[name]
        undistortion_maps[name] = compute_undistortion_maps(cam)

    if not undistortion_maps:
        print("Error: No valid cameras found", file=sys.stderr)
        sys.exit(1)

    # 4. Open videos and seek to frame
    try:
        with VideoSet(config.camera_video_map) as videos:
            # Read the specified frame
            frame_found = False
            for frame_idx, raw_images in videos.iterate_frames(  # noqa: B007
                start=frame, stop=frame + 1, step=1
            ):
                if frame_idx == frame:
                    frame_found = True
                    break

            if not frame_found:
                print(
                    f"Error: Frame {frame} not found in videos (beyond end?)",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Filter out None images (cameras that failed to read)
            images = {name: img for name, img in raw_images.items() if img is not None}
            if not images:
                print(f"Error: Frame {frame} has no valid images", file=sys.stderr)
                sys.exit(1)

            # 5. Undistort all images
            undistorted = {}
            for name, img in images.items():
                if name in undistortion_maps:
                    undistorted[name] = undistort_image(img, undistortion_maps[name])

            # 6. Save to output directory
            output_dir = Path(config.output_dir) / "reference_images"
            output_dir.mkdir(parents=True, exist_ok=True)

            exported_count = 0
            for name, img in undistorted.items():
                output_path = output_dir / f"{name}.png"
                cv2.imwrite(str(output_path), img)
                exported_count += 1

            # 7. Print summary
            print(f"\nExported {exported_count} reference image(s) to:")
            print(f"  {output_dir}")
            print("\nUse these images to draw ROI masks in an external image editor.")
            print("Save masks as {camera_name}.png in a dedicated mask directory,")
            print("then add 'mask_dir: /path/to/masks' to your config YAML.\n")

    except Exception as e:
        print(f"Error: Failed to export reference images: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


def run_command(
    config_path: Path, verbose: bool = False, device: str | None = None
) -> None:
    """Execute the reconstruction pipeline from a config file.

    Args:
        config_path: Path to the pipeline config YAML file.
        verbose: If True, set logging to DEBUG level.
        device: Optional device override (replaces config.device.device).
    """
    # 1. Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Silence noisy third-party loggers
    for name in ("matplotlib", "PIL", "open3d"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # 2. Load config
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        config = PipelineConfig.from_yaml(config_path)
    except Exception as e:
        print(f"Error: Failed to load config: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Apply CLI overrides
    if device is not None:
        config.device.device = device

    # 4. Validate
    try:
        config.validate()
    except ValueError as e:
        print(f"Error: Invalid configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # 5. Run pipeline
    from aquamvs.pipeline import run_pipeline

    run_pipeline(config)


def benchmark_command(config_path: Path, frame: int = 0) -> None:
    """Run benchmark sweep on a single frame.

    Args:
        config_path: Path to the pipeline config YAML file.
        frame: Frame index to benchmark (default: 0).
    """
    # 1. Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Silence noisy third-party loggers
    for name in ("matplotlib", "PIL", "open3d"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # 2. Load config
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        config = PipelineConfig.from_yaml(config_path)
    except Exception as e:
        print(f"Error: Failed to load config: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Validate
    try:
        config.validate()
    except ValueError as e:
        print(f"Error: Invalid configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # 4. Run benchmark
    from aquamvs.benchmark import generate_report, run_benchmark

    try:
        results = run_benchmark(config, frame)
        report_path = generate_report(results, Path(config.output_dir))

        # 5. Print summary
        print("\nBenchmark complete!")
        print(f"Tested {len(results.results)} configuration(s) on frame {frame}")
        print(f"Report: {report_path}\n")

    except Exception as e:
        print(f"Error: Benchmark failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


def preprocess_command(args) -> None:
    """Apply temporal median filtering to remove fish/debris from underwater video.

    Args:
        args: Parsed command-line arguments.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Lazy import to avoid loading cv2/numpy at CLI parse time
    from aquamvs.preprocess import process_batch

    # Validate input
    input_path = args.input
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Run preprocessing
    try:
        results = process_batch(
            input_path=input_path,
            output_dir=args.output_dir,
            window=args.window,
            framestep=args.framestep,
            output_format=args.format,
        )

        # Print summary
        print("\nPreprocessing complete!")
        print(f"Processed {len(results)} video(s):")
        for video_name, count in results.items():
            print(f"  {video_name}: {count} frames")
        print()

    except Exception as e:
        print(f"Error: Preprocessing failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


def export_mesh_command(args) -> None:
    """Export mesh(es) to different format with optional simplification.

    Args:
        args: Parsed command-line arguments.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Lazy import to avoid loading Open3D at CLI parse time
    from aquamvs.surface import export_mesh

    # Determine mode: single file or batch
    if args.input_dir is not None:
        # Batch mode
        input_dir = args.input_dir
        if not input_dir.exists() or not input_dir.is_dir():
            print(
                f"Error: Input directory does not exist: {input_dir}", file=sys.stderr
            )
            sys.exit(1)

        # Find all PLY files
        ply_files = list(input_dir.glob("*.ply"))
        if not ply_files:
            print(f"Error: No .ply files found in {input_dir}", file=sys.stderr)
            sys.exit(1)

        # Determine output directory
        output_dir = args.output_dir if args.output_dir is not None else input_dir

        print(f"\nBatch converting {len(ply_files)} mesh(es) to {args.format.upper()}")
        if args.simplify is not None:
            print(f"Simplifying to {args.simplify} faces")

        success_count = 0
        for ply_file in ply_files:
            output_path = output_dir / f"{ply_file.stem}.{args.format}"
            try:
                export_mesh(
                    input_path=ply_file,
                    output_path=output_path,
                    simplify=args.simplify,
                )
                success_count += 1
            except Exception as e:
                print(f"Failed to export {ply_file.name}: {e}", file=sys.stderr)

        print(f"\nBatch export complete: {success_count}/{len(ply_files)} succeeded\n")

    else:
        # Single file mode
        input_path = args.input
        if not input_path.exists():
            print(f"Error: Input file does not exist: {input_path}", file=sys.stderr)
            sys.exit(1)

        # Determine output path
        output_path = input_path.with_suffix(f".{args.format}")

        # Run export
        try:
            export_mesh(
                input_path=input_path,
                output_path=output_path,
                simplify=args.simplify,
            )
            print(f"\nExport complete: {output_path}\n")

        except Exception as e:
            print(f"Error: Export failed: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            sys.exit(1)


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

    # run subcommand
    run_parser = subparsers.add_parser(
        "run",
        help="Run reconstruction pipeline",
    )
    run_parser.add_argument(
        "config",
        type=Path,
        help="Path to pipeline config YAML file",
    )
    run_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    run_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g., 'cpu' or 'cuda')",
    )

    # export-refs subcommand
    export_refs_parser = subparsers.add_parser(
        "export-refs",
        help="Export undistorted reference images for mask drawing",
    )
    export_refs_parser.add_argument(
        "config",
        type=Path,
        help="Path to pipeline config YAML file",
    )
    export_refs_parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to export (default: 0)",
    )

    # benchmark subcommand
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run comparative benchmark of feature extraction configurations",
    )
    benchmark_parser.add_argument(
        "config",
        type=Path,
        help="Path to pipeline config YAML file",
    )
    benchmark_parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to benchmark (default: 0)",
    )

    # preprocess subcommand
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Apply temporal median filtering to remove fish/debris from underwater video",
    )
    preprocess_parser.add_argument(
        "input",
        type=Path,
        help="Video file or directory of videos",
    )
    preprocess_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (auto-computed if omitted)",
    )
    preprocess_parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Median window size in frames (default: 30)",
    )
    preprocess_parser.add_argument(
        "--framestep",
        type=int,
        default=1,
        help="Output every Nth frame (default: 1)",
    )
    preprocess_parser.add_argument(
        "--format",
        type=str,
        choices=["png", "mp4"],
        default="png",
        help="Output format (default: png)",
    )

    # export-mesh subcommand
    export_mesh_parser = subparsers.add_parser(
        "export-mesh",
        help="Export mesh to different format with optional simplification",
    )
    export_mesh_parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Input PLY mesh file (omit for batch mode with --input-dir)",
    )
    export_mesh_parser.add_argument(
        "--format",
        type=str,
        required=True,
        choices=["obj", "stl", "gltf", "glb"],
        help="Output format",
    )
    export_mesh_parser.add_argument(
        "--simplify",
        type=int,
        default=None,
        help="Target face count for simplification (optional)",
    )
    export_mesh_parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Batch mode: convert all PLY files in directory",
    )
    export_mesh_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for batch mode (defaults to input-dir)",
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
        run_command(
            config_path=args.config,
            verbose=args.verbose,
            device=args.device,
        )
    elif args.command == "export-refs":
        export_refs_command(
            config_path=args.config,
            frame=args.frame,
        )
    elif args.command == "benchmark":
        benchmark_command(
            config_path=args.config,
            frame=args.frame,
        )
    elif args.command == "preprocess":
        preprocess_command(args)
    elif args.command == "export-mesh":
        export_mesh_command(args)
    else:
        parser.print_help()
        sys.exit(1)
