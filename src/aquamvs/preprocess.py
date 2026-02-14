"""Temporal median preprocessing for fish and debris removal from underwater video."""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Video file extensions to process
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}


def process_video_temporal_median(
    video_path: Path,
    output_dir: Path,
    window: int = 30,
    framestep: int = 1,
    output_format: str = "png",
) -> int:
    """Apply temporal median filtering to a video to remove transient objects.

    Computes the median pixel value over a sliding window of frames.
    Effectively removes fish, bubbles, debris, and other moving objects
    while preserving static background structure.

    Args:
        video_path: Path to input video file.
        output_dir: Directory for output frames or video.
        window: Number of frames in median window (default: 30).
        framestep: Output every Nth frame (default: 1 = every frame).
        output_format: Output format, "png" for image sequence or "mp4" for video.

    Returns:
        Number of output frames produced.

    Raises:
        RuntimeError: If video file cannot be opened.
    """
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        f"Processing {video_path.name}: {width}x{height} @ {fps:.1f} fps, "
        f"{total_frames} frames"
    )
    logger.info(f"Window size: {window} frames, framestep: {framestep}")

    # Setup output
    output_dir.mkdir(parents=True, exist_ok=True)

    video_writer = None
    if output_format == "mp4":
        output_fps = fps / framestep
        output_path = output_dir / f"{video_path.stem}_median.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(output_path), fourcc, output_fps, (width, height)
        )
        logger.info(f"Writing video to: {output_path} @ {output_fps:.1f} fps")

    # Circular buffer for sliding window (FIFO)
    buffer = []
    frame_idx = 0
    output_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Add to buffer
            buffer.append(frame)
            if len(buffer) > window:
                buffer.pop(0)  # Remove oldest frame

            # Compute median when buffer is full and framestep matches
            if len(buffer) == window and frame_idx % framestep == 0:
                # Stack frames along axis 0: (window, H, W, 3)
                frame_stack = np.array(buffer)
                median_frame = np.median(frame_stack, axis=0).astype(np.uint8)

                # Output
                if output_format == "png":
                    output_path = output_dir / f"frame_{frame_idx:06d}.png"
                    cv2.imwrite(str(output_path), median_frame)
                elif output_format == "mp4":
                    video_writer.write(median_frame)

                output_count += 1

                if output_count % 10 == 0:
                    logger.info(
                        f"Processed {frame_idx}/{total_frames} frames, "
                        f"output {output_count} median frames"
                    )

            frame_idx += 1

    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()

    logger.info(
        f"Complete: processed {frame_idx} frames, produced {output_count} output frames"
    )
    return output_count


def process_batch(
    input_path: Path,
    output_dir: Path | None = None,
    window: int = 30,
    framestep: int = 1,
    output_format: str = "png",
) -> dict[str, int]:
    """Process a single video or batch of videos with temporal median filtering.

    Args:
        input_path: Video file or directory containing videos.
        output_dir: Output directory. If None, defaults to parent of input_path
            (single file) or {input_path.parent}/{input_path.name}_median (directory).
        window: Median window size in frames.
        framestep: Output every Nth frame.
        output_format: Output format ("png" or "mp4").

    Returns:
        Dictionary mapping video stem to output frame count.
    """
    results = {}

    if input_path.is_file():
        # Single file mode
        if output_dir is None:
            output_dir = input_path.parent / input_path.stem
        else:
            output_dir = output_dir / input_path.stem

        count = process_video_temporal_median(
            video_path=input_path,
            output_dir=output_dir,
            window=window,
            framestep=framestep,
            output_format=output_format,
        )
        results[input_path.stem] = count

    elif input_path.is_dir():
        # Batch mode
        video_files = [
            f
            for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        ]

        if not video_files:
            logger.warning(f"No video files found in {input_path}")
            return results

        if output_dir is None:
            output_dir = input_path.parent / f"{input_path.name}_median"

        logger.info(f"Batch processing {len(video_files)} video(s)")

        for video_file in video_files:
            video_output_dir = output_dir / video_file.stem

            try:
                count = process_video_temporal_median(
                    video_path=video_file,
                    output_dir=video_output_dir,
                    window=window,
                    framestep=framestep,
                    output_format=output_format,
                )
                results[video_file.stem] = count
            except Exception as e:
                logger.error(f"Failed to process {video_file.name}: {e}")
                results[video_file.stem] = 0

    else:
        raise ValueError(f"Input path is neither file nor directory: {input_path}")

    return results
