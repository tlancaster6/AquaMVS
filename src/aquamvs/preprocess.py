"""Temporal median preprocessing for fish and debris removal from underwater video."""

import logging
import time
from collections import deque
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
    exact_seek: bool = False,
    window_step: int = 1,
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
        exact_seek: Force sequential reading (default: False = hybrid seek mode).
        window_step: Sample every Nth frame within the median window (default: 1).

    Returns:
        Number of output frames produced.

    Raises:
        RuntimeError: If video file cannot be opened.
    """
    start_time = time.time()

    # Open video
    # Note: H.264 streams may emit "Invalid NAL unit size" warnings from ffmpeg
    # at the end of decoding. This is benign and does not indicate corrupted frames.
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
    logger.info(
        f"Window size: {window} frames, framestep: {framestep}, window_step: {window_step}"
    )

    if not exact_seek:
        logger.warning(
            "Using hybrid seek mode. If output quality looks wrong (seek artifacts), "
            "re-run with --exact-seek"
        )

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

    output_count = 0

    try:
        if exact_seek:
            # Exact seek mode: sequential reading with deque buffer
            output_count = _process_exact_seek(
                cap,
                output_dir,
                video_path,
                video_writer,
                window,
                framestep,
                window_step,
                output_format,
                width,
                height,
                total_frames,
            )
        else:
            # Hybrid seek mode: jump to each output position
            output_count = _process_hybrid_seek(
                cap,
                output_dir,
                video_path,
                video_writer,
                window,
                framestep,
                window_step,
                output_format,
                width,
                height,
                total_frames,
            )

    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()

    elapsed = time.time() - start_time
    logger.info(
        f"Complete: produced {output_count} output frames in {elapsed:.1f}s "
        f"({output_count / elapsed:.1f} frames/s)"
    )
    return output_count


def _process_hybrid_seek(
    cap,
    output_dir,
    video_path,
    video_writer,
    window,
    framestep,
    window_step,
    output_format,
    width,
    height,
    total_frames,
) -> int:
    """Process video using hybrid seek mode (jump to each output position)."""
    output_count = 0
    effective_window_size = (window + window_step - 1) // window_step
    frame_stack = np.empty((effective_window_size, height, width, 3), dtype=np.uint8)

    # Compute output positions
    output_positions = range(window - 1, total_frames, framestep)

    for out_frame_idx in output_positions:
        # Compute window start
        seek_pos = max(0, out_frame_idx - window + 1)

        # Seek to window start
        cap.set(cv2.CAP_PROP_POS_FRAMES, seek_pos)

        # Read frames within window with subsampling
        frames_collected = []
        for i in range(window):
            ret, frame = cap.read()
            if not ret:
                break

            # Apply window_step subsampling
            if i % window_step == 0:
                frames_collected.append(frame)

        if not frames_collected:
            continue

        # Copy into pre-allocated array
        num_frames = len(frames_collected)
        for i, frame in enumerate(frames_collected):
            frame_stack[i] = frame

        # Compute median
        median_frame = np.median(frame_stack[:num_frames], axis=0).astype(np.uint8)

        # Output
        if output_format == "png":
            output_path = output_dir / f"frame_{out_frame_idx:06d}.png"
            cv2.imwrite(str(output_path), median_frame)
        elif output_format == "mp4":
            video_writer.write(median_frame)

        output_count += 1

        if output_count % 10 == 0:
            logger.info(
                f"Processed output frame {output_count} (source frame {out_frame_idx}/{total_frames})"
            )

    return output_count


def _process_exact_seek(
    cap,
    output_dir,
    video_path,
    video_writer,
    window,
    framestep,
    window_step,
    output_format,
    width,
    height,
    total_frames,
) -> int:
    """Process video using exact seek mode (sequential reading with deque buffer)."""
    output_count = 0
    effective_window_size = (window + window_step - 1) // window_step
    buffer = deque(maxlen=effective_window_size)
    frame_stack = np.empty((effective_window_size, height, width, 3), dtype=np.uint8)

    frame_idx = 0
    frames_in_current_window = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply window_step subsampling
        if frames_in_current_window % window_step == 0:
            buffer.append(frame)

        frames_in_current_window += 1

        # Reset window frame counter at window boundaries
        if frames_in_current_window >= window:
            frames_in_current_window = 0

        # Compute median when buffer is full and framestep matches
        if len(buffer) == effective_window_size and frame_idx % framestep == 0:
            # Copy from deque into pre-allocated array
            for i, f in enumerate(buffer):
                frame_stack[i] = f

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

    return output_count


def process_batch(
    input_path: Path,
    output_dir: Path | None = None,
    window: int = 30,
    framestep: int = 1,
    output_format: str = "png",
    exact_seek: bool = False,
    window_step: int = 1,
) -> dict[str, int]:
    """Process a single video or batch of videos with temporal median filtering.

    Args:
        input_path: Video file or directory containing videos.
        output_dir: Output directory. If None, defaults to parent of input_path
            (single file) or {input_path.parent}/{input_path.name}_median (directory).
        window: Median window size in frames.
        framestep: Output every Nth frame.
        output_format: Output format ("png" or "mp4").
        exact_seek: Force sequential reading (default: False = hybrid seek mode).
        window_step: Sample every Nth frame within the median window (default: 1).

    Returns:
        Dictionary mapping video stem to output frame count.
    """
    results = {}

    if input_path.is_file():
        # Single file mode
        if output_dir is None:
            if output_format == "png":
                output_dir = input_path.parent / input_path.stem
            else:
                output_dir = input_path.parent
        elif output_format == "png":
            output_dir = output_dir / input_path.stem

        count = process_video_temporal_median(
            video_path=input_path,
            output_dir=output_dir,
            window=window,
            framestep=framestep,
            output_format=output_format,
            exact_seek=exact_seek,
            window_step=window_step,
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
            # PNG sequences need per-video subdirectories; mp4 files go
            # directly into output_dir (named by video stem already).
            if output_format == "png":
                video_output_dir = output_dir / video_file.stem
            else:
                video_output_dir = output_dir

            try:
                count = process_video_temporal_median(
                    video_path=video_file,
                    output_dir=video_output_dir,
                    window=window,
                    framestep=framestep,
                    output_format=output_format,
                    exact_seek=exact_seek,
                    window_step=window_step,
                )
                results[video_file.stem] = count
            except Exception as e:
                logger.error(f"Failed to process {video_file.name}: {e}")
                results[video_file.stem] = 0

    else:
        raise ValueError(f"Input path is neither file nor directory: {input_path}")

    return results
