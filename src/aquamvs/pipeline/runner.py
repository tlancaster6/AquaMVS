"""Pipeline runner: orchestrates stage execution and provides public API."""

import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from aquacal.io.video import VideoSet
from tqdm import tqdm

from ..config import PipelineConfig
from ..io import ImageDirectorySet, detect_input_type
from .builder import build_pipeline_context
from .context import PipelineContext
from .stages.dense_matching import run_roma_full_path, run_roma_sparse_path
from .stages.depth_estimation import run_depth_estimation
from .stages.fusion import run_fusion_stage
from .stages.sparse_matching import run_lightglue_path, run_triangulation
from .stages.surface import run_sparse_surface_stage, run_surface_stage
from .stages.undistortion import run_undistortion_stage
from .visualization import run_visualization_pass

logger = logging.getLogger(__name__)


def process_frame(
    frame_idx: int,
    raw_images: dict[str, np.ndarray],
    ctx: PipelineContext,
) -> None:
    """Process a single frame through the full reconstruction pipeline.

    Runs all stages sequentially, saving outputs to the frame's output
    directory. Each stage logs its completion. Visualization calls are
    gated by VizConfig and output persistence by OutputConfig.

    Args:
        frame_idx: Frame index (for output directory naming).
        raw_images: Camera name to raw BGR image (H, W, 3) uint8 mapping.
            May contain None values for cameras that failed to read.
        ctx: Precomputed pipeline context from setup_pipeline().
    """
    config = ctx.config

    # Create frame output directory
    frame_dir = Path(config.output_dir) / f"frame_{frame_idx:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage 1: Undistortion ---
    undistorted, undistorted_tensors, camera_centers = run_undistortion_stage(
        raw_images, ctx, frame_idx, frame_dir=frame_dir
    )
    if not undistorted:
        return

    # --- Dispatch on matcher type ---
    if config.matcher_type == "roma":
        # --- RoMa path: branch on pipeline_mode ---
        if config.pipeline_mode == "full":
            # --- RoMa + full: warps -> depth maps -> fusion -> surface ---
            depth_maps, confidence_maps = run_roma_full_path(
                undistorted_tensors, ctx, frame_dir, frame_idx
            )

            # Jump to fusion (skip triangulation, depth ranges, plane sweep)
            fused = run_fusion_stage(
                depth_maps,
                confidence_maps,
                undistorted,
                ctx,
                frame_dir,
                frame_idx,
                skip_filter=True,
            )

            # Surface reconstruction (full mode)
            run_surface_stage(
                fused, undistorted_tensors, camera_centers, ctx, frame_dir, frame_idx
            )

        else:
            # --- RoMa + sparse: correspondences -> triangulation -> sparse surface ---
            result = run_roma_sparse_path(
                undistorted_tensors, ctx, frame_dir, frame_idx
            )

            # Triangulation
            sparse_cloud, depth_ranges = run_triangulation(
                result["all_matches"], ctx, frame_dir, frame_idx
            )

            # Sparse surface reconstruction
            run_sparse_surface_stage(
                sparse_cloud,
                undistorted_tensors,
                camera_centers,
                ctx,
                frame_dir,
                frame_idx,
            )

    elif config.matcher_type == "lightglue":
        # --- LightGlue path: extraction + matching ---
        result = run_lightglue_path(undistorted_tensors, ctx, frame_dir, frame_idx)

        # Triangulation
        sparse_cloud, depth_ranges = run_triangulation(
            result["all_matches"], ctx, frame_dir, frame_idx
        )

        # --- Sparse mode branch ---
        if config.pipeline_mode == "sparse":
            # Sparse surface reconstruction
            run_sparse_surface_stage(
                sparse_cloud,
                undistorted_tensors,
                camera_centers,
                ctx,
                frame_dir,
                frame_idx,
            )
            return

        # --- Full mode: dense stereo + fusion + surface ---
        # Stage 6: Dense Stereo
        depth_maps, confidence_maps = run_depth_estimation(
            undistorted_tensors, depth_ranges, ctx, frame_dir, frame_idx
        )

        # Stage 7-8: Fusion
        fused = run_fusion_stage(
            depth_maps,
            confidence_maps,
            undistorted,
            ctx,
            frame_dir,
            frame_idx,
        )

        # Stage 9: Surface reconstruction
        run_surface_stage(
            fused, undistorted_tensors, camera_centers, ctx, frame_dir, frame_idx
        )

    logger.info("Frame %d: complete", frame_idx)


def run_pipeline(config: PipelineConfig) -> None:
    """Run the full reconstruction pipeline over video frames.

    Uses a two-pass architecture to avoid Open3D OpenGL / CUDA GPU memory
    conflicts on Windows:

    1. **Compute pass** — matching, depth estimation, fusion, surface
       reconstruction. All outputs saved to disk.
    2. **Viz pass** — reload saved artifacts from disk, render all
       visualizations.

    Between passes, ``torch.cuda.empty_cache()`` frees GPU memory so
    Open3D's OpenGL context can allocate without competing with CUDA.

    Args:
        config: Full pipeline configuration.
    """
    # One-time setup
    ctx = build_pipeline_context(config)

    # --- Compute pass ---
    input_type = detect_input_type(config.camera_input_map)
    if input_type == "images":
        logger.info("Detected image directory input")
        context_manager = ImageDirectorySet(config.camera_input_map)
    else:
        logger.info("Opening video files")
        context_manager = VideoSet(config.camera_input_map)

    with context_manager as videos:
        logger.info(
            "Processing frames %d to %s (step %d)",
            config.preprocessing.frame_start,
            config.preprocessing.frame_stop or "end",
            config.preprocessing.frame_step,
        )

        for frame_idx, raw_images in tqdm(
            videos.iterate_frames(
                start=config.preprocessing.frame_start,
                stop=config.preprocessing.frame_stop,
                step=config.preprocessing.frame_step,
            ),
            desc="Processing frames",
            disable=config.runtime.quiet or not sys.stderr.isatty(),
            unit="frame",
        ):
            try:
                process_frame(frame_idx, raw_images, ctx)
            except Exception:
                logger.exception("Frame %d: processing failed, skipping", frame_idx)
                continue

    # --- Free GPU memory before viz ---
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Viz pass ---
    if config.runtime.viz_enabled:
        run_visualization_pass(config, ctx)

    # --- Deferred cleanup ---
    if not config.runtime.keep_intermediates:
        _cleanup_intermediates(config)

    logger.info("Pipeline complete")


def _cleanup_intermediates(config: PipelineConfig) -> None:
    """Remove intermediate depth maps from all frame directories.

    Args:
        config: Pipeline configuration (for output_dir).
    """
    output_dir = Path(config.output_dir)
    for frame_dir in sorted(output_dir.glob("frame_*")):
        depth_dir = frame_dir / "depth_maps"
        if depth_dir.exists():
            shutil.rmtree(depth_dir)
            logger.debug("Removed intermediate depth maps from %s", frame_dir.name)
        undist_dir = frame_dir / "undistorted"
        if undist_dir.exists():
            shutil.rmtree(undist_dir)
            logger.debug(
                "Removed intermediate undistorted images from %s", frame_dir.name
            )


class Pipeline:
    """Multi-view stereo reconstruction pipeline.

    Primary programmatic entry point for AquaMVS.

    Example:
        pipeline = Pipeline(config)
        pipeline.run()
    """

    def __init__(self, config: PipelineConfig):
        """Initialize the pipeline with configuration.

        Args:
            config: Full pipeline configuration.
        """
        self.config = config
        self.context = None

    def run(self) -> None:
        """Run the full reconstruction pipeline.

        Equivalent to calling run_pipeline(config).
        """
        # Delegate to run_pipeline
        run_pipeline(self.config)
