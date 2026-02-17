"""Undistortion and color normalization stage."""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from ...calibration import undistort_image
from ..context import PipelineContext
from ..helpers import _should_viz

logger = logging.getLogger(__name__)


def run_undistortion_stage(
    raw_images: dict[str, np.ndarray],
    ctx: PipelineContext,
    frame_idx: int,
    frame_dir: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, torch.Tensor], dict[str, np.ndarray]]:
    """Undistort images and optionally apply color normalization.

    When features visualization is active and *frame_dir* is provided,
    undistorted images are saved to ``frame_dir/undistorted/{cam}.png``
    so the viz pass can reload them from disk.

    Args:
        raw_images: Camera name to raw BGR image (H, W, 3) uint8 mapping.
            May contain None values for cameras that failed to read.
        ctx: Pipeline context with config, undistortion maps, calibration.
        frame_idx: Frame index (for logging).
        frame_dir: Frame output directory (optional, needed for saving undistorted images).

    Returns:
        Tuple of (undistorted_numpy, undistorted_tensors, camera_centers):
            - undistorted_numpy: Dict of undistorted BGR images (H, W, 3) uint8.
            - undistorted_tensors: Dict of undistorted BGR images as torch.Tensor.
            - camera_centers: Dict of camera positions in world frame, shape (3,) float64.
    """
    from ...profiling import timed_stage

    with timed_stage("undistortion", logger):
        config = ctx.config

        # Filter out None images (cameras that failed to read)
        images = {name: img for name, img in raw_images.items() if img is not None}
        if not images:
            logger.warning("Frame %d: no valid images, skipping", frame_idx)
            return {}, {}, {}

        # --- Stage 1: Undistort ---
        logger.info("Frame %d: undistorting images", frame_idx)
        undistorted = {}
        for name, img in images.items():
            if name in ctx.undistortion_maps:
                undistorted[name] = undistort_image(img, ctx.undistortion_maps[name])

        # --- Stage 1b: Color normalization (if enabled) ---
        if config.preprocessing.color_norm_enabled:
            from ...coloring import normalize_colors

            logger.info("Frame %d: normalizing colors across cameras", frame_idx)
            undistorted = normalize_colors(
                undistorted, method=config.preprocessing.color_norm_method
            )

        # Save undistorted images when features viz is active (viz pass needs them)
        if frame_dir is not None and _should_viz(config, "features"):
            undist_dir = frame_dir / "undistorted"
            undist_dir.mkdir(exist_ok=True)
            for name, img in undistorted.items():
                cv2.imwrite(str(undist_dir / f"{name}.png"), img)

        # Convert to tensors for feature extraction
        undistorted_tensors = {
            name: torch.from_numpy(img) for name, img in undistorted.items()
        }

        # Compute camera centers once for coloring (used by sparse cloud and mesh coloring)
        camera_centers = {
            name: pos.cpu().numpy()
            for name, pos in ctx.calibration.camera_positions().items()
        }

        return undistorted, undistorted_tensors, camera_centers
