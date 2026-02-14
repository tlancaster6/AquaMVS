"""ROI mask loading and application for feature and depth filtering."""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from .calibration import CameraData

logger = logging.getLogger(__name__)


def load_mask(
    mask_dir: Path,
    camera_name: str,
    expected_size: tuple[int, int],
) -> np.ndarray | None:
    """Load a per-camera ROI mask from disk.

    Looks for {mask_dir}/{camera_name}.png. Returns None if not found or if
    size mismatch. Masks are single-channel images (255 = valid, 0 = excluded)
    at undistorted image resolution.

    Args:
        mask_dir: Directory containing mask PNGs.
        camera_name: Camera identifier (e.g., "e3v83eb").
        expected_size: Expected image size as (width, height) tuple.

    Returns:
        Grayscale mask array (H, W) uint8, or None if not found or invalid.
    """
    mask_path = mask_dir / f"{camera_name}.png"
    if not mask_path.exists():
        logger.debug("Mask not found for %s: %s", camera_name, mask_path)
        return None

    # Load as grayscale
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.warning(
            "Failed to load mask for %s from %s (invalid image)", camera_name, mask_path
        )
        return None

    # Validate size (H, W) vs (width, height)
    width, height = expected_size
    if mask.shape != (height, width):
        logger.warning(
            "Mask size mismatch for %s: expected (%d, %d), got %s. Ignoring mask.",
            camera_name,
            height,
            width,
            mask.shape,
        )
        return None

    return mask


def load_all_masks(
    mask_dir: str | Path | None,
    cameras: dict[str, CameraData],
) -> dict[str, np.ndarray]:
    """Load ROI masks for all cameras.

    For each camera, looks for {mask_dir}/{camera_name}.png. Missing masks
    are silently skipped (returns empty dict entry). Size mismatches generate
    warnings.

    Args:
        mask_dir: Directory containing mask PNGs, or None to disable masking.
        cameras: Per-camera calibration data (for image_size validation).

    Returns:
        Dict of camera_name -> mask array (H, W) uint8. Excludes cameras with
        missing or invalid masks.
    """
    if mask_dir is None:
        return {}

    mask_dir = Path(mask_dir)
    if not mask_dir.exists():
        logger.warning("Mask directory does not exist: %s", mask_dir)
        return {}

    masks = {}
    for camera_name, camera_data in cameras.items():
        mask = load_mask(mask_dir, camera_name, camera_data.image_size)
        if mask is not None:
            masks[camera_name] = mask

    if masks:
        logger.info("Loaded %d mask(s) from %s", len(masks), mask_dir)

    return masks


def apply_mask_to_features(
    features: dict[str, torch.Tensor],
    mask: np.ndarray,
) -> dict[str, torch.Tensor]:
    """Filter keypoints by ROI mask.

    Removes keypoints that fall outside the valid region (mask value 0).
    Keypoint coordinates are clamped to image bounds before sampling.

    Args:
        features: Dict with keys:
            "keypoints": shape (N, 2), float32 -- pixel coordinates (u, v)
            "descriptors": shape (N, D), float32 -- feature descriptors
            "scores": shape (N,), float32 -- detection confidence scores
        mask: Binary mask (H, W) uint8, where nonzero = valid region.

    Returns:
        Filtered features dict with same keys but reduced N. If all keypoints
        are masked out, returns tensors with N=0.
    """
    keypoints = features["keypoints"]  # (N, 2)
    descriptors = features["descriptors"]  # (N, D)
    scores = features["scores"]  # (N,)

    N = keypoints.shape[0]
    if N == 0:
        return features  # Already empty

    H, W = mask.shape

    # Convert keypoints to integer indices with clamping
    u = keypoints[:, 0].cpu().numpy()  # column
    v = keypoints[:, 1].cpu().numpy()  # row

    v_int = np.clip(np.round(v).astype(int), 0, H - 1)
    u_int = np.clip(np.round(u).astype(int), 0, W - 1)

    # Sample mask at keypoint locations
    mask_values = mask[v_int, u_int]

    # Keep keypoints where mask > 0
    valid = mask_values > 0

    # Filter tensors
    filtered_keypoints = keypoints[valid]
    filtered_descriptors = descriptors[valid]
    filtered_scores = scores[valid]

    return {
        "keypoints": filtered_keypoints,
        "descriptors": filtered_descriptors,
        "scores": filtered_scores,
    }


def apply_mask_to_depth(
    depth_map: torch.Tensor,
    confidence: torch.Tensor,
    mask: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ROI mask to depth map by setting excluded pixels to invalid.

    Pixels outside the mask (mask value 0) are set to NaN depth and 0 confidence.
    Operates on a copy; does not modify inputs in-place.

    Args:
        depth_map: Ray depth map (H, W) float32.
        confidence: Confidence map (H, W) float32.
        mask: Binary mask (H, W) uint8, where nonzero = valid region.

    Returns:
        Tuple of (masked_depth, masked_confidence). Both are clones of the
        inputs with excluded pixels set to NaN/0.
    """
    # Convert mask to boolean tensor on same device as depth_map
    mask_tensor = torch.from_numpy(mask).to(depth_map.device) > 0  # (H, W) bool

    # Clone inputs to avoid in-place modification
    masked_depth = depth_map.clone()
    masked_confidence = confidence.clone()

    # Set excluded pixels to invalid
    masked_depth[~mask_tensor] = float("nan")
    masked_confidence[~mask_tensor] = 0.0

    return masked_depth, masked_confidence


__all__ = [
    "load_mask",
    "load_all_masks",
    "apply_mask_to_features",
    "apply_mask_to_depth",
]
