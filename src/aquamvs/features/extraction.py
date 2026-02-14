"""Feature extraction using configurable detector backends."""

from pathlib import Path

import cv2
import numpy as np
import torch
from lightglue import ALIKED, DISK, SuperPoint

from aquamvs.config import SparseMatchingConfig


def create_extractor(
    config: SparseMatchingConfig,
    device: str = "cpu",
) -> torch.nn.Module:
    """Create and initialize a feature extractor.

    Args:
        config: Feature extraction configuration.
        device: Device to place the model on.

    Returns:
        Initialized extractor model (SuperPoint, ALIKED, or DISK) in eval mode.

    Raises:
        ValueError: If extractor_type is not recognized.
    """
    if config.extractor_type == "superpoint":
        extractor = SuperPoint(
            max_num_keypoints=config.max_keypoints,
            detection_threshold=config.detection_threshold,
        )
    elif config.extractor_type == "aliked":
        extractor = ALIKED(
            max_num_keypoints=config.max_keypoints,
            detection_threshold=config.detection_threshold,
        )
    elif config.extractor_type == "disk":
        extractor = DISK(
            max_num_keypoints=config.max_keypoints,
            detection_threshold=config.detection_threshold,
        )
    else:
        raise ValueError(
            f"Unknown extractor_type: {config.extractor_type!r}. "
            "Valid types: 'superpoint', 'aliked', 'disk'"
        )

    return extractor.eval().to(device)


def _apply_clahe(
    gray: torch.Tensor,
    clip_limit: float,
) -> torch.Tensor:
    """Apply CLAHE preprocessing to grayscale image.

    Args:
        gray: Grayscale tensor of shape (H, W). May be uint8 or float32.
        clip_limit: Contrast limit for CLAHE (higher = more enhancement).

    Returns:
        Enhanced grayscale tensor with same dtype and device as input.
    """
    # Remember original dtype and device
    original_dtype = gray.dtype
    original_device = gray.device

    # Convert to numpy uint8 for OpenCV CLAHE
    if gray.dtype.is_floating_point:
        # If float and max > 1.0, assume [0, 255] range
        if gray.max() > 1.0:
            gray_uint8 = torch.clamp(gray, 0, 255).cpu().numpy().astype(np.uint8)
        else:
            # If float and max <= 1.0, assume [0, 1] range
            gray_uint8 = (torch.clamp(gray, 0, 1) * 255).cpu().numpy().astype(np.uint8)
    else:
        # Already uint8
        gray_uint8 = gray.cpu().numpy().astype(np.uint8)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_uint8)

    # Convert back to torch tensor
    result = torch.from_numpy(enhanced)

    # Convert back to original dtype
    if original_dtype.is_floating_point:
        if gray.max() > 1.0:
            # Return float in [0, 255] range
            result = result.float()
        else:
            # Return float in [0, 1] range
            result = result.float() / 255.0
    else:
        # Return uint8
        result = result.to(torch.uint8)

    # Move to original device
    result = result.to(original_device)

    return result


def extract_features(
    image: torch.Tensor | np.ndarray,
    config: SparseMatchingConfig,
    extractor: torch.nn.Module | None = None,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Extract keypoints and descriptors from a single image.

    Args:
        image: Undistorted image as tensor or array. Supported formats:
            - torch.Tensor: (H, W, 3) or (H, W), uint8 or float32
            - np.ndarray: (H, W, 3) or (H, W), uint8 or float32
            If uint8 HWC, converted to float32 grayscale (1, 1, H, W) internally.
        config: Feature extraction configuration.
        extractor: Pre-initialized extractor model. If None, creates one using config.extractor_type.
        device: Device string for the extractor (used only if extractor is None).

    Returns:
        Dict with keys:
            "keypoints": shape (N, 2), float32 -- pixel coordinates (u, v)
            "descriptors": shape (N, D), float32 -- feature descriptors
            "scores": shape (N,), float32 -- detection confidence scores
    """
    # Convert numpy to tensor if needed
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    # Track if we need to normalize from [0, 255] to [0, 1]
    needs_normalization = image.dtype == torch.uint8

    # Convert to grayscale float32 (1, 1, H, W) in [0, 1]
    if image.dim() == 3 and image.shape[-1] == 3:
        # HWC format with 3 channels - convert BGR to grayscale
        # Standard luminance formula: 0.299*R + 0.587*G + 0.114*B
        # OpenCV uses BGR order, so indices are [2, 1, 0] for [R, G, B]
        gray = 0.299 * image[..., 2] + 0.587 * image[..., 1] + 0.114 * image[..., 0]
    elif image.dim() == 2:
        # Already grayscale HW
        gray = image
    else:
        raise ValueError(
            f"Expected image with shape (H, W) or (H, W, 3), got {image.shape}"
        )

    # Optional CLAHE preprocessing
    if config.clahe_enabled:
        gray = _apply_clahe(gray, config.clahe_clip_limit)

    # Ensure float type
    if not gray.dtype.is_floating_point:
        gray = gray.float()

    # Normalize to [0, 1] if input was uint8
    if needs_normalization:
        gray = gray / 255.0

    # Reshape to (1, 1, H, W) batch format
    gray = gray.unsqueeze(0).unsqueeze(0)

    # Move to device
    gray = gray.to(device)

    # Create extractor if not provided
    if extractor is None:
        extractor = create_extractor(config, device)

    # Extract features with no gradient tracking
    with torch.no_grad():
        feats = extractor.extract(gray)

    # Squeeze batch dimension and rename keys
    # All extractors output: keypoints (1, N, 2), descriptors (1, N, D), keypoint_scores (1, N)
    keypoints = feats["keypoints"].squeeze(0)  # (N, 2)
    descriptors = feats["descriptors"].squeeze(0)  # (N, D)
    scores = feats["keypoint_scores"].squeeze(0)  # (N,)

    return {
        "keypoints": keypoints,
        "descriptors": descriptors,
        "scores": scores,
    }


def extract_features_batch(
    images: dict[str, torch.Tensor | np.ndarray],
    config: SparseMatchingConfig,
    device: str = "cpu",
) -> dict[str, dict[str, torch.Tensor]]:
    """Extract features from multiple images.

    Args:
        images: Camera name to image tensor/array mapping.
        config: Feature extraction configuration.
        device: Device for the extractor.

    Returns:
        Camera name to features dict mapping.
    """
    # Create extractor once for all images
    extractor = create_extractor(config, device)

    # Extract features for each image
    features = {}
    for camera_name, image in images.items():
        features[camera_name] = extract_features(
            image, config, extractor=extractor, device=device
        )

    return features


def save_features(features: dict[str, torch.Tensor], path: str | Path) -> None:
    """Save features dict to a .pt file.

    Args:
        features: Features dict with "keypoints", "descriptors", "scores".
        path: Output file path (should end with .pt).
    """
    torch.save(features, path)


def load_features(path: str | Path) -> dict[str, torch.Tensor]:
    """Load features dict from a .pt file.

    Args:
        path: Path to .pt file.

    Returns:
        Features dict with "keypoints", "descriptors", "scores".
    """
    return torch.load(path, weights_only=True)
