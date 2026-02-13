"""Feature extraction using SuperPoint."""

from pathlib import Path

import numpy as np
import torch
from lightglue import SuperPoint

from aquamvs.config import FeatureExtractionConfig


def create_extractor(
    config: FeatureExtractionConfig,
    device: str = "cpu",
) -> SuperPoint:
    """Create and initialize a SuperPoint feature extractor.

    Args:
        config: Feature extraction configuration.
        device: Device to place the model on.

    Returns:
        Initialized SuperPoint model in eval mode.
    """
    extractor = SuperPoint(
        max_num_keypoints=config.max_keypoints,
        detection_threshold=config.detection_threshold,
    ).eval()
    return extractor.to(device)


def extract_features(
    image: torch.Tensor | np.ndarray,
    config: FeatureExtractionConfig,
    extractor: SuperPoint | None = None,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Extract SuperPoint keypoints and descriptors from a single image.

    Args:
        image: Undistorted image as tensor or array. Supported formats:
            - torch.Tensor: (H, W, 3) or (H, W), uint8 or float32
            - np.ndarray: (H, W, 3) or (H, W), uint8 or float32
            If uint8 HWC, converted to float32 grayscale (1, 1, H, W) internally.
        config: Feature extraction configuration.
        extractor: Pre-initialized SuperPoint model. If None, creates one.
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
    # SuperPoint outputs: (1, N, 2), (1, N, 256), (1, N)
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
    config: FeatureExtractionConfig,
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
