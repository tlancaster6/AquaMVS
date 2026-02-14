"""Feature matching using LightGlue."""

from pathlib import Path

import torch
from lightglue import LightGlue

from ..config import SparseMatchingConfig


def create_matcher(
    extractor_type: str = "superpoint", device: str = "cpu"
) -> LightGlue:
    """Create and initialize a LightGlue feature matcher.

    Args:
        extractor_type: Feature extractor backend ("superpoint", "aliked", or "disk").
        device: Device to place the model on.

    Returns:
        Initialized LightGlue model in eval mode, configured for the specified extractor.
    """
    matcher = LightGlue(features=extractor_type).eval()
    return matcher.to(device)


def _prepare_lightglue_input(
    feats: dict[str, torch.Tensor],
    image_size: tuple[int, int],
    device: str,
) -> dict[str, torch.Tensor]:
    """Convert P.10 features dict to LightGlue input format.

    Args:
        feats: Features dict with "keypoints" (N, 2), "descriptors" (N, D), "scores" (N,).
        image_size: Image dimensions as (width, height) in pixels.
        device: Device to place tensors on.

    Returns:
        Dict with LightGlue-compatible format:
            "keypoints": (1, N, 2)
            "descriptors": (1, N, D)
            "keypoint_scores": (1, N,)
            "image_size": (1, 2) as (height, width)
    """
    w, h = image_size
    return {
        "keypoints": feats["keypoints"].unsqueeze(0).to(device),
        "descriptors": feats["descriptors"].unsqueeze(0).to(device),
        "keypoint_scores": feats["scores"].unsqueeze(0).to(device),
        "image_size": torch.tensor([[h, w]], device=device),  # (H, W) order
    }


def match_pair(
    feats_ref: dict[str, torch.Tensor],
    feats_src: dict[str, torch.Tensor],
    image_size: tuple[int, int],
    config: SparseMatchingConfig,
    matcher: LightGlue | None = None,
    device: str = "cpu",
    extractor_type: str = "superpoint",
) -> dict[str, torch.Tensor]:
    """Match features between a reference and source image pair.

    Args:
        feats_ref: Reference image features from extract_features().
            Keys: "keypoints" (N, 2), "descriptors" (N, D), "scores" (N,).
        feats_src: Source image features from extract_features().
            Same format as feats_ref.
        image_size: Image dimensions as (width, height) in pixels.
        config: Matching configuration.
        matcher: Pre-initialized LightGlue model. If None, creates one.
        device: Device string (used only if matcher is None).
        extractor_type: Feature extractor backend (used only if matcher is None).

    Returns:
        Dict with keys:
            "ref_keypoints": shape (M, 2), float32 -- matched reference pixel coords
            "src_keypoints": shape (M, 2), float32 -- matched source pixel coords
            "scores": shape (M,), float32 -- match confidence scores
        Where M is the number of matches passing the filter threshold.
        Returns empty tensors (M=0) if no matches pass the threshold.
    """
    # Create matcher if not provided
    if matcher is None:
        matcher = create_matcher(extractor_type, device)

    # Prepare LightGlue input format
    feats0 = _prepare_lightglue_input(feats_ref, image_size, device)
    feats1 = _prepare_lightglue_input(feats_src, image_size, device)

    # Match with no gradient tracking
    with torch.no_grad():
        result = matcher({"image0": feats0, "image1": feats1})

    # Extract matches and scores, remove batch dimension
    matches0 = result["matches0"].squeeze(0)  # (N_ref,) -- index into src, or -1
    scores0 = result["matching_scores0"].squeeze(0)  # (N_ref,)

    # Filter: matched (index >= 0) AND above threshold
    matched_mask = (matches0 >= 0) & (scores0 >= config.filter_threshold)

    # Get indices
    ref_indices = torch.where(matched_mask)[0]  # indices into ref keypoints
    src_indices = matches0[matched_mask]  # corresponding src indices

    # Gather matched keypoints
    ref_kpts = feats_ref["keypoints"].to(device)[ref_indices]  # (M, 2)
    src_kpts = feats_src["keypoints"].to(device)[src_indices]  # (M, 2)
    match_scores = scores0[matched_mask]  # (M,)

    return {
        "ref_keypoints": ref_kpts,
        "src_keypoints": src_kpts,
        "scores": match_scores,
    }


def match_all_pairs(
    all_features: dict[str, dict[str, torch.Tensor]],
    pairs: dict[str, list[str]],
    image_size: tuple[int, int],
    config: SparseMatchingConfig,
    device: str = "cpu",
    extractor_type: str = "superpoint",
) -> dict[tuple[str, str], dict[str, torch.Tensor]]:
    """Match features for all selected camera pairs.

    Args:
        all_features: Camera name to features dict mapping (from extract_features_batch).
        pairs: Reference camera to source camera list mapping (from select_pairs).
        image_size: Image dimensions as (width, height) in pixels.
        config: Matching configuration.
        device: Device for the matcher.
        extractor_type: Feature extractor backend.

    Returns:
        Dict mapping (ref_camera, src_camera) tuple to matches dict.
        Pairs with zero matches after filtering are included with empty tensors.
        Each unordered pair {A, B} appears exactly once with canonical key (min(A,B), max(A,B)).
    """
    # Create matcher once for all pairs
    matcher = create_matcher(extractor_type, device)

    # Match all pairs, avoiding duplicates from bidirectional pair lists
    all_matches = {}
    seen_pairs: set[tuple[str, str]] = set()

    for ref_cam, src_cams in pairs.items():
        for src_cam in src_cams:
            # Canonicalize pair order to avoid duplicate matching
            canonical = (min(ref_cam, src_cam), max(ref_cam, src_cam))
            if canonical in seen_pairs:
                continue
            seen_pairs.add(canonical)

            # Always match in canonical order (A < B)
            feats_a = all_features[canonical[0]]
            feats_b = all_features[canonical[1]]

            matches = match_pair(
                feats_a,
                feats_b,
                image_size,
                config,
                matcher=matcher,
                device=device,
            )

            all_matches[canonical] = matches

    return all_matches


def save_matches(matches: dict[str, torch.Tensor], path: str | Path) -> None:
    """Save matches dict to a .pt file.

    Args:
        matches: Matches dict with "ref_keypoints", "src_keypoints", "scores".
        path: Output file path (should end with .pt).
    """
    torch.save(matches, path)


def load_matches(path: str | Path) -> dict[str, torch.Tensor]:
    """Load matches dict from a .pt file.

    Args:
        path: Path to .pt file.

    Returns:
        Matches dict with "ref_keypoints", "src_keypoints", "scores".
    """
    return torch.load(path, weights_only=True)
