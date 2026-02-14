"""Dense matching using RoMa v2."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from romav2 import RoMaV2
from romav2.geometry import to_pixel

from ..config import DenseMatchingConfig


def create_roma_matcher(device: str = "cpu") -> RoMaV2:
    """Create and initialize a RoMa v2 dense matcher.

    Args:
        device: Device to place the model on.

    Returns:
        Initialized RoMa v2 model in eval mode.
    """
    # Disable compilation to avoid Triton issues on Windows
    cfg = RoMaV2.Cfg(compile=False)
    matcher = RoMaV2(cfg).eval()
    return matcher.to(device)


def _extract_correspondences(
    warp: torch.Tensor,
    overlap: torch.Tensor,
    H_A: int,
    W_A: int,
    H_B: int,
    W_B: int,
    config: DenseMatchingConfig,
    device: str,
) -> dict[str, torch.Tensor]:
    """Extract discrete correspondences from dense RoMa warp field.

    Args:
        warp: Dense warp field (H_warp, W_warp, 2) in normalized coords [-1, 1].
        overlap: Overlap certainty map (H_warp, W_warp).
        H_A: Height of reference image A in pixels.
        W_A: Width of reference image A in pixels.
        H_B: Height of source image B in pixels.
        W_B: Width of source image B in pixels.
        config: Dense matching configuration.
        device: Device for output tensors.

    Returns:
        Dict with keys:
            "ref_keypoints": (M, 2) float32 -- reference pixel coords (u, v)
            "src_keypoints": (M, 2) float32 -- source pixel coords (u, v)
            "scores": (M,) float32 -- correspondence certainty scores
    """
    # Threshold by certainty
    mask = overlap > config.certainty_threshold

    # Get warp grid coordinates
    H_warp, W_warp = warp.shape[:2]
    v_grid, u_grid = torch.meshgrid(
        torch.arange(H_warp, device=device),
        torch.arange(W_warp, device=device),
        indexing="ij",
    )

    # Get surviving pixel indices in warp grid
    v_ref_warp = v_grid[mask]  # (N,)
    u_ref_warp = u_grid[mask]  # (N,)

    # Convert warp grid coords to original image A coords
    # warp grid [0, W_warp) -> normalized [-1, 1] -> image A [0, W_A)
    u_ref_norm = (u_ref_warp.float() / W_warp) * 2 - 1
    v_ref_norm = (v_ref_warp.float() / H_warp) * 2 - 1

    u_ref = (u_ref_norm + 1) / 2 * W_A
    v_ref = (v_ref_norm + 1) / 2 * H_A

    ref_keypoints = torch.stack([u_ref, v_ref], dim=-1)  # (N, 2)

    # Get warp values at surviving pixels and convert to image B coords
    warp_values = warp[v_ref_warp, u_ref_warp]  # (N, 2) in normalized coords
    src_keypoints = to_pixel(warp_values, H=H_B, W=W_B)  # (N, 2)

    # Get certainty scores
    scores = overlap[v_ref_warp, u_ref_warp]  # (N,)

    # Subsample if too many correspondences using spatially uniform sampling.
    # Global top-K by certainty clusters in a tiny fraction of the image
    # because RoMa scores are tightly packed (e.g., 0.96-1.0). Instead,
    # divide the reference image into a grid and take the top-scoring
    # correspondences within each bin to preserve spatial coverage.
    if len(scores) > config.max_correspondences:
        n_bins = 32  # 32x32 grid over the reference image
        budget = config.max_correspondences

        # Assign each correspondence to a spatial bin
        u_bin = (ref_keypoints[:, 0] / W_A * n_bins).long().clamp(0, n_bins - 1)
        v_bin = (ref_keypoints[:, 1] / H_A * n_bins).long().clamp(0, n_bins - 1)
        bin_idx = v_bin * n_bins + u_bin  # flat bin index

        # Count correspondences per bin and allocate budget proportionally
        n_total_bins = n_bins * n_bins
        bin_counts = torch.zeros(n_total_bins, dtype=torch.long, device=device)
        bin_counts.scatter_add_(0, bin_idx, torch.ones_like(bin_idx))
        occupied = bin_counts > 0
        n_occupied = occupied.sum().item()

        # Equal allocation per occupied bin, distribute remainder
        per_bin = budget // max(n_occupied, 1)
        remainder = budget - per_bin * n_occupied
        bin_budget = torch.zeros(n_total_bins, dtype=torch.long, device=device)
        bin_budget[occupied] = per_bin
        # Give +1 to the first `remainder` occupied bins
        if remainder > 0:
            occupied_indices = torch.where(occupied)[0][:remainder]
            bin_budget[occupied_indices] += 1

        # Select top-scoring correspondences within each bin
        selected_indices = []
        for b in torch.where(occupied)[0]:
            in_bin = (bin_idx == b).nonzero(as_tuple=True)[0]
            k = min(bin_budget[b].item(), len(in_bin))
            if k > 0:
                top_in_bin = in_bin[torch.topk(scores[in_bin], k=k).indices]
                selected_indices.append(top_in_bin)

        if selected_indices:
            selected = torch.cat(selected_indices)
            ref_keypoints = ref_keypoints[selected]
            src_keypoints = src_keypoints[selected]
            scores = scores[selected]

    return {
        "ref_keypoints": ref_keypoints,
        "src_keypoints": src_keypoints,
        "scores": scores,
    }


def _run_roma(
    img_ref: torch.Tensor,
    img_src: torch.Tensor,
    matcher: RoMaV2,
) -> dict:
    """Run RoMa on an image pair and return raw results.

    Args:
        img_ref: Reference image tensor (H, W, 3) uint8 or (C, H, W) float32.
        img_src: Source image tensor, same format as img_ref.
        matcher: Pre-initialized RoMa v2 model.

    Returns:
        Dict with keys:
            "warp_AB": (H_warp, W_warp, 2) -- dense warp in normalized [-1, 1]
            "overlap_AB": (H_warp, W_warp) -- overlap certainty
            "H_ref", "W_ref", "H_src", "W_src": int -- original image dims
    """
    # Get image dimensions (handle both HWC and CHW formats)
    if img_ref.ndim == 3 and img_ref.shape[-1] == 3:
        # HWC format
        H_ref, W_ref = img_ref.shape[:2]
        H_src, W_src = img_src.shape[:2]
        # Convert to numpy for PIL
        img_ref_np = img_ref.cpu().numpy()
        img_src_np = img_src.cpu().numpy()
    else:
        # CHW format
        H_ref, W_ref = img_ref.shape[-2:]
        H_src, W_src = img_src.shape[-2:]
        # Convert to HWC numpy for PIL
        img_ref_np = img_ref.permute(1, 2, 0).cpu().numpy()
        img_src_np = img_src.permute(1, 2, 0).cpu().numpy()

    # Convert to PIL Images (RoMa expects PIL or path strings)
    if img_ref_np.dtype == np.float32 or img_ref_np.dtype == np.float64:
        img_ref_np = (img_ref_np * 255).astype(np.uint8)
    if img_src_np.dtype == np.float32 or img_src_np.dtype == np.float64:
        img_src_np = (img_src_np * 255).astype(np.uint8)

    img_ref_pil = Image.fromarray(img_ref_np)
    img_src_pil = Image.fromarray(img_src_np)

    # Match with RoMa
    with torch.no_grad():
        result = matcher.match(img_ref_pil, img_src_pil)

    # Extract warp and overlap, remove batch dimensions
    warp_AB = result["warp_AB"][0]  # (H_warp, W_warp, 2)
    overlap_AB = result["overlap_AB"][0, ..., 0]  # (H_warp, W_warp)

    return {
        "warp_AB": warp_AB,
        "overlap_AB": overlap_AB,
        "H_ref": H_ref,
        "W_ref": W_ref,
        "H_src": H_src,
        "W_src": W_src,
    }


def match_pair_roma(
    img_ref: torch.Tensor,
    img_src: torch.Tensor,
    config: DenseMatchingConfig,
    matcher: RoMaV2 | None = None,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Match a reference and source image pair using RoMa v2.

    Args:
        img_ref: Reference image tensor (H, W, 3) uint8 or (C, H, W) float32.
        img_src: Source image tensor, same format as img_ref.
        config: Dense matching configuration.
        matcher: Pre-initialized RoMa v2 model. If None, creates one.
        device: Device string (used only if matcher is None).

    Returns:
        Dict with keys:
            "ref_keypoints": (M, 2) float32 -- matched reference pixel coords
            "src_keypoints": (M, 2) float32 -- matched source pixel coords
            "scores": (M,) float32 -- match confidence scores
        Where M is the number of correspondences after thresholding and subsampling.
    """
    # Create matcher if not provided
    if matcher is None:
        matcher = create_roma_matcher(device)

    # Run RoMa
    roma_result = _run_roma(img_ref, img_src, matcher)

    # Extract correspondences from dense warp
    correspondences = _extract_correspondences(
        warp=roma_result["warp_AB"],
        overlap=roma_result["overlap_AB"],
        H_A=roma_result["H_ref"],
        W_A=roma_result["W_ref"],
        H_B=roma_result["H_src"],
        W_B=roma_result["W_src"],
        config=config,
        device=next(iter(matcher.parameters())).device,
    )

    return correspondences


def apply_mask_to_correspondences(
    correspondences: dict[str, torch.Tensor],
    mask: np.ndarray,
) -> dict[str, torch.Tensor]:
    """Apply ROI mask to correspondences, filtering out-of-mask reference pixels.

    Args:
        correspondences: Matches dict with "ref_keypoints", "src_keypoints", "scores".
        mask: Binary mask (H, W) uint8, where 255 = valid region.

    Returns:
        Filtered correspondences dict with same keys.
    """
    ref_kpts = correspondences["ref_keypoints"]  # (M, 2)
    src_kpts = correspondences["src_keypoints"]  # (M, 2)
    scores = correspondences["scores"]  # (M,)

    # Convert keypoints to integer pixel coords for mask lookup
    u = ref_kpts[:, 0].long()
    v = ref_kpts[:, 1].long()

    # Clamp to image bounds
    H, W = mask.shape
    u = u.clamp(0, W - 1)
    v = v.clamp(0, H - 1)

    # Check mask values (255 = valid)
    mask_tensor = torch.from_numpy(mask).to(ref_kpts.device)
    valid_mask = mask_tensor[v, u] > 0

    # Filter correspondences
    return {
        "ref_keypoints": ref_kpts[valid_mask],
        "src_keypoints": src_kpts[valid_mask],
        "scores": scores[valid_mask],
    }


def run_roma_all_pairs(
    undistorted_images: dict[str, torch.Tensor],
    pairs: dict[str, list[str]],
    config: DenseMatchingConfig,
    device: str = "cpu",
    masks: dict[str, np.ndarray] | None = None,
) -> dict[tuple[str, str], dict]:
    """Run RoMa on all pairs, returning raw warps.

    Unlike match_all_pairs_roma which returns extracted correspondences with
    canonical pair deduplication, this function returns raw RoMa warps using
    directed keys. For each (ref, src) in pairs, the warp goes from ref to src.
    This enables warp-to-depth conversion where each reference camera needs
    warps pointing to its sources.

    Args:
        undistorted_images: Camera name to undistorted image tensor mapping.
        pairs: Reference camera to source camera list mapping (from select_pairs).
        config: Dense matching configuration (only certainty_threshold is used).
        device: Device for the matcher.
        masks: Optional per-camera ROI masks. Masks are NOT applied here
            (they are applied during depth conversion).

    Returns:
        Dict mapping directed (ref, src) tuple to raw warp dict from _run_roma().
        Each dict contains "warp_AB", "overlap_AB", "H_ref", "W_ref", "H_src", "W_src".
    """
    # Create matcher once for all pairs
    matcher = create_roma_matcher(device)

    all_warps = {}

    for ref_cam, src_cams in pairs.items():
        for src_cam in src_cams:
            # Use directed keys (ref, src) -- no deduplication
            # This ensures each reference camera has warps pointing to its sources
            img_ref = undistorted_images[ref_cam]
            img_src = undistorted_images[src_cam]

            warp_result = _run_roma(img_ref, img_src, matcher)
            all_warps[(ref_cam, src_cam)] = warp_result

    return all_warps


def match_all_pairs_roma(
    undistorted_images: dict[str, torch.Tensor],
    pairs: dict[str, list[str]],
    config: DenseMatchingConfig,
    device: str = "cpu",
    masks: dict[str, np.ndarray] | None = None,
) -> dict[tuple[str, str], dict[str, torch.Tensor]]:
    """Match all camera pairs using RoMa v2.

    Args:
        undistorted_images: Camera name to undistorted image tensor mapping.
        pairs: Reference camera to source camera list mapping (from select_pairs).
        config: Dense matching configuration.
        device: Device for the matcher.
        masks: Optional per-camera ROI masks (camera name to mask array).

    Returns:
        Dict mapping (ref_camera, src_camera) tuple to matches dict.
        Each unordered pair {A, B} appears exactly once with canonical key (min(A,B), max(A,B)).
    """
    # Create matcher once for all pairs
    matcher = create_roma_matcher(device)

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
            img_a = undistorted_images[canonical[0]]
            img_b = undistorted_images[canonical[1]]

            matches = match_pair_roma(
                img_a,
                img_b,
                config,
                matcher=matcher,
                device=device,
            )

            # Apply mask if available (filter correspondences where ref pixel is outside mask)
            if masks is not None and canonical[0] in masks:
                matches = apply_mask_to_correspondences(matches, masks[canonical[0]])

            all_matches[canonical] = matches

    return all_matches
