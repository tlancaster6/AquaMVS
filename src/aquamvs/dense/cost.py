"""Photometric cost functions for dense matching."""

import torch
import torch.nn.functional as F


def compute_ncc(
    ref: torch.Tensor,
    src: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """Compute normalized cross-correlation cost between two images.

    NCC measures similarity in local windows, normalized by local mean and
    standard deviation. This makes it robust to linear intensity changes
    between views (e.g., vignetting, exposure differences).

    Cost = 1 - NCC, so 0 = perfect match, 1 = uncorrelated, 2 = anti-correlated.

    Args:
        ref: Reference image, shape (H, W), float32 in [0, 1].
        src: Warped source image, shape (H, W), float32 in [0, 1].
            May contain NaN for invalid (out-of-bounds) pixels.
        window_size: Local window size (must be odd).

    Returns:
        Cost map, shape (H, W), float32 in [0, 2].
        Invalid regions (where src is NaN or window has insufficient
        valid pixels) are set to 1.0.
    """
    # Replace NaN with 0 and build a validity mask
    valid = ~torch.isnan(src)
    src_clean = torch.where(valid, src, torch.zeros_like(src))
    mask = valid.float()  # 1 where valid, 0 where NaN

    # Add batch+channel dims for conv2d: (1, 1, H, W)
    ref_4d = ref.unsqueeze(0).unsqueeze(0)
    src_4d = src_clean.unsqueeze(0).unsqueeze(0)
    mask_4d = mask.unsqueeze(0).unsqueeze(0)

    # Uniform box filter kernel
    pad = window_size // 2
    kernel = torch.ones(
        1, 1, window_size, window_size, device=ref.device, dtype=ref.dtype
    )

    # Count valid pixels per window
    count = F.conv2d(mask_4d, kernel, padding=pad)  # (1, 1, H, W)
    count = count.clamp(min=1.0)  # avoid division by zero

    # Local means (using only valid pixels)
    # We need to mask both ref and src to only use pixels where src is valid
    mean_ref = F.conv2d(ref_4d * mask_4d, kernel, padding=pad) / count
    mean_src = F.conv2d(src_4d * mask_4d, kernel, padding=pad) / count

    # Local sums for variance and covariance
    # Center and mask both images
    ref_centered = (ref_4d - mean_ref) * mask_4d
    src_centered = (src_4d - mean_src) * mask_4d

    var_ref = F.conv2d(ref_centered**2, kernel, padding=pad) / count
    var_src = F.conv2d(src_centered**2, kernel, padding=pad) / count
    covar = F.conv2d(ref_centered * src_centered, kernel, padding=pad) / count

    # NCC = covariance / (std_ref * std_src + epsilon)
    eps = 1e-8
    ncc = covar / (torch.sqrt(var_ref * var_src) + eps)

    # Cost = 1 - NCC
    cost = 1.0 - ncc

    # Set invalid regions to 1.0 (where too few valid pixels in window)
    min_valid = (window_size * window_size) // 2  # require at least half the window
    insufficient = F.conv2d(mask_4d, kernel, padding=pad) < min_valid
    cost = torch.where(insufficient, torch.ones_like(cost), cost)

    # Clamp to valid range [0, 2] to handle numerical edge cases
    cost = cost.clamp(0.0, 2.0)

    return cost.squeeze(0).squeeze(0)  # (H, W)


def compute_ssim(
    ref: torch.Tensor,
    src: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """Compute structural similarity cost between two images.

    SSIM compares luminance, contrast, and structure in local windows.
    More perceptually motivated than NCC.

    Cost = 1 - SSIM, so 0 = perfect match, 1 = maximally dissimilar.

    Args:
        ref: Reference image, shape (H, W), float32 in [0, 1].
        src: Warped source image, shape (H, W), float32 in [0, 1].
            May contain NaN for invalid (out-of-bounds) pixels.
        window_size: Local window size (must be odd).

    Returns:
        Cost map, shape (H, W), float32 in [0, 1].
        Invalid regions are set to 1.0.
    """
    # Replace NaN with 0 and build a validity mask
    valid = ~torch.isnan(src)
    src_clean = torch.where(valid, src, torch.zeros_like(src))
    mask = valid.float()  # 1 where valid, 0 where NaN

    # Add batch+channel dims for conv2d: (1, 1, H, W)
    ref_4d = ref.unsqueeze(0).unsqueeze(0)
    src_4d = src_clean.unsqueeze(0).unsqueeze(0)
    mask_4d = mask.unsqueeze(0).unsqueeze(0)

    # Create Gaussian kernel
    sigma = 1.5  # standard for SSIM
    pad = window_size // 2
    coords = (
        torch.arange(window_size, device=ref.device, dtype=ref.dtype) - window_size // 2
    )
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel_4d = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)

    # Count valid pixels per window (with Gaussian weighting)
    count = F.conv2d(mask_4d, kernel_4d, padding=pad)
    count = count.clamp(min=1e-8)  # avoid division by zero

    # Local means (Gaussian-weighted, normalized by valid pixel weights)
    mean_ref = F.conv2d(ref_4d * mask_4d, kernel_4d, padding=pad) / count
    mean_src = F.conv2d(src_4d * mask_4d, kernel_4d, padding=pad) / count

    # Local variances and covariance
    ref_centered = (ref_4d - mean_ref) * mask_4d
    src_centered = (src_4d - mean_src) * mask_4d

    var_ref = F.conv2d(ref_centered**2, kernel_4d, padding=pad) / count
    var_src = F.conv2d(src_centered**2, kernel_4d, padding=pad) / count
    covar = F.conv2d(ref_centered * src_centered, kernel_4d, padding=pad) / count

    # SSIM constants (Wang et al. 2004)
    L = 1.0  # dynamic range
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    # SSIM formula
    numerator = (2 * mean_ref * mean_src + C1) * (2 * covar + C2)
    denominator = (mean_ref**2 + mean_src**2 + C1) * (var_ref + var_src + C2)
    ssim = numerator / denominator

    # Cost = 1 - SSIM
    cost = 1.0 - ssim

    # Set invalid regions to 1.0 (where too few valid pixels in window)
    # For SSIM with Gaussian weighting, require at least 50% of the total Gaussian weight
    min_valid_weight = 0.5
    insufficient = count < (min_valid_weight * kernel_2d.sum())
    cost = torch.where(insufficient, torch.ones_like(cost), cost)

    # Clamp to valid range [0, 1] to handle numerical edge cases
    cost = cost.clamp(0.0, 1.0)

    return cost.squeeze(0).squeeze(0)  # (H, W)


def compute_cost(
    ref: torch.Tensor,
    src: torch.Tensor,
    cost_function: str = "ncc",
    window_size: int = 11,
) -> torch.Tensor:
    """Compute photometric cost between reference and warped source images.

    Dispatches to the appropriate cost function based on the string name.

    Args:
        ref: Reference image, shape (H, W), float32 in [0, 1].
        src: Warped source image, shape (H, W), float32 in [0, 1].
        cost_function: "ncc" or "ssim".
        window_size: Local window size (must be odd).

    Returns:
        Cost map, shape (H, W), float32. Lower = better match.

    Raises:
        ValueError: If cost_function is not "ncc" or "ssim".
    """
    match cost_function:
        case "ncc":
            return compute_ncc(ref, src, window_size)
        case "ssim":
            return compute_ssim(ref, src, window_size)
        case _:
            raise ValueError(
                f"Unknown cost function: {cost_function!r}. Expected 'ncc' or 'ssim'."
            )


def aggregate_costs(
    costs: list[torch.Tensor],
) -> torch.Tensor:
    """Aggregate cost maps from multiple source views.

    Takes the mean cost across all source views at each pixel.
    This is the simplest aggregation strategy and works well when
    source views have similar quality.

    Args:
        costs: List of cost maps, each shape (H, W), float32.
            Must have at least one element.

    Returns:
        Aggregated cost map, shape (H, W), float32.
    """
    return torch.stack(costs, dim=0).mean(dim=0)