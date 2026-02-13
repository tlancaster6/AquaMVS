"""Tests for depth extraction from cost volumes."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from aquamvs.dense import extract_depth, load_depth_map, save_depth_map


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Parametrize tests over CPU and CUDA devices."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return request.param


def test_wta_picks_correct_minimum(device):
    """Test that winner-take-all selects the depth index with minimum cost."""
    # Create a synthetic cost volume with known minimum locations
    H, W, D = 4, 4, 8
    cost_volume = torch.ones(H, W, D, device=device)

    # Set specific pixels to have minimum cost at known depth indices
    # Pixel (0, 0) has minimum at depth index 2
    cost_volume[0, 0, 2] = 0.1
    # Pixel (1, 1) has minimum at depth index 5
    cost_volume[1, 1, 5] = 0.15
    # Pixel (2, 2) has minimum at depth index 3
    cost_volume[2, 2, 3] = 0.05
    # Pixel (3, 3) has minimum at depth index 7 (boundary)
    cost_volume[3, 3, 7] = 0.2

    # Create depth hypotheses
    depths = torch.linspace(0.5, 1.5, D, device=device)

    # Extract depth
    depth_map, confidence = extract_depth(cost_volume, depths)

    # Verify dimensions
    assert depth_map.shape == (H, W)
    assert confidence.shape == (H, W)

    # Verify the depth values are close to the expected discrete depths
    # (allowing for sub-pixel refinement but they should be near)
    depth_step = depths[1] - depths[0]
    assert abs(depth_map[0, 0] - depths[2]) < depth_step
    assert abs(depth_map[1, 1] - depths[5]) < depth_step
    assert abs(depth_map[2, 2] - depths[3]) < depth_step

    # Pixel (3, 3) is at boundary (idx=7=D-1), should be NaN
    assert torch.isnan(depth_map[3, 3])
    assert confidence[3, 3] == 0.0


def test_subpixel_parabola_accuracy(device):
    """Test that sub-pixel parabola refinement accurately recovers sub-depth-step minima."""
    H, W, D = 4, 4, 16
    depths = torch.linspace(0.5, 1.5, D, device=device)
    depth_step = depths[1] - depths[0]

    # Create a cost volume with quadratic cost profiles centered at sub-pixel depths
    cost_volume = torch.zeros(H, W, D, device=device)

    # True depth at pixel (0, 0): between depths[5] and depths[6], offset = +0.3
    d_true_00 = depths[5] + 0.3 * depth_step
    # True depth at pixel (1, 1): between depths[8] and depths[9], offset = -0.2
    d_true_11 = depths[8] - 0.2 * depth_step
    # True depth at pixel (2, 2): exactly at depths[10], offset = 0.0
    d_true_22 = depths[10]

    # Construct quadratic cost profiles: cost = (d - d_true)^2
    for d_idx in range(D):
        cost_volume[0, 0, d_idx] = (depths[d_idx] - d_true_00) ** 2
        cost_volume[1, 1, d_idx] = (depths[d_idx] - d_true_11) ** 2
        cost_volume[2, 2, d_idx] = (depths[d_idx] - d_true_22) ** 2

    # Extract depth
    depth_map, confidence = extract_depth(cost_volume, depths)

    # Verify the refined depths match the true sub-pixel depths
    assert torch.isclose(depth_map[0, 0], d_true_00, atol=1e-4)
    assert torch.isclose(depth_map[1, 1], d_true_11, atol=1e-4)
    assert torch.isclose(depth_map[2, 2], d_true_22, atol=1e-4)

    # All should have high confidence (low cost)
    assert confidence[0, 0] > 0.5
    assert confidence[1, 1] > 0.5
    assert confidence[2, 2] > 0.5


def test_parabola_clamp(device):
    """Test that the parabola offset is clamped to [-0.5, 0.5]."""
    H, W, D = 2, 2, 8
    depths = torch.linspace(0.5, 1.5, D, device=device)
    depth_step = depths[1] - depths[0]

    # Create a skewed cost profile where the parabola formula would give |offset| > 0.5
    cost_volume = torch.ones(H, W, D, device=device)

    # Pixel (0, 0): minimum at index 3, but with highly asymmetric neighbors
    # Create a profile where c_minus >> c_center << c_plus to force large offset
    cost_volume[0, 0, 3] = 0.1  # center (minimum)
    cost_volume[0, 0, 2] = 0.15  # minus
    cost_volume[0, 0, 4] = 0.9  # plus (much higher)

    # Extract depth
    depth_map, confidence = extract_depth(cost_volume, depths)

    # The refined depth should be clamped within [depths[2], depths[4]]
    # Even though the parabola would extrapolate beyond
    assert depth_map[0, 0] >= depths[2] - 1e-6  # allow small numerical error
    assert depth_map[0, 0] <= depths[4] + 1e-6


def test_boundary_masking(device):
    """Test that pixels with minimum at boundary depths are marked as invalid."""
    H, W, D = 4, 4, 8
    depths = torch.linspace(0.5, 1.5, D, device=device)
    cost_volume = torch.ones(H, W, D, device=device)

    # Pixel (0, 0): minimum at d_idx = 0 (first depth)
    cost_volume[0, 0, 0] = 0.05
    # Pixel (1, 1): minimum at d_idx = D-1 (last depth)
    cost_volume[1, 1, D - 1] = 0.1
    # Pixel (2, 2): minimum at d_idx = 4 (interior, valid)
    cost_volume[2, 2, 4] = 0.08

    # Extract depth
    depth_map, confidence = extract_depth(cost_volume, depths)

    # Boundary pixels should have NaN depth and confidence = 0
    assert torch.isnan(depth_map[0, 0])
    assert confidence[0, 0] == 0.0
    assert torch.isnan(depth_map[1, 1])
    assert confidence[1, 1] == 0.0

    # Interior pixel should be valid
    assert not torch.isnan(depth_map[2, 2])
    assert confidence[2, 2] > 0.0


def test_confidence_range(device):
    """Test that confidence is always in [0, 1] for various cost profiles."""
    H, W, D = 8, 8, 16
    depths = torch.linspace(0.5, 1.5, D, device=device)

    # Test 1: Very low cost (near-perfect match) should produce high confidence
    cost_volume_low = torch.ones(H, W, D, device=device) * 0.5
    cost_volume_low[:, :, D // 2] = 0.01  # very low cost at middle depth
    depth_map_low, confidence_low = extract_depth(cost_volume_low, depths)

    # Test 2: High cost (poor match) should produce low confidence
    cost_volume_high = torch.ones(H, W, D, device=device) * 0.8
    cost_volume_high[:, :, D // 2] = 0.75  # slightly lower but still high cost
    depth_map_high, confidence_high = extract_depth(cost_volume_high, depths)

    # Test 3: Uniform cost (no distinctive minimum) should produce low confidence
    cost_volume_uniform = torch.ones(H, W, D, device=device) * 0.5
    # Add tiny variation to avoid exact ties
    cost_volume_uniform[:, :, D // 2] = 0.499
    depth_map_uniform, confidence_uniform = extract_depth(cost_volume_uniform, depths)

    # All confidence values should be in [0, 1]
    assert torch.all((confidence_low >= 0.0) & (confidence_low <= 1.0))
    assert torch.all((confidence_high >= 0.0) & (confidence_high <= 1.0))
    assert torch.all((confidence_uniform >= 0.0) & (confidence_uniform <= 1.0))

    # Low cost should give higher confidence than high cost (for non-boundary pixels)
    # Check interior pixels (avoid boundaries at idx=0 and idx=D-1)
    interior_mask = ~torch.isnan(depth_map_low)
    if interior_mask.any():
        assert torch.mean(confidence_low[interior_mask]) > torch.mean(
            confidence_high[interior_mask]
        )

    # Uniform cost (no distinctness) should give lower confidence
    interior_mask_uniform = ~torch.isnan(depth_map_uniform)
    if interior_mask_uniform.any():
        assert torch.mean(confidence_uniform[interior_mask_uniform]) < torch.mean(
            confidence_low[interior_mask]
        )


def test_confidence_zero_for_invalid(device):
    """Test that boundary pixels have confidence exactly 0."""
    H, W, D = 4, 4, 8
    depths = torch.linspace(0.5, 1.5, D, device=device)
    # Start with high costs everywhere
    cost_volume = torch.ones(H, W, D, device=device) * 0.9

    # Force some pixels to have minimum at boundaries
    cost_volume[0, :, 0] = 0.01  # first row, all at d_idx=0
    cost_volume[1, :, D - 1] = 0.01  # second row, all at d_idx=D-1

    # Extract depth
    depth_map, confidence = extract_depth(cost_volume, depths)

    # All boundary pixels should have confidence exactly 0
    assert torch.all(confidence[0, :] == 0.0)
    assert torch.all(confidence[1, :] == 0.0)

    # And NaN depth
    assert torch.all(torch.isnan(depth_map[0, :]))
    assert torch.all(torch.isnan(depth_map[1, :]))


def test_save_load_roundtrip(device):
    """Test that depth and confidence maps can be saved and loaded without loss."""
    H, W = 8, 8
    # Create synthetic depth and confidence maps with NaN values
    depth_map = torch.rand(H, W, device=device) * 2.0 + 0.5
    confidence = torch.rand(H, W, device=device)

    # Add some NaN values to depth
    depth_map[0, 0] = float("nan")
    depth_map[2, 3] = float("nan")
    depth_map[5, 7] = float("nan")

    # Corresponding confidence should be 0
    confidence[0, 0] = 0.0
    confidence[2, 3] = 0.0
    confidence[5, 7] = 0.0

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        save_depth_map(depth_map, confidence, tmp_path)

        # Load back
        loaded_depth, loaded_confidence = load_depth_map(tmp_path, device=device)

        # Verify shapes
        assert loaded_depth.shape == depth_map.shape
        assert loaded_confidence.shape == confidence.shape

        # Verify values (use numpy for NaN-aware comparison)
        np.testing.assert_array_equal(
            loaded_depth.cpu().numpy(), depth_map.cpu().numpy()
        )
        np.testing.assert_array_equal(
            loaded_confidence.cpu().numpy(), confidence.cpu().numpy()
        )

        # Verify device
        assert loaded_depth.device.type == device
        assert loaded_confidence.device.type == device

    finally:
        # Clean up
        tmp_path.unlink()


def test_save_load_path_types():
    """Test that save/load work with both str and Path objects."""
    H, W = 4, 4
    depth_map = torch.rand(H, W)
    confidence = torch.rand(H, W)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Test with string path
        save_depth_map(depth_map, confidence, str(tmp_path))
        loaded_depth_str, loaded_confidence_str = load_depth_map(str(tmp_path))
        assert loaded_depth_str.shape == (H, W)
        assert loaded_confidence_str.shape == (H, W)

        # Test with Path object
        save_depth_map(depth_map, confidence, tmp_path)
        loaded_depth_path, loaded_confidence_path = load_depth_map(tmp_path)
        assert loaded_depth_path.shape == (H, W)
        assert loaded_confidence_path.shape == (H, W)

    finally:
        tmp_path.unlink()


def test_extract_depth_output_shapes(device):
    """Test that extract_depth returns tensors with correct shapes."""
    # Various sizes
    test_cases = [
        (4, 4, 8),
        (16, 16, 32),
        (8, 12, 16),
    ]

    for H, W, D in test_cases:
        cost_volume = torch.rand(H, W, D, device=device)
        depths = torch.linspace(0.5, 1.5, D, device=device)

        depth_map, confidence = extract_depth(cost_volume, depths)

        assert depth_map.shape == (H, W), f"Expected {(H, W)}, got {depth_map.shape}"
        assert confidence.shape == (H, W), f"Expected {(H, W)}, got {confidence.shape}"
        assert depth_map.dtype == torch.float32
        assert confidence.dtype == torch.float32
        assert depth_map.device.type == device
        assert confidence.device.type == device


def test_extract_depth_with_all_boundary_minima(device):
    """Test edge case where all pixels have minimum at boundaries."""
    H, W, D = 4, 4, 8
    depths = torch.linspace(0.5, 1.5, D, device=device)
    cost_volume = torch.ones(H, W, D, device=device)

    # All pixels have minimum at first depth
    cost_volume[:, :, 0] = 0.01

    depth_map, confidence = extract_depth(cost_volume, depths)

    # All should be NaN depth and 0 confidence
    assert torch.all(torch.isnan(depth_map))
    assert torch.all(confidence == 0.0)


def test_extract_depth_numerical_stability(device):
    """Test that extract_depth handles numerical edge cases gracefully."""
    H, W, D = 4, 4, 8
    depths = torch.linspace(0.5, 1.5, D, device=device)

    # Test 1: Cost volume with very small values (near zero)
    cost_volume_small = torch.ones(H, W, D, device=device) * 1e-8
    cost_volume_small[:, :, 3] = 1e-9
    depth_map_small, confidence_small = extract_depth(cost_volume_small, depths)
    assert not torch.any(torch.isnan(confidence_small[2, 2]))  # interior pixel
    assert torch.all((confidence_small >= 0.0) & (confidence_small <= 1.0))

    # Test 2: Cost volume with identical costs (flat profile)
    cost_volume_flat = torch.ones(H, W, D, device=device) * 0.5
    depth_map_flat, confidence_flat = extract_depth(cost_volume_flat, depths)
    # Should not crash, confidence should be valid
    assert torch.all((confidence_flat >= 0.0) & (confidence_flat <= 1.0))

    # Test 3: Cost volume with very large values
    cost_volume_large = torch.ones(H, W, D, device=device) * 1e6
    cost_volume_large[:, :, 4] = 1e5
    depth_map_large, confidence_large = extract_depth(cost_volume_large, depths)
    assert torch.all((confidence_large >= 0.0) & (confidence_large <= 1.0))
