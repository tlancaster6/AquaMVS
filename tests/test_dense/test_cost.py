"""Tests for photometric cost functions."""

import pytest
import torch

from aquamvs.dense.cost import (
    aggregate_costs,
    compute_cost,
    compute_ncc,
    compute_ssim,
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Parametrize tests over CPU and CUDA devices."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


class TestComputeNCC:
    """Tests for compute_ncc function."""

    def test_identical_images_zero_cost(self, device):
        """Identical images should produce near-zero cost."""
        torch.manual_seed(42)
        H, W = 64, 48
        window_size = 11
        ref = torch.rand(H, W, device=device, dtype=torch.float32)
        src = ref.clone()

        cost = compute_ncc(ref, src, window_size=window_size)

        assert cost.shape == (H, W)
        # Check interior only (borders may have cost=1.0 due to insufficient coverage)
        border = window_size // 2 + 1
        interior = cost[border:-border, border:-border]
        assert torch.allclose(interior, torch.zeros_like(interior), atol=1e-5)

    def test_uncorrelated_images_high_cost(self, device):
        """Uncorrelated images should produce cost around 1.0."""
        torch.manual_seed(42)
        H, W = 64, 48
        ref = torch.rand(H, W, device=device, dtype=torch.float32)
        src = torch.rand(H, W, device=device, dtype=torch.float32)

        cost = compute_ncc(ref, src, window_size=11)

        assert cost.shape == (H, W)
        # Cost should be around 1.0 for uncorrelated noise (NCC ~ 0)
        # Not exact due to local window statistics, but should be high
        assert cost.mean() > 0.8
        assert cost.mean() < 1.2

    def test_linear_shift_robustness(self, device):
        """NCC should be robust to linear intensity changes."""
        torch.manual_seed(42)
        H, W = 64, 48
        window_size = 11
        ref = torch.rand(H, W, device=device, dtype=torch.float32)
        # Apply linear transformation: brightness + contrast
        src = ref * 0.5 + 0.2

        cost = compute_ncc(ref, src, window_size=window_size)

        assert cost.shape == (H, W)
        # NCC normalizes by local mean and std, so should be robust
        # Check interior only (borders may have cost=1.0 due to insufficient coverage)
        border = window_size // 2 + 1
        interior = cost[border:-border, border:-border]
        assert torch.allclose(interior, torch.zeros_like(interior), atol=0.05)

    def test_nan_handling(self, device):
        """NaN pixels in source should produce cost = 1.0."""
        torch.manual_seed(42)
        H, W = 64, 48
        ref = torch.rand(H, W, device=device, dtype=torch.float32)
        src = torch.rand(H, W, device=device, dtype=torch.float32)

        # Set a region to NaN
        src[10:20, 15:25] = float("nan")

        cost = compute_ncc(ref, src, window_size=11)

        assert cost.shape == (H, W)
        # Center of NaN region should have cost = 1.0 (insufficient valid pixels)
        center_cost = cost[15, 20]
        assert torch.isclose(center_cost, torch.tensor(1.0, device=device), atol=1e-5)

        # Far from NaN region should still work
        # (more than window_size // 2 away)
        valid_region_cost = cost[40, 40]
        assert not torch.isnan(valid_region_cost)

    def test_output_shape_various_sizes(self, device):
        """Output shape should match input for various sizes."""
        for H, W in [(32, 32), (64, 48), (100, 80)]:
            ref = torch.rand(H, W, device=device, dtype=torch.float32)
            src = torch.rand(H, W, device=device, dtype=torch.float32)

            for window_size in [5, 11, 21]:
                cost = compute_ncc(ref, src, window_size=window_size)
                assert cost.shape == (H, W)

    def test_differentiability(self, device):
        """NCC should support autograd."""
        torch.manual_seed(42)
        H, W = 32, 32
        ref = torch.rand(H, W, device=device, dtype=torch.float32, requires_grad=True)
        src = torch.rand(H, W, device=device, dtype=torch.float32)

        cost = compute_ncc(ref, src, window_size=11)
        loss = cost.mean()
        loss.backward()

        assert ref.grad is not None
        assert ref.grad.shape == ref.shape
        assert not torch.isnan(ref.grad).any()


class TestComputeSSIM:
    """Tests for compute_ssim function."""

    def test_identical_images_zero_cost(self, device):
        """Identical images should produce near-zero cost."""
        torch.manual_seed(42)
        H, W = 64, 48
        window_size = 11
        ref = torch.rand(H, W, device=device, dtype=torch.float32)
        src = ref.clone()

        cost = compute_ssim(ref, src, window_size=window_size)

        assert cost.shape == (H, W)
        # Check interior only (borders may have cost=1.0 due to insufficient coverage)
        border = window_size // 2 + 1
        interior = cost[border:-border, border:-border]
        assert torch.allclose(interior, torch.zeros_like(interior), atol=1e-5)

    def test_uncorrelated_images_high_cost(self, device):
        """Uncorrelated images should produce high cost."""
        torch.manual_seed(42)
        H, W = 64, 48
        ref = torch.rand(H, W, device=device, dtype=torch.float32)
        src = torch.rand(H, W, device=device, dtype=torch.float32)

        cost = compute_ssim(ref, src, window_size=11)

        assert cost.shape == (H, W)
        # SSIM cost should be high (> 0.5) for random images
        assert cost.mean() > 0.5

    def test_linear_shift_sensitivity(self, device):
        """SSIM is sensitive to luminance shifts (unlike NCC)."""
        torch.manual_seed(42)
        H, W = 64, 48
        ref = torch.rand(H, W, device=device, dtype=torch.float32) * 0.5 + 0.25
        # Apply brightness shift
        src = ref + 0.2

        cost = compute_ssim(ref, src, window_size=11)

        assert cost.shape == (H, W)
        # SSIM should detect the brightness shift (cost > 0)
        # but not as bad as random (cost < 0.5)
        assert cost.mean() > 0.05
        assert cost.mean() < 0.5

    def test_nan_handling(self, device):
        """NaN pixels in source should produce cost = 1.0."""
        torch.manual_seed(42)
        H, W = 64, 48
        ref = torch.rand(H, W, device=device, dtype=torch.float32)
        src = torch.rand(H, W, device=device, dtype=torch.float32)

        # Set a region to NaN
        src[10:20, 15:25] = float("nan")

        cost = compute_ssim(ref, src, window_size=11)

        assert cost.shape == (H, W)
        # Center of NaN region should have cost = 1.0
        center_cost = cost[15, 20]
        assert torch.isclose(center_cost, torch.tensor(1.0, device=device), atol=1e-5)

        # Far from NaN region should still work
        valid_region_cost = cost[40, 40]
        assert not torch.isnan(valid_region_cost)

    def test_output_shape_various_sizes(self, device):
        """Output shape should match input for various sizes."""
        for H, W in [(32, 32), (64, 48), (100, 80)]:
            ref = torch.rand(H, W, device=device, dtype=torch.float32)
            src = torch.rand(H, W, device=device, dtype=torch.float32)

            for window_size in [5, 11, 21]:
                cost = compute_ssim(ref, src, window_size=window_size)
                assert cost.shape == (H, W)

    def test_differentiability(self, device):
        """SSIM should support autograd."""
        torch.manual_seed(42)
        H, W = 32, 32
        ref = torch.rand(H, W, device=device, dtype=torch.float32, requires_grad=True)
        src = torch.rand(H, W, device=device, dtype=torch.float32)

        cost = compute_ssim(ref, src, window_size=11)
        loss = cost.mean()
        loss.backward()

        assert ref.grad is not None
        assert ref.grad.shape == ref.shape
        assert not torch.isnan(ref.grad).any()


class TestComputeCost:
    """Tests for compute_cost dispatch function."""

    def test_dispatch_ncc(self, device):
        """Dispatch to NCC should work."""
        torch.manual_seed(42)
        H, W = 32, 32
        window_size = 11
        ref = torch.rand(H, W, device=device, dtype=torch.float32)
        src = ref.clone()

        cost = compute_cost(ref, src, cost_function="ncc", window_size=window_size)

        assert cost.shape == (H, W)
        # Check interior only
        border = window_size // 2 + 1
        interior = cost[border:-border, border:-border]
        assert torch.allclose(interior, torch.zeros_like(interior), atol=1e-5)

    def test_dispatch_ssim(self, device):
        """Dispatch to SSIM should work."""
        torch.manual_seed(42)
        H, W = 32, 32
        window_size = 11
        ref = torch.rand(H, W, device=device, dtype=torch.float32)
        src = ref.clone()

        cost = compute_cost(ref, src, cost_function="ssim", window_size=window_size)

        assert cost.shape == (H, W)
        # Check interior only
        border = window_size // 2 + 1
        interior = cost[border:-border, border:-border]
        assert torch.allclose(interior, torch.zeros_like(interior), atol=1e-5)

    def test_unknown_cost_function_raises(self, device):
        """Unknown cost function should raise ValueError."""
        torch.manual_seed(42)
        H, W = 32, 32
        ref = torch.rand(H, W, device=device, dtype=torch.float32)
        src = torch.rand(H, W, device=device, dtype=torch.float32)

        with pytest.raises(ValueError, match="Unknown cost function"):
            compute_cost(ref, src, cost_function="unknown", window_size=11)


class TestAggregateCosts:
    """Tests for aggregate_costs function."""

    def test_mean_aggregation(self, device):
        """Should compute mean across source views."""
        H, W = 32, 32
        # Create 3 cost maps with known values
        cost1 = torch.ones(H, W, device=device, dtype=torch.float32) * 0.2
        cost2 = torch.ones(H, W, device=device, dtype=torch.float32) * 0.4
        cost3 = torch.ones(H, W, device=device, dtype=torch.float32) * 0.6

        aggregated = aggregate_costs([cost1, cost2, cost3])

        expected = torch.ones(H, W, device=device, dtype=torch.float32) * 0.4
        assert torch.allclose(aggregated, expected, atol=1e-6)

    def test_single_cost_map(self, device):
        """Should work with a single cost map."""
        H, W = 32, 32
        cost = torch.rand(H, W, device=device, dtype=torch.float32)

        aggregated = aggregate_costs([cost])

        assert torch.allclose(aggregated, cost, atol=1e-6)


class TestWindowSizeEffect:
    """Test that window size affects smoothness."""

    def test_larger_window_smoother(self):
        """Larger window should produce smoother cost maps."""
        torch.manual_seed(42)
        H, W = 64, 64
        device = torch.device("cpu")

        # Create slightly noisy but similar images
        ref = torch.rand(H, W, device=device, dtype=torch.float32)
        # Add small amount of noise to keep images well-correlated
        src = ref + torch.randn(H, W, device=device, dtype=torch.float32) * 0.02

        # Compute with small and large windows
        cost_small = compute_ncc(ref, src, window_size=5)
        cost_large = compute_ncc(ref, src, window_size=21)

        # Larger window should have lower variance (smoother)
        # We exclude borders where insufficient valid pixels cause artifacts
        inner_slice = slice(11, -11)
        var_small = cost_small[inner_slice, inner_slice].var()
        var_large = cost_large[inner_slice, inner_slice].var()

        assert var_large < var_small


class TestCostRange:
    """Test that cost values stay in expected ranges."""

    def test_ncc_range(self):
        """NCC cost should be in [0, 2]."""
        torch.manual_seed(42)
        H, W = 64, 48
        device = torch.device("cpu")

        # Test with various image pairs
        for _ in range(5):
            ref = torch.rand(H, W, device=device, dtype=torch.float32)
            src = torch.rand(H, W, device=device, dtype=torch.float32)

            cost = compute_ncc(ref, src, window_size=11)

            assert cost.min() >= 0.0
            assert cost.max() <= 2.0

    def test_ssim_range(self):
        """SSIM cost should be in [0, 1]."""
        torch.manual_seed(42)
        H, W = 64, 48
        device = torch.device("cpu")

        # Test with various image pairs
        for _ in range(5):
            ref = torch.rand(H, W, device=device, dtype=torch.float32)
            src = torch.rand(H, W, device=device, dtype=torch.float32)

            cost = compute_ssim(ref, src, window_size=11)

            assert cost.min() >= 0.0
            assert cost.max() <= 1.0
