"""Shared pytest fixtures for AquaMVS tests."""

import pytest
import torch


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Parametrized device fixture for CPU and CUDA testing.

    Args:
        request: pytest fixture request object.

    Returns:
        torch.device: Device to use for testing.

    Raises:
        pytest.skip: If CUDA is requested but not available.
    """
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)
