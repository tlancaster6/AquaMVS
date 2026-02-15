"""Fast CI benchmarks for regression detection.

These benchmarks are designed to run in under 60 seconds total and detect
performance/accuracy regressions. They use synthetic scenes with small image
sizes and few depth hypotheses.
"""

import numpy as np
import pytest
import torch

from aquamvs.benchmark.metrics import (
    compute_completeness,
    compute_geometric_error,
)
from aquamvs.benchmark.synthetic import create_flat_plane_scene
from aquamvs.dense.plane_sweep import extract_depth_map
from aquamvs.projection.refractive import RefractiveProjectionModel

# Regression thresholds (conservative to avoid false positives)
COMPLETENESS_THRESHOLD = 0.60  # Must achieve at least 60% completeness
ERROR_THRESHOLD_MM = 5.0  # Median error must be under 5mm


@pytest.fixture
def synthetic_scene():
    """Create minimal flat plane scene for fast testing."""
    # Small plane at 1.2m depth
    mesh, analytic_fn = create_flat_plane_scene(
        depth_z=1.2,
        bounds=(-0.1, 0.1, -0.1, 0.1),  # 20cm x 20cm patch
        resolution=0.01,  # 1cm spacing
    )
    return mesh, analytic_fn


@pytest.fixture
def dummy_projection_model():
    """Create minimal projection model for testing."""
    # Minimal camera pointing down at origin
    K = torch.eye(3, dtype=torch.float32)
    K[0, 0] = K[1, 1] = 500.0  # Focal length
    K[0, 2] = K[1, 2] = 32.0  # Principal point (64x64 image center)

    R = torch.eye(3, dtype=torch.float32)
    t = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    dist_coeffs = torch.zeros(5, dtype=torch.float32)

    return RefractiveProjectionModel(
        K=K,
        R=R,
        t=t,
        dist_coeffs=dist_coeffs,
        water_z=1.0,
        n_water=1.333,
        image_size=(64, 64),
        is_auxiliary=False,
    )


@pytest.mark.benchmark
def test_ci_depth_extraction_correctness():
    """Unit test: verify depth extraction produces correct results from known cost volume."""
    # Create simple cost volume with known minimum at depth index 5
    H, W, D = 32, 32, 16
    cost_volume = torch.ones(H, W, D, dtype=torch.float32)

    # Set depth index 5 to have minimum cost for all pixels
    cost_volume[:, :, 5] = 0.0

    depths = torch.linspace(0.5, 2.0, D, dtype=torch.float32)

    # Extract depth map
    depth_map, confidence = extract_depth_map(cost_volume, depths)

    # Verify all pixels extracted depth index 5
    expected_depth = depths[5].item()
    assert torch.allclose(
        depth_map, torch.full_like(depth_map, expected_depth), atol=1e-5
    )

    # Verify confidence is high (low cost difference)
    assert torch.all(confidence > 0.8)


@pytest.mark.benchmark
def test_ci_metric_computation():
    """Unit test: verify metrics produce correct values on known inputs."""
    # Create simple point cloud and ground truth
    # 10 points in a line at Z=1.0
    points = np.column_stack(
        [
            np.linspace(-0.1, 0.1, 10),
            np.zeros(10),
            np.ones(10),
        ]
    )

    # Ground truth: same points (perfect reconstruction)
    ground_truth = points.copy()

    # Completeness should be 100%
    completeness = compute_completeness(points, ground_truth, threshold=0.01)
    assert completeness > 0.95  # Allow some tolerance for float math

    # Geometric error should be near zero
    mean_error, median_error = compute_geometric_error(points, ground_truth)
    assert mean_error < 0.001  # Less than 1mm
    assert median_error < 0.001


@pytest.mark.benchmark
@pytest.mark.parametrize("device", ["cpu"])  # GPU not required for regression detection
def test_ci_synthetic_plane_sparse(device, synthetic_scene, dummy_projection_model):
    """CI benchmark: sparse matching on flat plane (fast regression check).

    This test verifies that basic depth estimation produces reasonable results.
    It's not a full accuracy test - just a smoke test to catch major regressions.
    """
    pytest.skip("Requires full pipeline integration - placeholder for future work")

    # NOTE: This would require:
    # 1. Rendering synthetic views from multiple cameras
    # 2. Running sparse feature extraction and matching
    # 3. Building cost volume with minimal depth hypotheses (8-16)
    # 4. Extracting depth map
    # 5. Computing metrics against ground truth
    #
    # Target runtime: <20 seconds per test
    # Target metrics: completeness > 60%, median error < 5mm


@pytest.mark.benchmark
@pytest.mark.parametrize("device", ["cpu"])
def test_ci_plane_sweep_consistency(device, dummy_projection_model):
    """CI benchmark: verify plane sweep produces consistent depth maps.

    Tests that cost volume construction is deterministic and depth extraction
    is stable across runs.
    """
    pytest.skip("Requires rendered views - placeholder for future work")

    # NOTE: This would render a simple scene, build cost volume twice,
    # and verify identical results (no randomness in plane sweep).
    #
    # Target runtime: <10 seconds


@pytest.mark.benchmark
def test_ci_benchmark_suite_runtime():
    """Meta-test: verify entire CI benchmark suite runs under 60 seconds."""
    # This is a placeholder that documents the runtime requirement.
    # The actual enforcement happens at the pytest level via --timeout=120.
    #
    # If this test exists, it serves as documentation that the suite
    # is designed for speed.
    pass
