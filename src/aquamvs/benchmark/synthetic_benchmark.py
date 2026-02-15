"""Synthetic benchmark script for reconstruction accuracy comparison.

Benchmarks pipeline accuracy on synthetic scenes without requiring real video
files or calibration. Exercises the accuracy metrics infrastructure with
simulated reconstruction quality profiles.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from .metrics import compute_accuracy_metrics, compute_plane_fit_metrics
from .synthetic import create_flat_plane_scene, create_undulating_scene

logger = logging.getLogger(__name__)


def simulate_reconstruction(
    mesh,
    num_points: int,
    noise_sigma_m: float,
    dropout_rate: float,
    seed: int = 42,
) -> np.ndarray:
    """Simulate a reconstruction by sampling mesh with noise and dropout.

    Args:
        mesh: Ground truth Open3D mesh.
        num_points: Number of points to sample from mesh.
        noise_sigma_m: Gaussian noise standard deviation (meters).
        dropout_rate: Fraction of points to drop (0-1).
        seed: Random seed for reproducibility.

    Returns:
        N × 3 array of noisy sampled points.
    """
    np.random.seed(seed)

    # Sample points from mesh surface
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    points = np.asarray(pcd.points)

    # Add Gaussian noise
    noise = np.random.normal(0, noise_sigma_m, points.shape)
    noisy_points = points + noise

    # Apply dropout
    keep_mask = np.random.rand(len(noisy_points)) > dropout_rate
    final_points = noisy_points[keep_mask]

    logger.info(
        "Simulated reconstruction: %d points (%.1f%% kept after dropout)",
        len(final_points),
        100.0 * len(final_points) / num_points,
    )

    return final_points


def run_benchmark(output_path: Path) -> dict:
    """Run synthetic benchmark comparing simulated reconstruction configs.

    Args:
        output_path: Path to save JSON results.

    Returns:
        Dictionary with benchmark results.
    """
    logger.info("Starting synthetic benchmark")

    # Scene parameters (reference geometry)
    depth_z = 1.178  # water_z + 200mm
    bounds = (-0.25, 0.25, -0.15, 0.15)  # 50cm x 30cm patch
    resolution = 0.005  # 5mm mesh resolution

    # Create synthetic scenes
    logger.info("Generating flat plane scene")
    flat_mesh, flat_analytic = create_flat_plane_scene(depth_z, bounds, resolution)

    logger.info("Generating undulating scene")
    undulating_mesh, _ = create_undulating_scene(
        base_depth_z=depth_z,
        amplitude=0.005,  # 5mm amplitude
        wavelength=0.05,  # 5cm wavelength
        bounds=bounds,
        resolution=resolution,
    )

    # Simulated reconstruction configs
    configs = {
        "LightGlue-like": {
            "num_points": 1000,
            "noise_sigma_m": 0.002,  # 2mm noise
            "dropout_rate": 0.10,  # 10% dropout
        },
        "RoMa-like": {
            "num_points": 3000,
            "noise_sigma_m": 0.001,  # 1mm noise
            "dropout_rate": 0.05,  # 5% dropout
        },
    }

    results = {}

    # Benchmark each config on both scenes
    for config_name, config_params in configs.items():
        logger.info("Benchmarking %s configuration", config_name)

        # Flat plane scene
        flat_points = simulate_reconstruction(flat_mesh, **config_params, seed=42)
        flat_metrics = compute_accuracy_metrics(
            flat_points, flat_mesh, tolerance_mm=5.0
        )
        flat_plane_metrics = compute_plane_fit_metrics(flat_points)

        # Undulating scene
        undulating_points = simulate_reconstruction(
            undulating_mesh, **config_params, seed=43
        )
        undulating_metrics = compute_accuracy_metrics(
            undulating_points, undulating_mesh, tolerance_mm=5.0
        )

        results[config_name] = {
            "flat_plane": {
                "mean_error_mm": flat_metrics["mean_error_mm"],
                "median_error_mm": flat_metrics["median_error_mm"],
                "completeness_pct": flat_metrics["raw_completeness_pct"],
                "plane_fit_rmse_mm": flat_plane_metrics["plane_fit_rmse_mm"],
            },
            "undulating": {
                "mean_error_mm": undulating_metrics["mean_error_mm"],
                "median_error_mm": undulating_metrics["median_error_mm"],
                "completeness_pct": undulating_metrics["raw_completeness_pct"],
            },
        }

    # Print comparison table
    print("\n" + "=" * 70)
    print("Synthetic Benchmark: Reconstruction Quality Comparison")
    print("=" * 70)
    print("\nFlat Plane Scene:")
    print(
        f"{'Config':<20s} {'Mean Error (mm)':<18s} {'Median Error (mm)':<20s} "
        f"{'Completeness (%)':<18s} {'Plane RMSE (mm)':<15s}"
    )
    print("-" * 95)
    for config_name, config_results in results.items():
        flat = config_results["flat_plane"]
        print(
            f"{config_name:<20s} {flat['mean_error_mm']:<18.2f} "
            f"{flat['median_error_mm']:<20.2f} {flat['completeness_pct']:<18.2f} "
            f"{flat['plane_fit_rmse_mm']:<15.2f}"
        )

    print("\nUndulating Scene:")
    print(
        f"{'Config':<20s} {'Mean Error (mm)':<18s} {'Median Error (mm)':<20s} "
        f"{'Completeness (%)':<18s}"
    )
    print("-" * 80)
    for config_name, config_results in results.items():
        und = config_results["undulating"]
        print(
            f"{config_name:<20s} {und['mean_error_mm']:<18.2f} "
            f"{und['median_error_mm']:<20.2f} {und['completeness_pct']:<18.2f}"
        )

    print("\nNote: These are SIMULATED reconstructions with synthetic noise profiles.")
    print(
        "Purpose: Validate that benchmark metrics can discriminate quality differences."
    )
    print("Real pipeline comparison requires full dataset and calibration.\n")

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Benchmark results saved to: %s", output_path)

    return results


def main():
    """CLI entry point for synthetic benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark reconstruction accuracy on synthetic data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".benchmarks/benchmark_results.json"),
        help="Output path for JSON results",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        run_benchmark(args.output)
        return 0

    except Exception as e:
        logger.exception("Benchmark failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
