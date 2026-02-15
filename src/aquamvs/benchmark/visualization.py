"""Benchmark visualization: error heatmaps, bar charts, and depth comparisons."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .runner import BenchmarkRunResult

logger = logging.getLogger(__name__)


def generate_visualizations(run_dir: Path, results: BenchmarkRunResult) -> list[Path]:
    """Generate all visualization plots for a benchmark run.

    Creates:
        - Error heatmaps (per config with spatial error data)
        - Grouped bar charts (accuracy metrics, timing metrics)
        - Depth map side-by-side comparisons (per test with depth maps)

    All plots saved to {run_dir}/plots/

    Args:
        run_dir: Benchmark run directory (timestamped).
        results: Benchmark run results with per-test metrics.

    Returns:
        List of generated plot file paths.
    """
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    generated_plots = []

    # Generate error heatmaps (if spatial error data available)
    logger.info("Generating error heatmaps...")
    heatmap_plots = _plot_error_heatmaps(plots_dir, results)
    generated_plots.extend(heatmap_plots)

    # Generate grouped bar charts for accuracy metrics
    logger.info("Generating accuracy bar charts...")
    accuracy_plot = _plot_accuracy_bars(plots_dir, results)
    if accuracy_plot:
        generated_plots.append(accuracy_plot)

    # Generate grouped bar charts for timing metrics
    logger.info("Generating timing bar charts...")
    timing_plot = _plot_timing_bars(plots_dir, results)
    if timing_plot:
        generated_plots.append(timing_plot)

    # Generate depth map comparisons (if depth maps available)
    logger.info("Generating depth map comparisons...")
    depth_plots = _plot_depth_comparisons(plots_dir, results)
    generated_plots.extend(depth_plots)

    logger.info(f"Generated {len(generated_plots)} plots in {plots_dir}")
    return generated_plots


def _plot_error_heatmaps(plots_dir: Path, results: BenchmarkRunResult) -> list[Path]:
    """Generate error heatmaps for each config with spatial error data.

    Args:
        plots_dir: Output directory for plots.
        results: Benchmark run results.

    Returns:
        List of generated heatmap file paths.
    """
    generated = []

    # For each test, check if spatial error data exists
    for test_name, test_result in results.test_results.items():
        test_plots_dir = plots_dir / test_name
        test_plots_dir.mkdir(parents=True, exist_ok=True)

        for _config_name, _metrics in test_result.configs.items():
            # Check if spatial error data available
            # (Future: load from config output directory)
            # For now, skip heatmaps (no spatial data in current metrics)
            pass

    return generated


def _plot_accuracy_bars(plots_dir: Path, results: BenchmarkRunResult) -> Path | None:
    """Generate grouped bar chart for accuracy metrics across configs.

    Plots mean_error_mm, median_error_mm, completeness_pct for all configs.

    Args:
        plots_dir: Output directory for plots.
        results: Benchmark run results.

    Returns:
        Path to generated plot, or None if no accuracy data.
    """
    # Collect all configs and their accuracy metrics
    config_names = []
    mean_errors = []
    median_errors = []
    completeness = []

    for test_result in results.test_results.values():
        for config_name, metrics in test_result.configs.items():
            if "mean_error_mm" in metrics:
                config_names.append(config_name)
                mean_errors.append(metrics.get("mean_error_mm", 0.0))
                median_errors.append(metrics.get("median_error_mm", 0.0))
                completeness.append(metrics.get("raw_completeness_pct", 0.0))

    if not config_names:
        logger.warning("No accuracy metrics found - skipping accuracy bar chart")
        return None

    # Create grouped bar chart
    x = np.arange(len(config_names))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(max(10, len(config_names) * 0.8), 6))

    # Error bars (left y-axis)
    ax1.bar(x - width, mean_errors, width, label="Mean Error (mm)", color="coral")
    ax1.bar(x, median_errors, width, label="Median Error (mm)", color="skyblue")
    ax1.set_xlabel("Configuration", fontsize=12)
    ax1.set_ylabel("Error (mm)", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_names, rotation=45, ha="right")
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", alpha=0.3)

    # Completeness bars (right y-axis)
    ax2 = ax1.twinx()
    ax2.bar(
        x + width,
        completeness,
        width,
        label="Completeness (%)",
        color="lightgreen",
    )
    ax2.set_ylabel("Completeness (%)", fontsize=12)
    ax2.legend(loc="upper right")

    plt.title("Accuracy Metrics Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = plots_dir / "accuracy_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved accuracy bar chart: {output_path}")
    return output_path


def _plot_timing_bars(plots_dir: Path, results: BenchmarkRunResult) -> Path | None:
    """Generate grouped bar chart for timing metrics across configs.

    Plots timing_seconds for all configs.

    Args:
        plots_dir: Output directory for plots.
        results: Benchmark run results.

    Returns:
        Path to generated plot, or None if no timing data.
    """
    # Collect all configs and their timing metrics
    config_names = []
    timings = []

    for test_result in results.test_results.values():
        for config_name, metrics in test_result.configs.items():
            if "timing_seconds" in metrics:
                config_names.append(config_name)
                timings.append(metrics["timing_seconds"])

    if not config_names:
        logger.warning("No timing metrics found - skipping timing bar chart")
        return None

    # Create bar chart
    x = np.arange(len(config_names))
    width = 0.6

    fig, ax = plt.subplots(figsize=(max(10, len(config_names) * 0.6), 6))
    ax.bar(x, timings, width, color="steelblue")
    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.title("Timing Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = plots_dir / "timing_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved timing bar chart: {output_path}")
    return output_path


def _plot_depth_comparisons(plots_dir: Path, results: BenchmarkRunResult) -> list[Path]:
    """Generate side-by-side depth map comparisons for each test.

    Creates subplots showing depth maps from different configs with shared colorbar.

    Args:
        plots_dir: Output directory for plots.
        results: Benchmark run results.

    Returns:
        List of generated comparison plot file paths.
    """
    generated = []

    # For each test, check if depth maps available
    for _test_name, _test_result in results.test_results.items():
        # Check if depth map data exists in config output directories
        # (Future: load depth maps from config_dir / "depth_map.npy")
        # For now, skip depth comparisons (no depth map loading implemented)
        pass

    return generated


__all__ = ["generate_visualizations"]
