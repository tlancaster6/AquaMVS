"""Benchmark suite for comparing feature extraction configurations and reconstruction accuracy."""

from .config import BenchmarkConfig, BenchmarkDataset, BenchmarkTests
from .metrics import (
    BenchmarkResults,
    ConfigResult,
    compute_accuracy_metrics,
    compute_charuco_metrics,
    compute_plane_fit_metrics,
)
from .report import generate_report
from .runner import run_benchmark
from .visualization import render_comparison_grids, render_config_outputs

__all__ = [
    # Configuration models
    "BenchmarkConfig",
    "BenchmarkDataset",
    "BenchmarkTests",
    # Accuracy metrics (new)
    "compute_accuracy_metrics",
    "compute_charuco_metrics",
    "compute_plane_fit_metrics",
    # Feature extraction benchmark (legacy)
    "run_benchmark",
    "ConfigResult",
    "BenchmarkResults",
    "generate_report",
    "render_config_outputs",
    "render_comparison_grids",
]
