"""Benchmark suite for comparing feature extraction configurations and reconstruction accuracy."""

from .comparison import (
    ComparisonResult,
    MetricDelta,
    compare_runs,
    detect_regressions,
    format_comparison,
)
from .config import BenchmarkConfig, BenchmarkDataset, BenchmarkTests
from .datasets import DatasetContext, load_charuco_ground_truth, load_dataset
from .metrics import (
    BenchmarkResults,
    ConfigResult,
    compute_accuracy_metrics,
    compute_charuco_metrics,
    compute_plane_fit_metrics,
)
from .report import generate_report
from .runner import BenchmarkRunResult, TestResult, run_benchmark, run_benchmarks
from .visualization import (
    generate_visualizations,
    render_comparison_grids,
    render_config_outputs,
)

__all__ = [
    # Configuration models
    "BenchmarkConfig",
    "BenchmarkDataset",
    "BenchmarkTests",
    # Dataset loaders
    "DatasetContext",
    "load_dataset",
    "load_charuco_ground_truth",
    # Accuracy metrics (new)
    "compute_accuracy_metrics",
    "compute_charuco_metrics",
    "compute_plane_fit_metrics",
    # Benchmark orchestration (new)
    "run_benchmarks",
    "BenchmarkRunResult",
    "TestResult",
    # Feature extraction benchmark (legacy)
    "run_benchmark",
    "ConfigResult",
    "BenchmarkResults",
    "generate_report",
    "render_config_outputs",
    "render_comparison_grids",
    # Visualization (new)
    "generate_visualizations",
    # Comparison (new)
    "compare_runs",
    "detect_regressions",
    "format_comparison",
    "ComparisonResult",
    "MetricDelta",
]
