"""Benchmark comparison and regression detection."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from tabulate import tabulate

logger = logging.getLogger(__name__)


@dataclass
class MetricDelta:
    """Delta for a single metric between two runs.

    Attributes:
        metric_name: Name of the metric (e.g., "mean_error_mm").
        run1_value: Value from baseline run.
        run2_value: Value from current run.
        absolute_delta: Absolute difference (run2 - run1).
        percent_delta: Percentage change ((run2 - run1) / run1 * 100).
        is_regression: Whether this metric shows a regression.
    """

    metric_name: str
    run1_value: float
    run2_value: float
    absolute_delta: float
    percent_delta: float
    is_regression: bool


@dataclass
class ComparisonResult:
    """Results from comparing two benchmark runs.

    Attributes:
        run1_id: Baseline run identifier.
        run2_id: Current run identifier.
        metric_deltas: Mapping of metric names to MetricDelta objects.
        regressions: List of metric names that show regressions.
    """

    run1_id: str
    run2_id: str
    metric_deltas: dict[str, MetricDelta]
    regressions: list[str]


def compare_runs(run1_dir: Path, run2_dir: Path) -> ComparisonResult:
    """Compare two benchmark runs and detect regressions.

    Loads summary.json from both run directories, computes deltas for all
    metrics, and identifies regressions based on default thresholds:
    - Accuracy metrics (error, completeness): 5% threshold
    - Runtime metrics (timing): 10% threshold

    Args:
        run1_dir: Directory for baseline run.
        run2_dir: Directory for current run.

    Returns:
        ComparisonResult with per-metric deltas and regression flags.

    Raises:
        FileNotFoundError: If summary.json missing from either directory.
        ValueError: If run directories are identical.
    """
    run1_dir = Path(run1_dir)
    run2_dir = Path(run2_dir)

    if run1_dir == run2_dir:
        raise ValueError("Cannot compare a run to itself")

    # Load summary.json from both runs
    run1_summary_path = run1_dir / "summary.json"
    run2_summary_path = run2_dir / "summary.json"

    if not run1_summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {run1_summary_path}")
    if not run2_summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {run2_summary_path}")

    with open(run1_summary_path) as f:
        run1_data = json.load(f)

    with open(run2_summary_path) as f:
        run2_data = json.load(f)

    run1_id = run1_data["run_id"]
    run2_id = run2_data["run_id"]

    # Flatten metrics from both runs
    run1_metrics = _flatten_metrics(run1_data)
    run2_metrics = _flatten_metrics(run2_data)

    # Default thresholds per research recommendation
    thresholds = _default_thresholds(run1_metrics)

    # Detect regressions
    regression_results = detect_regressions(run1_metrics, run2_metrics, thresholds)

    # Build MetricDelta objects
    metric_deltas = {}
    regressions = []

    for metric_name, (
        baseline_val,
        current_val,
        is_regression,
    ) in regression_results.items():
        absolute_delta = current_val - baseline_val
        percent_delta = (
            (absolute_delta / baseline_val * 100.0) if baseline_val != 0 else 0.0
        )

        delta = MetricDelta(
            metric_name=metric_name,
            run1_value=baseline_val,
            run2_value=current_val,
            absolute_delta=absolute_delta,
            percent_delta=percent_delta,
            is_regression=is_regression,
        )

        metric_deltas[metric_name] = delta

        if is_regression:
            regressions.append(metric_name)

    return ComparisonResult(
        run1_id=run1_id,
        run2_id=run2_id,
        metric_deltas=metric_deltas,
        regressions=regressions,
    )


def detect_regressions(
    baseline: dict[str, float],
    current: dict[str, float],
    thresholds: dict[str, float],
) -> dict[str, tuple[float, float, bool]]:
    """Detect regressions by comparing current metrics to baseline.

    Regression rules:
    - Error metrics (contains "error"): regression if current > baseline by threshold
    - Coverage/completeness metrics (contains "completeness"): regression if current < baseline by threshold
    - Timing metrics (contains "time"): regression if current > baseline by threshold

    Args:
        baseline: Baseline metric values.
        current: Current metric values.
        thresholds: Per-metric percentage thresholds (e.g., {"mean_error_mm": 0.05}).

    Returns:
        Dict mapping metric name to (baseline_val, current_val, is_regression).
    """
    regressions = {}

    for metric, baseline_val in baseline.items():
        if metric not in current:
            continue

        current_val = current[metric]
        threshold_pct = thresholds.get(metric, 0.10)  # Default 10%

        # Avoid division by zero
        if baseline_val == 0:
            is_regression = False
        else:
            # For error and timing metrics: regression if current > baseline
            # For coverage/accuracy: regression if current < baseline
            if "error" in metric.lower() or "time" in metric.lower():
                is_regression = (
                    current_val - baseline_val
                ) / baseline_val > threshold_pct
            elif "completeness" in metric.lower():
                is_regression = (
                    baseline_val - current_val
                ) / baseline_val > threshold_pct
            else:
                # Unknown metric type - assume "higher is worse" (like error)
                is_regression = (
                    current_val - baseline_val
                ) / baseline_val > threshold_pct

        regressions[metric] = (baseline_val, current_val, is_regression)

    return regressions


def format_comparison(result: ComparisonResult) -> str:
    """Format comparison result as ASCII table.

    Creates a tabulate grid showing metric name, baseline value, current value,
    absolute delta, percent delta, and status (OK or REGRESSION).

    Args:
        result: Comparison result to format.

    Returns:
        ASCII table string.
    """
    # Build table rows
    headers = ["Metric", "Baseline", "Current", "Delta", "% Delta", "Status"]
    rows = []

    for metric_name, delta in result.metric_deltas.items():
        status = "REGRESSION" if delta.is_regression else "OK"

        row = [
            metric_name,
            f"{delta.run1_value:.2f}",
            f"{delta.run2_value:.2f}",
            f"{delta.absolute_delta:+.2f}",
            f"{delta.percent_delta:+.1f}%",
            status,
        ]
        rows.append(row)

    # Format table
    table = tabulate(rows, headers=headers, tablefmt="grid")

    # Add header with run IDs
    header = f"\nBenchmark Comparison: {result.run1_id} → {result.run2_id}\n"
    header += "=" * 80 + "\n"

    # Add regression summary
    if result.regressions:
        summary = f"\nRegressions detected: {len(result.regressions)}\n"
        summary += "  - " + "\n  - ".join(result.regressions) + "\n"
    else:
        summary = "\nNo regressions detected.\n"

    return header + table + "\n" + summary


def _flatten_metrics(run_data: dict) -> dict[str, float]:
    """Flatten nested metrics from summary.json into flat dict.

    Aggregates metrics across all tests by averaging.

    Args:
        run_data: Loaded summary.json data.

    Returns:
        Flat dict mapping metric name to averaged value.
    """
    all_metrics = {}

    # Collect metrics from test_results (per-config metrics)
    if "test_results" in run_data:
        for _test_name, test_data in run_data["test_results"].items():
            if "configs" in test_data:
                for _config_name, metrics in test_data["configs"].items():
                    for metric_name, value in metrics.items():
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = []
                        all_metrics[metric_name].append(value)

    # Average metrics across configs
    averaged = {
        metric: sum(values) / len(values) for metric, values in all_metrics.items()
    }

    # Add summary-level metrics
    if "summary" in run_data:
        averaged.update(run_data["summary"])

    return averaged


def _default_thresholds(metrics: dict[str, float]) -> dict[str, float]:
    """Generate default per-metric thresholds.

    Uses research recommendation:
    - Accuracy metrics (error, completeness): 5%
    - Runtime metrics (timing): 10%

    Args:
        metrics: Metric names to generate thresholds for.

    Returns:
        Dict mapping metric name to threshold percentage.
    """
    thresholds = {}

    for metric_name in metrics:
        # Error and completeness metrics: 5% threshold
        if "error" in metric_name.lower() or "completeness" in metric_name.lower():
            thresholds[metric_name] = 0.05
        # Timing metrics: 10% threshold
        elif "time" in metric_name.lower():
            thresholds[metric_name] = 0.10
        else:
            # Unknown metric: default 10%
            thresholds[metric_name] = 0.10

    return thresholds


__all__ = [
    "compare_runs",
    "detect_regressions",
    "format_comparison",
    "ComparisonResult",
    "MetricDelta",
]
