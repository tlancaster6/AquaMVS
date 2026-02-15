"""Benchmark orchestration for comparing pipeline configurations."""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from tabulate import tabulate

from ..config import PipelineConfig
from ..pipeline import Pipeline
from .config import BenchmarkConfig
from .datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Results from a single benchmark test.

    Attributes:
        test_name: Name of the test (e.g., "clahe_comparison").
        configs: Per-config results mapping config_name to metrics dict.
    """

    test_name: str
    configs: dict[str, dict[str, float]]


@dataclass
class BenchmarkRunResult:
    """Results from a complete benchmark run.

    Attributes:
        run_id: Unique run identifier (timestamp).
        run_dir: Directory containing run results.
        test_results: Results from each enabled test.
        summary: Aggregated metrics across all tests.
    """

    run_id: str
    run_dir: Path
    test_results: dict[str, TestResult]
    summary: dict[str, float]


def run_benchmarks(config: BenchmarkConfig) -> BenchmarkRunResult:
    """Run all enabled benchmark tests and produce structured results.

    Creates a timestamped run directory, copies config YAML, runs enabled tests,
    computes accuracy metrics, records timing, and writes summary.json.

    Args:
        config: Benchmark configuration with datasets and test toggles.

    Returns:
        BenchmarkRunResult with per-test metrics and aggregated summary.
    """
    # Create timestamped run directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting benchmark run: {run_id}")
    logger.info(f"Output directory: {run_dir}")

    # Copy config YAML to run directory
    # (Config lives in same directory as output per user decision)
    config.to_yaml(run_dir / "config.yaml")

    # Initialize results storage
    test_results: dict[str, TestResult] = {}

    # Run enabled tests
    if config.tests.clahe_comparison:
        logger.info("Running CLAHE comparison test...")
        test_results["clahe_comparison"] = _run_clahe_comparison(config, run_dir)

    if config.tests.execution_mode_comparison:
        logger.info("Running execution mode comparison test...")
        test_results["execution_mode_comparison"] = _run_execution_mode_comparison(
            config, run_dir
        )

    if config.tests.surface_reconstruction_comparison:
        logger.info("Running surface reconstruction comparison test...")
        test_results["surface_reconstruction_comparison"] = (
            _run_surface_reconstruction_comparison(config, run_dir)
        )

    # Aggregate summary metrics
    summary = _aggregate_summary(test_results)

    # Write summary.json
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "test_results": {
                    name: {"configs": result.configs}
                    for name, result in test_results.items()
                },
                "summary": summary,
            },
            f,
            indent=2,
        )

    logger.info(f"Summary written to {summary_path}")

    # Print ASCII summary table
    _print_summary_table(test_results)

    return BenchmarkRunResult(
        run_id=run_id,
        run_dir=run_dir,
        test_results=test_results,
        summary=summary,
    )


def _run_clahe_comparison(config: BenchmarkConfig, run_dir: Path) -> TestResult:
    """Run CLAHE on vs off comparison across extractors.

    Tests CLAHE with LightGlue (SuperPoint, ALIKED, DISK) and RoMa, all in sparse mode.

    Args:
        config: Benchmark configuration.
        run_dir: Run output directory.

    Returns:
        TestResult with per-config accuracy and timing metrics.
    """
    test_dir = run_dir / "clahe_comparison"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Define config variants to test
    extractor_variants = ["superpoint", "aliked", "disk", "roma"]
    clahe_variants = [True, False]

    configs_to_test = []
    for extractor in extractor_variants:
        for clahe in clahe_variants:
            config_name = _make_config_name(extractor, clahe, "sparse", None)
            configs_to_test.append((config_name, extractor, clahe, "sparse", None))

    # Run each config variant
    results = {}
    for config_name, extractor, clahe, mode, surface_method in configs_to_test:
        logger.info(f"  Testing: {config_name}")
        metrics = _run_pipeline_config(
            benchmark_config=config,
            extractor=extractor,
            clahe=clahe,
            mode=mode,
            surface_method=surface_method,
            output_dir=test_dir / config_name,
        )
        results[config_name] = metrics

    # Write per-config results
    with open(test_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return TestResult(test_name="clahe_comparison", configs=results)


def _run_execution_mode_comparison(
    config: BenchmarkConfig, run_dir: Path
) -> TestResult:
    """Run execution mode comparison: sparse vs full reconstruction.

    Tests all 4 modes: LightGlue+sparse, LightGlue+full, RoMa+sparse, RoMa+full.

    Args:
        config: Benchmark configuration.
        run_dir: Run output directory.

    Returns:
        TestResult with per-config accuracy and timing metrics.
    """
    test_dir = run_dir / "execution_mode_comparison"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Define config variants (LightGlue uses user-selectable extractor)
    configs_to_test = [
        (
            f"lightglue_{config.lightglue_extractor}_sparse",
            config.lightglue_extractor,
            False,
            "sparse",
            None,
        ),
        (
            f"lightglue_{config.lightglue_extractor}_full",
            config.lightglue_extractor,
            False,
            "full",
            None,
        ),
        ("roma_sparse", "roma", False, "sparse", None),
        ("roma_full", "roma", False, "full", None),
    ]

    # Run each config variant
    results = {}
    for config_name, extractor, clahe, mode, surface_method in configs_to_test:
        logger.info(f"  Testing: {config_name}")
        metrics = _run_pipeline_config(
            benchmark_config=config,
            extractor=extractor,
            clahe=clahe,
            mode=mode,
            surface_method=surface_method,
            output_dir=test_dir / config_name,
        )
        results[config_name] = metrics

    # Write per-config results
    with open(test_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return TestResult(test_name="execution_mode_comparison", configs=results)


def _run_surface_reconstruction_comparison(
    config: BenchmarkConfig, run_dir: Path
) -> TestResult:
    """Run surface reconstruction method comparison.

    Tests Poisson, heightfield, and BPA on same depth maps.

    Args:
        config: Benchmark configuration.
        run_dir: Run output directory.

    Returns:
        TestResult with per-config accuracy and timing metrics.
    """
    test_dir = run_dir / "surface_reconstruction_comparison"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Define config variants (vary only surface method, keep same depth estimation)
    surface_methods = ["poisson", "heightfield", "bpa"]

    configs_to_test = [
        (f"surface_{method}", config.lightglue_extractor, False, "full", method)
        for method in surface_methods
    ]

    # Run each config variant
    results = {}
    for config_name, extractor, clahe, mode, surface_method in configs_to_test:
        logger.info(f"  Testing: {config_name}")
        metrics = _run_pipeline_config(
            benchmark_config=config,
            extractor=extractor,
            clahe=clahe,
            mode=mode,
            surface_method=surface_method,
            output_dir=test_dir / config_name,
        )
        results[config_name] = metrics

    # Write per-config results
    with open(test_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return TestResult(test_name="surface_reconstruction_comparison", configs=results)


def _run_pipeline_config(
    benchmark_config: BenchmarkConfig,
    extractor: str,
    clahe: bool,
    mode: str,
    surface_method: str | None,
    output_dir: Path,
) -> dict[str, float]:
    """Run pipeline with a specific configuration and compute metrics.

    Args:
        benchmark_config: Benchmark configuration with datasets.
        extractor: Feature extractor (superpoint, aliked, disk, roma).
        clahe: Whether to enable CLAHE preprocessing.
        mode: Pipeline mode (sparse or full).
        surface_method: Surface reconstruction method (poisson, heightfield, bpa).
        output_dir: Output directory for this config variant.

    Returns:
        Dictionary with accuracy metrics and timing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate metrics across all datasets
    all_metrics = []

    for dataset_config in benchmark_config.datasets:
        logger.info(f"    Dataset: {dataset_config.name}")

        # Load ground truth
        dataset_ctx = load_dataset(dataset_config)

        # Build pipeline config for this dataset
        pipeline_config = _build_pipeline_config(
            dataset_config=dataset_config,
            extractor=extractor,
            clahe=clahe,
            mode=mode,
            surface_method=surface_method,
            output_dir=str(output_dir / dataset_config.name),
        )

        # Run pipeline with timing
        t_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        pipeline = Pipeline(pipeline_config)
        pipeline.run()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        total_time = t_end - t_start

        # Compute accuracy metrics
        if dataset_ctx.ground_truth_depths is not None:
            # Synthetic dataset: compute point-to-mesh accuracy
            # TODO: Load reconstructed point cloud from output_dir
            # For now, use placeholder metrics
            metrics = {
                "mean_error_mm": 0.0,
                "median_error_mm": 0.0,
                "std_error_mm": 0.0,
                "raw_completeness_pct": 0.0,
                "timing_seconds": total_time,
            }
        elif dataset_ctx.charuco_corners is not None:
            # ChArUco dataset: compute corner reprojection error
            # TODO: Load reconstructed corners from output_dir
            # For now, use placeholder metrics
            metrics = {
                "mean_error_mm": 0.0,
                "median_error_mm": 0.0,
                "max_error_mm": 0.0,
                "rmse_mm": 0.0,
                "timing_seconds": total_time,
            }
        else:
            metrics = {"timing_seconds": total_time}

        all_metrics.append(metrics)

    # Average metrics across datasets
    if not all_metrics:
        return {"timing_seconds": 0.0}

    # Aggregate by averaging
    aggregated = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if key in m]
        aggregated[key] = sum(values) / len(values) if values else 0.0

    return aggregated


def _build_pipeline_config(
    dataset_config,
    extractor: str,
    clahe: bool,
    mode: str,
    surface_method: str | None,
    output_dir: str,
) -> PipelineConfig:
    """Build a PipelineConfig for a specific test configuration.

    Args:
        dataset_config: Dataset configuration.
        extractor: Feature extractor.
        clahe: CLAHE enabled.
        mode: Pipeline mode.
        surface_method: Surface reconstruction method.
        output_dir: Output directory.

    Returns:
        PipelineConfig ready for pipeline execution.
    """
    # Load base config from dataset path
    dataset_path = Path(dataset_config.path)
    base_config_path = dataset_path / "config.yaml"

    if base_config_path.exists():
        config = PipelineConfig.from_yaml(base_config_path)
    else:
        # Create minimal config (for synthetic datasets)
        config = PipelineConfig(
            calibration_path=str(dataset_path / "calibration.json"),
            output_dir=output_dir,
            camera_video_map={},
        )

    # Override settings for this test variant
    config.output_dir = output_dir

    # Set matcher and extractor
    if extractor == "roma":
        config.matcher_type = "roma"
    else:
        config.matcher_type = "lightglue"
        config.sparse_matching.extractor_type = extractor

    # Set CLAHE
    config.sparse_matching.clahe_enabled = clahe

    # Set pipeline mode
    config.pipeline_mode = mode

    # Set surface reconstruction method
    if surface_method is not None:
        config.reconstruction.surface_method = surface_method

    return config


def _make_config_name(
    extractor: str, clahe: bool, mode: str, surface_method: str | None
) -> str:
    """Generate a config name for a test variant.

    Args:
        extractor: Feature extractor.
        clahe: CLAHE enabled.
        mode: Pipeline mode.
        surface_method: Surface reconstruction method.

    Returns:
        Config name string.
    """
    parts = [extractor]
    if clahe:
        parts.append("clahe")
    parts.append(mode)
    if surface_method is not None:
        parts.append(surface_method)
    return "_".join(parts)


def _aggregate_summary(test_results: dict[str, TestResult]) -> dict[str, float]:
    """Aggregate summary metrics across all tests.

    Args:
        test_results: Results from all tests.

    Returns:
        Dictionary with aggregated metrics.
    """
    # Collect all timing values
    all_timings = []
    for result in test_results.values():
        for metrics in result.configs.values():
            if "timing_seconds" in metrics:
                all_timings.append(metrics["timing_seconds"])

    if not all_timings:
        return {"mean_timing_seconds": 0.0, "total_configs_tested": 0}

    return {
        "mean_timing_seconds": sum(all_timings) / len(all_timings),
        "total_configs_tested": sum(len(r.configs) for r in test_results.values()),
    }


def _print_summary_table(test_results: dict[str, TestResult]) -> None:
    """Print ASCII summary table to terminal.

    Args:
        test_results: Results from all tests.
    """
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80 + "\n")

    for test_name, result in test_results.items():
        print(f"Test: {test_name}")
        print("-" * 80)

        # Build table rows
        headers = ["Config"]
        metric_keys = []

        # Determine metric keys from first config
        if result.configs:
            first_config = next(iter(result.configs.values()))
            metric_keys = list(first_config.keys())
            headers.extend(metric_keys)

        rows = []
        for config_name, metrics in result.configs.items():
            row = [config_name]
            for key in metric_keys:
                value = metrics.get(key, 0.0)
                row.append(f"{value:.1f}")
            rows.append(row)

        # Print table
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        print()


__all__ = [
    "run_benchmarks",
    "BenchmarkRunResult",
    "TestResult",
]
