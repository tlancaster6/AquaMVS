"""Markdown report generation and comparison visualizations."""

import logging
from pathlib import Path

import matplotlib

# Use non-interactive backend for headless environments
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .metrics import BenchmarkResults, total_keypoints, total_matches
from .visualization import render_comparison_grids

logger = logging.getLogger(__name__)


def generate_report(
    results: BenchmarkResults,
    output_dir: Path,
) -> Path:
    """Generate a comparative Markdown report with charts.

    Creates:
    - {output_dir}/benchmark/report.md -- Markdown report
    - {output_dir}/benchmark/comparison/*.png -- Chart images

    Args:
        results: Benchmark results from run_benchmark.
        output_dir: Root output directory (benchmark/ subdirectory will be created).

    Returns:
        Path to the generated report.md file.
    """
    # Create output directories
    report_dir = Path(output_dir) / "benchmark"
    comparison_dir = report_dir / "comparison"
    report_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Generate charts
    _generate_charts(results, comparison_dir)

    # Generate comparison grids
    config_names = [r.config_name for r in results.results]
    render_comparison_grids(config_names, results.camera_names, report_dir)

    # Generate Markdown
    report_path = report_dir / "report.md"
    _write_markdown(results, report_path)

    logger.info("Benchmark report generated: %s", report_path)
    return report_path


def _generate_charts(results: BenchmarkResults, output_dir: Path) -> None:
    """Generate comparison bar charts.

    Args:
        results: Benchmark results.
        output_dir: Directory to save charts.
    """
    # Try to use a clean style, fall back to default
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        # Style not available, use default
        pass

    # Extract data
    config_names = [r.config_name for r in results.results]
    keypoint_totals = [total_keypoints(r) for r in results.results]
    match_totals = [total_matches(r) for r in results.results]
    extraction_times = [r.extraction_time for r in results.results]
    matching_times = [r.matching_time for r in results.results]
    triangulation_times = [r.triangulation_time for r in results.results]

    # Color map by extractor type
    extractor_colors = {
        "superpoint": "#1f77b4",
        "aliked": "#ff7f0e",
        "disk": "#2ca02c",
    }
    colors = [extractor_colors.get(r.extractor_type, "#7f7f7f") for r in results.results]

    # Chart 1: Keypoint counts
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(config_names, keypoint_totals, color=colors)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Total Keypoints")
    ax.set_title("Keypoint Detection Counts")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "keypoint_counts.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Chart 2: Match counts
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(config_names, match_totals, color=colors)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Total Matches")
    ax.set_title("Feature Match Counts")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "match_counts.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Chart 3: Timing breakdown (stacked bar)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(config_names, extraction_times, label="Extraction", color="#1f77b4")
    ax.bar(
        config_names,
        matching_times,
        bottom=extraction_times,
        label="Matching",
        color="#ff7f0e",
    )
    bottom_tri = [e + m for e, m in zip(extraction_times, matching_times)]
    ax.bar(
        config_names,
        triangulation_times,
        bottom=bottom_tri,
        label="Triangulation",
        color="#2ca02c",
    )
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Time (s)")
    ax.set_title("Per-Stage Timing Breakdown")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "timing.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Generated 3 comparison charts in %s", output_dir)


def _write_markdown(results: BenchmarkResults, output_path: Path) -> None:
    """Write Markdown report to file.

    Args:
        results: Benchmark results.
        output_path: Path to output .md file.
    """
    lines = []

    # Header
    lines.append("# AquaMVS Benchmark Report")
    lines.append("")
    lines.append(f"Frame: {results.frame_idx}")
    lines.append(f"Configurations: {len(results.results)}")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Config | Keypoints | Matches | Sparse Points | Total Time |")
    lines.append("|--------|-----------|---------|---------------|------------|")
    for result in results.results:
        lines.append(
            f"| {result.config_name} | "
            f"{total_keypoints(result):,} | "
            f"{total_matches(result):,} | "
            f"{result.sparse_point_count:,} | "
            f"{result.total_time:.2f}s |"
        )
    lines.append("")

    # Per-stage timing table
    lines.append("## Per-Stage Timing")
    lines.append("")
    lines.append("| Config | Extraction | Matching | Triangulation | Total |")
    lines.append("|--------|------------|----------|---------------|-------|")
    for result in results.results:
        lines.append(
            f"| {result.config_name} | "
            f"{result.extraction_time:.3f}s | "
            f"{result.matching_time:.3f}s | "
            f"{result.triangulation_time:.3f}s | "
            f"{result.total_time:.3f}s |"
        )
    lines.append("")

    # Per-camera keypoints table
    lines.append("## Per-Camera Keypoints")
    lines.append("")
    # Header row
    header = "| Camera | " + " | ".join([r.config_name for r in results.results]) + " |"
    lines.append(header)
    separator = "|--------|" + "|".join(["---" for _ in results.results]) + "|"
    lines.append(separator)
    # Data rows
    for cam_name in results.camera_names:
        row = f"| {cam_name} | "
        row += " | ".join(
            [
                str(r.keypoint_counts.get(cam_name, 0))
                for r in results.results
            ]
        )
        row += " |"
        lines.append(row)
    lines.append("")

    # Visualizations
    lines.append("## Visualizations")
    lines.append("")
    lines.append("### Metrics")
    lines.append("")
    lines.append("![Keypoint Counts](comparison/keypoint_counts.png)")
    lines.append("")
    lines.append("![Match Counts](comparison/match_counts.png)")
    lines.append("")
    lines.append("![Timing Breakdown](comparison/timing.png)")
    lines.append("")
    lines.append("### Comparison Grids")
    lines.append("")
    lines.append("![Keypoints Grid](comparison/keypoints_grid.png)")
    lines.append("")
    lines.append("![Sparse Renders Grid](comparison/sparse_renders_grid.png)")
    lines.append("")
    lines.append("![Mesh Renders Grid](comparison/mesh_grid.png)")
    lines.append("")

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info("Markdown report written to %s", output_path)


__all__ = ["generate_report"]
