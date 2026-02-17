"""Benchmark report formatting (console table and markdown file)."""

import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runner import BenchmarkResult

# Stage names as they appear in profiler snapshots — mapped to display headers.
# Stages not present for a pathway show "—" in the table.
_STAGE_MAP = {
    "undistortion": "Undist (s)",
    "sparse_matching": "Match (s)",
    "dense_matching": "Match (s)",  # RoMa dense path uses a different key
    "depth_estimation": "Depth (s)",
    "fusion": "Fusion (s)",
    "surface": "Surface (s)",
}

_DISPLAY_STAGES = [
    ("undistortion", "Undist (s)"),
    ("sparse_matching", "Match (s)"),
    ("depth_estimation", "Depth (s)"),
    ("fusion", "Fusion (s)"),
    ("surface", "Surface (s)"),
]


def _get_stage_time(report, stage_name: str) -> str:
    """Get formatted wall time for a stage, or '—' if not present.

    Args:
        report: ProfileReport (may be None).
        stage_name: Stage name key to look up.

    Returns:
        Formatted time string (e.g. "1.2") or "—".
    """
    if report is None:
        return "—"
    stage = report.stages.get(stage_name)
    if stage is None:
        return "—"
    return f"{stage.wall_time_ms / 1000.0:.1f}"


def format_console_table(result: "BenchmarkResult") -> str:
    """Format benchmark results as a pretty-printed console table.

    Columns: Pathway | Undist (s) | Match (s) | Depth (s) | Fusion (s) |
             Surface (s) | Total (s) | Points | Density (pts/m²) | Outlier %

    Uses tabulate when available (already a project dependency). Stages not
    run by a pathway (e.g., depth/fusion in sparse mode) show "—".
    Times in seconds with 1 decimal place.

    Args:
        result: BenchmarkResult from run_benchmark.

    Returns:
        Formatted console table as a string.
    """
    try:
        from tabulate import tabulate

        _have_tabulate = True
    except ImportError:
        _have_tabulate = False

    headers = [
        "Pathway",
        "Undist (s)",
        "Match (s)",
        "Depth (s)",
        "Fusion (s)",
        "Surface (s)",
        "Total (s)",
        "Points",
        "Density (pts/m²)",
        "Outlier %",
    ]

    rows = []
    for pw in result.results:
        report = pw.timing
        total_s = report.total_time_ms / 1000.0 if report else 0.0

        # For "Match (s)": use sparse_matching; fall back to dense_matching for RoMa full
        match_time = _get_stage_time(report, "sparse_matching")
        if match_time == "—":
            match_time = _get_stage_time(report, "dense_matching")

        row = [
            pw.pathway_name,
            _get_stage_time(report, "undistortion"),
            match_time,
            _get_stage_time(report, "depth_estimation"),
            _get_stage_time(report, "fusion"),
            _get_stage_time(report, "surface"),
            f"{total_s:.1f}",
            f"{int(pw.point_count):,}" if pw.point_count > 0 else "0",
            f"{pw.cloud_density:.0f}" if pw.cloud_density > 0 else "0",
            f"{pw.outlier_removal_pct:.1f}" if pw.outlier_removal_pct > 0 else "—",
        ]
        rows.append(row)

    if _have_tabulate:
        return tabulate(rows, headers=headers, tablefmt="grid")
    else:
        # Minimal fallback without tabulate
        col_widths = [
            max(len(h), max((len(str(r[i])) for r in rows), default=0))
            for i, h in enumerate(headers)
        ]
        sep = "-+-".join("-" * w for w in col_widths)
        header_line = " | ".join(
            h.ljust(w) for h, w in zip(headers, col_widths, strict=False)
        )
        lines = [header_line, sep]
        for row in rows:
            lines.append(
                " | ".join(
                    str(v).ljust(w) for v, w in zip(row, col_widths, strict=False)
                )
            )
        return "\n".join(lines)


def format_markdown_report(result: "BenchmarkResult") -> str:
    """Format benchmark results as a markdown report.

    Includes:
    - Header with config path, frame index, timestamp
    - Results table (same layout as console)
    - System info (device, platform)

    Args:
        result: BenchmarkResult from run_benchmark.

    Returns:
        Markdown report content as a string.
    """
    import torch

    lines: list[str] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines.append("# AquaMVS Benchmark Report")
    lines.append("")
    lines.append(f"**Config:** `{result.config_path}`")
    lines.append(f"**Frame:** {result.frame}")
    lines.append(f"**Generated:** {timestamp}")
    lines.append("")

    # Results table
    lines.append("## Results")
    lines.append("")
    lines.append(
        "| Pathway | Undist (s) | Match (s) | Depth (s) | Fusion (s) "
        "| Surface (s) | Total (s) | Points | Density (pts/m²) | Outlier % |"
    )
    lines.append(
        "|---------|-----------|----------|----------|----------|"
        "-----------|----------|--------|----------------|-----------|"
    )

    for pw in result.results:
        report = pw.timing
        total_s = report.total_time_ms / 1000.0 if report else 0.0

        match_time = _get_stage_time(report, "sparse_matching")
        if match_time == "—":
            match_time = _get_stage_time(report, "dense_matching")

        lines.append(
            f"| {pw.pathway_name} "
            f"| {_get_stage_time(report, 'undistortion')} "
            f"| {match_time} "
            f"| {_get_stage_time(report, 'depth_estimation')} "
            f"| {_get_stage_time(report, 'fusion')} "
            f"| {_get_stage_time(report, 'surface')} "
            f"| {total_s:.1f} "
            f"| {int(pw.point_count):,} "
            f"| {pw.cloud_density:.0f} "
            f"| {'—' if pw.outlier_removal_pct == 0 else f'{pw.outlier_removal_pct:.1f}'} |"
        )

    lines.append("")

    # System info
    lines.append("## System Info")
    lines.append("")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lines.append(f"**Device:** {device}")
    if torch.cuda.is_available():
        lines.append(f"**CUDA:** {torch.version.cuda}")
        lines.append(f"**GPU:** {torch.cuda.get_device_name(0)}")
    lines.append(f"**Platform:** {platform.platform()}")
    lines.append(f"**Python:** {platform.python_version()}")
    lines.append(f"**PyTorch:** {torch.__version__}")
    lines.append("")

    return "\n".join(lines)


def save_markdown_report(result: "BenchmarkResult", output_dir: Path) -> Path:
    """Save markdown report to output_dir/benchmark_YYYYMMDD_HHMMSS.md.

    Args:
        result: BenchmarkResult from run_benchmark.
        output_dir: Directory where the report file will be saved.

    Returns:
        Path to the saved report file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"benchmark_{timestamp}.md"
    content = format_markdown_report(result)
    with open(report_path, "w") as f:
        f.write(content)
    return report_path


__all__ = ["format_console_table", "format_markdown_report", "save_markdown_report"]
