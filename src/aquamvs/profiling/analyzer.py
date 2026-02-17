"""Profile report construction and formatting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .profiler import StageSnapshot


@dataclass
class StageProfile:
    """Performance metrics for a single pipeline stage."""

    name: str
    wall_time_ms: float
    cuda_time_ms: float
    cpu_memory_peak_mb: float
    cuda_memory_peak_mb: float


@dataclass
class ProfileReport:
    """Aggregated profile report with bottleneck identification."""

    stages: dict[str, StageProfile]
    top_bottlenecks: list[tuple[str, float, float]]  # (name, time_ms, memory_mb)
    total_time_ms: float
    total_memory_peak_mb: float
    device: str


def build_report(
    snapshots: dict[str, StageSnapshot],
    has_cuda: bool = False,
) -> ProfileReport:
    """Build a ProfileReport from collected stage snapshots.

    Args:
        snapshots: Per-stage timing and memory snapshots from PipelineProfiler.
        has_cuda: Whether CUDA was available during profiling.

    Returns:
        ProfileReport with per-stage metrics and top 3 bottlenecks.
    """
    stages: dict[str, StageProfile] = {}
    total_time = 0.0
    total_memory = 0.0

    for name, snap in snapshots.items():
        stage = StageProfile(
            name=name,
            wall_time_ms=snap.wall_time_ms,
            cuda_time_ms=snap.cuda_time_ms,
            cpu_memory_peak_mb=snap.cpu_memory_peak_mb,
            cuda_memory_peak_mb=snap.cuda_memory_peak_mb,
        )
        stages[name] = stage
        total_time += snap.wall_time_ms
        total_memory = max(
            total_memory, snap.cpu_memory_peak_mb + snap.cuda_memory_peak_mb
        )

    # Identify top 3 bottlenecks by wall time
    bottlenecks = []
    for name, stage in stages.items():
        total_mem = stage.cpu_memory_peak_mb + stage.cuda_memory_peak_mb
        bottlenecks.append((name, stage.wall_time_ms, total_mem))

    bottlenecks.sort(key=lambda x: x[1], reverse=True)
    top_bottlenecks = bottlenecks[:3]

    device = "cuda" if has_cuda else "cpu"

    return ProfileReport(
        stages=stages,
        top_bottlenecks=top_bottlenecks,
        total_time_ms=total_time,
        total_memory_peak_mb=total_memory,
        device=device,
    )


def format_report(report: ProfileReport) -> str:
    """Format ProfileReport as human-readable ASCII table.

    Args:
        report: ProfileReport to format.

    Returns:
        Formatted report string with stage breakdown and bottleneck highlights.
    """
    try:
        from tabulate import tabulate
    except ImportError:
        return _format_report_simple(report)

    lines = []
    lines.append(f"Profile Report (device: {report.device})")
    lines.append(f"Total time: {report.total_time_ms:.2f} ms")
    lines.append(f"Peak memory: {report.total_memory_peak_mb:.2f} MB")
    lines.append("")

    headers = ["Stage", "Wall (ms)", "CUDA (ms)", "CPU Mem (MB)", "CUDA Mem (MB)"]
    table_data = []
    for name, stage in report.stages.items():
        table_data.append(
            [
                name,
                f"{stage.wall_time_ms:.2f}",
                f"{stage.cuda_time_ms:.2f}",
                f"{stage.cpu_memory_peak_mb:.2f}",
                f"{stage.cuda_memory_peak_mb:.2f}",
            ]
        )

    lines.append(tabulate(table_data, headers=headers, tablefmt="grid"))
    lines.append("")

    lines.append("Top 3 Bottlenecks:")
    for i, (name, time_ms, memory_mb) in enumerate(report.top_bottlenecks, 1):
        lines.append(f"  {i}. {name}: {time_ms:.2f} ms, {memory_mb:.2f} MB")

    return "\n".join(lines)


def _format_report_simple(report: ProfileReport) -> str:
    """Simple fallback formatter without tabulate dependency."""
    lines = []
    lines.append(f"Profile Report (device: {report.device})")
    lines.append(f"Total time: {report.total_time_ms:.2f} ms")
    lines.append(f"Peak memory: {report.total_memory_peak_mb:.2f} MB")
    lines.append("")
    lines.append("Stages:")
    for name, stage in report.stages.items():
        total_mem = stage.cpu_memory_peak_mb + stage.cuda_memory_peak_mb
        lines.append(
            f"  {name}: {stage.wall_time_ms:.2f} ms "
            f"(CUDA: {stage.cuda_time_ms:.2f} ms), {total_mem:.2f} MB"
        )
    lines.append("")
    lines.append("Top 3 Bottlenecks:")
    for i, (name, time_ms, memory_mb) in enumerate(report.top_bottlenecks, 1):
        lines.append(f"  {i}. {name}: {time_ms:.2f} ms, {memory_mb:.2f} MB")
    return "\n".join(lines)
