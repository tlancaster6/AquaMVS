"""Profile result parsing, sorting, and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .profiler import MemorySnapshot


@dataclass
class StageProfile:
    """Performance metrics for a single pipeline stage."""

    name: str
    cpu_time_ms: float
    cuda_time_ms: float
    self_cpu_time_ms: float
    self_cuda_time_ms: float
    cpu_memory_mb: float
    cuda_memory_mb: float


@dataclass
class ProfileReport:
    """Aggregated profile report with bottleneck identification."""

    stages: dict[str, StageProfile]
    top_bottlenecks: list[tuple[str, float, float]]  # (name, time_ms, memory_mb)
    total_time_ms: float
    total_memory_peak_mb: float
    device: str


def analyze_profile(
    prof: torch.profiler.profile,
    memory_snapshots: dict[str, MemorySnapshot] | None = None,
) -> ProfileReport:
    """Analyze torch.profiler results and identify bottlenecks.

    Args:
        prof: Completed torch.profiler.profile instance.
        memory_snapshots: Per-stage memory snapshots captured by PipelineProfiler.
            When provided, these are used instead of profiler event memory fields
            (which require profile_memory=True and can cause OOM).

    Returns:
        ProfileReport with per-stage metrics and top 3 bottlenecks.
    """
    if memory_snapshots is None:
        memory_snapshots = {}

    # Get key averages grouped by input shapes
    key_avg = prof.key_averages()

    # Collect all events by name
    event_map = {}
    total_time = 0.0

    for evt in key_avg:
        name = evt.key
        cpu_time = evt.cpu_time_total / 1000.0  # us to ms
        cuda_time = evt.cuda_time_total / 1000.0  # us to ms
        self_cpu_time = evt.self_cpu_time_total / 1000.0  # us to ms
        self_cuda_time = evt.self_cuda_time_total / 1000.0  # us to ms

        event_map[name] = {
            "cpu_time_ms": cpu_time,
            "cuda_time_ms": cuda_time,
            "self_cpu_time_ms": self_cpu_time,
            "self_cuda_time_ms": self_cuda_time,
        }

        total_time = max(total_time, cpu_time + cuda_time)

    # Extract stage profiles (filter for known stage names)
    stage_names = [
        "undistortion",
        "sparse_matching",
        "dense_matching",
        "depth_estimation",
        "fusion",
        "surface_reconstruction",
        "build_cost_volume",
        "grid_sample_warp",
        "extract_depth",
    ]

    stage_data = {}
    total_memory = 0.0

    for stage_name in stage_names:
        if stage_name in event_map:
            evt_data = event_map[stage_name]

            # Use manual memory snapshots if available, otherwise zeros
            snap = memory_snapshots.get(stage_name)
            cpu_mem = snap.cpu_delta_mb if snap else 0.0
            cuda_mem = snap.cuda_peak_mb if snap else 0.0

            stage_data[stage_name] = StageProfile(
                name=stage_name,
                cpu_time_ms=evt_data["cpu_time_ms"],
                cuda_time_ms=evt_data["cuda_time_ms"],
                self_cpu_time_ms=evt_data["self_cpu_time_ms"],
                self_cuda_time_ms=evt_data["self_cuda_time_ms"],
                cpu_memory_mb=cpu_mem,
                cuda_memory_mb=cuda_mem,
            )

            total_memory = max(total_memory, cpu_mem + cuda_mem)

    # Identify top 3 bottlenecks by total time (CPU + CUDA)
    bottlenecks = []
    for name, stage in stage_data.items():
        total_stage_time = stage.cpu_time_ms + stage.cuda_time_ms
        total_stage_memory = abs(stage.cpu_memory_mb) + abs(stage.cuda_memory_mb)
        bottlenecks.append((name, total_stage_time, total_stage_memory))

    # Sort by time descending
    bottlenecks.sort(key=lambda x: x[1], reverse=True)
    top_bottlenecks = bottlenecks[:3]

    # Detect device
    device = "cuda" if any(s.cuda_time_ms > 0 for s in stage_data.values()) else "cpu"

    return ProfileReport(
        stages=stage_data,
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
        # Fallback if tabulate not available
        return _format_report_simple(report)

    lines = []
    lines.append(f"Profile Report (device: {report.device})")
    lines.append(f"Total time: {report.total_time_ms:.2f} ms")
    lines.append(f"Peak memory: {report.total_memory_peak_mb:.2f} MB")
    lines.append("")

    # Stage breakdown table
    table_data = []
    for name, stage in report.stages.items():
        total_time = stage.cpu_time_ms + stage.cuda_time_ms
        total_mem = abs(stage.cpu_memory_mb) + abs(stage.cuda_memory_mb)
        table_data.append(
            [
                name,
                f"{stage.cpu_time_ms:.2f}",
                f"{stage.cuda_time_ms:.2f}",
                f"{total_time:.2f}",
                f"{total_mem:.2f}",
            ]
        )

    headers = ["Stage", "CPU (ms)", "CUDA (ms)", "Total (ms)", "Memory (MB)"]
    lines.append(tabulate(table_data, headers=headers, tablefmt="grid"))
    lines.append("")

    # Top bottlenecks
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
        total_time = stage.cpu_time_ms + stage.cuda_time_ms
        total_mem = abs(stage.cpu_memory_mb) + abs(stage.cuda_memory_mb)
        lines.append(
            f"  {name}: {total_time:.2f} ms (CPU: {stage.cpu_time_ms:.2f}, "
            f"CUDA: {stage.cuda_time_ms:.2f}), {total_mem:.2f} MB"
        )
    lines.append("")
    lines.append("Top 3 Bottlenecks:")
    for i, (name, time_ms, memory_mb) in enumerate(report.top_bottlenecks, 1):
        lines.append(f"  {i}. {name}: {time_ms:.2f} ms, {memory_mb:.2f} MB")
    return "\n".join(lines)
