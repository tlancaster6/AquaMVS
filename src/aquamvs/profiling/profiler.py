"""PipelineProfiler using manual timing and memory snapshots.

Replaces torch.profiler (which OOMs on trace serialization for long pipeline
runs) with wall-clock timing, CUDA event timing, and explicit memory snapshots
at stage boundaries.
"""

import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch

from ..config import PipelineConfig
from .analyzer import ProfileReport, build_report


@dataclass
class StageSnapshot:
    """Timing and memory captured for a single pipeline stage."""

    wall_time_ms: float = 0.0
    cuda_time_ms: float = 0.0
    cpu_memory_peak_mb: float = 0.0
    cuda_memory_delta_mb: float = 0.0
    cuda_memory_peak_mb: float = 0.0


class PipelineProfiler:
    """Lightweight pipeline profiler using manual timing and memory tracking.

    Uses time.perf_counter for wall-clock timing, torch.cuda.Event for GPU
    timing, tracemalloc for CPU memory, and torch.cuda memory APIs for GPU
    memory. Avoids torch.profiler entirely to prevent OOM on trace
    serialization during long pipeline runs.
    """

    def __init__(self, output_dir: Path | None = None):
        """Initialize profiler.

        Args:
            output_dir: Directory for output files (optional, reserved for
                future use).
        """
        self.output_dir = output_dir
        self.snapshots: dict[str, StageSnapshot] = {}
        self._has_cuda = torch.cuda.is_available()

    def _cuda_warmup(self):
        """Perform CUDA warmup to avoid cold-start overhead in profile."""
        if self._has_cuda:
            dummy = torch.randn(100, 100, device="cuda")
            _ = dummy @ dummy
            torch.cuda.synchronize()

    def __enter__(self):
        """Start profiling context."""
        self._cuda_warmup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling context (no-op, all work done in stage())."""

    @contextmanager
    def stage(self, name: str):
        """Profile a pipeline stage, capturing timing and memory.

        Args:
            name: Stage name (e.g., "sparse_matching", "depth_estimation").

        Yields:
            None.
        """
        # --- Pre-stage snapshots ---
        if self._has_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            cuda_before = torch.cuda.memory_allocated()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        tracemalloc.start()
        wall_start = time.perf_counter()

        yield

        # --- Post-stage snapshots ---
        wall_end = time.perf_counter()
        _, cpu_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        snap = StageSnapshot(
            wall_time_ms=(wall_end - wall_start) * 1000.0,
            cpu_memory_peak_mb=cpu_peak / (1024 * 1024),
        )

        if self._has_cuda:
            end_event.record()
            torch.cuda.synchronize()
            snap.cuda_time_ms = start_event.elapsed_time(end_event)
            cuda_after = torch.cuda.memory_allocated()
            snap.cuda_memory_delta_mb = (cuda_after - cuda_before) / (1024 * 1024)
            snap.cuda_memory_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        self.snapshots[name] = snap

    def get_report(self) -> ProfileReport:
        """Generate structured profile report from collected snapshots.

        Returns:
            ProfileReport with per-stage metrics and bottleneck identification.
        """
        return build_report(self.snapshots, has_cuda=self._has_cuda)


def profile_pipeline(config: PipelineConfig, frame: int = 0) -> ProfileReport:
    """Convenience function to profile a single-frame pipeline run.

    Args:
        config: Pipeline configuration.
        frame: Frame index to profile (default 0).

    Returns:
        ProfileReport with performance metrics.
    """
    from aquacal.io.video import VideoSet

    from ..io import ImageDirectorySet, detect_input_type
    from ..pipeline.builder import build_pipeline_context
    from ..pipeline.runner import process_frame

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    ctx = build_pipeline_context(config)

    input_type = detect_input_type(config.camera_input_map)
    if input_type == "images":
        context_manager = ImageDirectorySet(config.camera_input_map)
    else:
        context_manager = VideoSet(config.camera_input_map)

    with context_manager as source:
        raw_images = source.read_frame(frame)

    profiler = PipelineProfiler(output_dir=Path(config.output_dir))

    with profiler:
        process_frame(frame, raw_images, ctx)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return profiler.get_report()
