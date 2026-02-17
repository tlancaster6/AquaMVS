"""PipelineProfiler class wrapping torch.profiler with memory tracking and CUDA warmup."""

import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from ..config import PipelineConfig
from .analyzer import ProfileReport, analyze_profile


@dataclass
class MemorySnapshot:
    """Memory usage captured at stage boundaries."""

    cpu_delta_mb: float = 0.0
    cuda_delta_mb: float = 0.0
    cuda_peak_mb: float = 0.0


class PipelineProfiler:
    """Wrapper around torch.profiler for pipeline profiling.

    Provides:
    - CUDA warmup before profiling
    - Manual memory tracking at stage boundaries (avoids profiler OOM)
    - Chrome trace export
    - Structured report generation

    Memory is tracked using torch.cuda APIs and tracemalloc rather than the
    profiler's built-in profile_memory option, which can cause OOM when
    serializing traces from long pipeline runs.
    """

    def __init__(
        self,
        activities: list[str] | None = None,
        record_shapes: bool = False,
        output_dir: Path | None = None,
    ):
        """Initialize profiler.

        Args:
            activities: List of activities to profile. Options: ["cpu", "cuda"].
                Defaults to ["cpu", "cuda"].
            record_shapes: Record tensor shapes. Disabled by default because it
                significantly increases trace data size and can cause OOM
                when the profiler serializes results.
            output_dir: Directory for Chrome trace output (optional).
        """
        if activities is None:
            activities = ["cpu", "cuda"]

        # Map string activities to ProfilerActivity enum
        self.activities = []
        if "cpu" in activities:
            self.activities.append(ProfilerActivity.CPU)
        if "cuda" in activities:
            self.activities.append(ProfilerActivity.CUDA)

        self.record_shapes = record_shapes
        self.output_dir = output_dir
        self.prof = None
        self.memory_snapshots: dict[str, MemorySnapshot] = {}

    def _cuda_warmup(self):
        """Perform CUDA warmup to avoid cold-start overhead in profile."""
        if torch.cuda.is_available():
            # Run a simple tensor operation and synchronize
            dummy = torch.randn(100, 100, device="cuda")
            _ = dummy @ dummy
            torch.cuda.synchronize()

    def __enter__(self):
        """Start profiling context."""
        # CUDA warmup before profiling starts
        self._cuda_warmup()

        # Create profiler — profile_memory=False to avoid trace OOM
        self.prof = profile(
            activities=self.activities,
            profile_memory=False,
            record_shapes=self.record_shapes,
            with_stack=False,
        )
        self.prof.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling and finalize."""
        if self.prof is not None:
            self.prof.__exit__(exc_type, exc_val, exc_tb)

    @contextmanager
    def stage(self, name: str):
        """Create a stage-level profiling context with memory tracking.

        Wraps torch.profiler.record_function for timing and additionally
        captures CPU (via tracemalloc) and CUDA memory deltas.

        Args:
            name: Stage name (e.g., "sparse_matching", "depth_estimation").

        Yields:
            None.
        """
        has_cuda = torch.cuda.is_available()

        # Snapshot memory before
        if has_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            cuda_before = torch.cuda.memory_allocated()

        tracemalloc.start()

        with record_function(name):
            yield

        # Snapshot memory after
        _, cpu_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        snap = MemorySnapshot(cpu_delta_mb=cpu_peak / (1024 * 1024))

        if has_cuda:
            torch.cuda.synchronize()
            cuda_after = torch.cuda.memory_allocated()
            cuda_peak = torch.cuda.max_memory_allocated()
            snap.cuda_delta_mb = (cuda_after - cuda_before) / (1024 * 1024)
            snap.cuda_peak_mb = cuda_peak / (1024 * 1024)

        self.memory_snapshots[name] = snap

    def export_chrome_trace(self, path: Path) -> None:
        """Export Chrome trace JSON for visualization.

        Args:
            path: Output path for trace JSON file.
        """
        if self.prof is None:
            raise RuntimeError("Profiler must be run before exporting trace")

        self.prof.export_chrome_trace(str(path))

    def get_report(self) -> ProfileReport:
        """Generate structured profile report.

        Returns:
            ProfileReport with stage metrics and bottleneck identification.
        """
        if self.prof is None:
            raise RuntimeError("Profiler must be run before generating report")

        return analyze_profile(self.prof, self.memory_snapshots)


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

    # CUDA sync before profiling
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Build pipeline context
    ctx = build_pipeline_context(config)

    # Detect input type and open appropriate reader
    input_type = detect_input_type(config.camera_input_map)

    if input_type == "images":
        context_manager = ImageDirectorySet(config.camera_input_map)
    else:
        context_manager = VideoSet(config.camera_input_map)

    with context_manager as source:
        raw_images = source.read_frame(frame)

    # Create profiler and run single frame
    profiler = PipelineProfiler(output_dir=Path(config.output_dir))

    with profiler:
        process_frame(frame, raw_images, ctx)

    # CUDA sync after profiling
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return profiler.get_report()
