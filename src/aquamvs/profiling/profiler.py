"""PipelineProfiler class wrapping torch.profiler with memory tracking and CUDA warmup."""

from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from ..config import PipelineConfig
from .analyzer import ProfileReport, analyze_profile


class PipelineProfiler:
    """Wrapper around torch.profiler for pipeline profiling.

    Provides:
    - CUDA warmup before profiling
    - Memory tracking (CPU + GPU)
    - Chrome trace export
    - Structured report generation
    """

    def __init__(
        self,
        activities: list[str] | None = None,
        profile_memory: bool = True,
        record_shapes: bool = True,
        output_dir: Path | None = None,
    ):
        """Initialize profiler.

        Args:
            activities: List of activities to profile. Options: ["cpu", "cuda"].
                Defaults to ["cpu", "cuda"].
            profile_memory: Enable memory profiling (recommended for GPU bottlenecks).
            record_shapes: Record tensor shapes (helps identify size-dependent bottlenecks).
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

        self.profile_memory = profile_memory
        self.record_shapes = record_shapes
        self.output_dir = output_dir
        self.prof = None

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

        # Create profiler
        self.prof = profile(
            activities=self.activities,
            profile_memory=self.profile_memory,
            record_shapes=self.record_shapes,
            with_stack=False,  # Stack traces add overhead, disable for production
        )
        self.prof.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling and finalize."""
        if self.prof is not None:
            self.prof.__exit__(exc_type, exc_val, exc_tb)

    def stage(self, name: str):
        """Create a stage-level profiling context.

        Args:
            name: Stage name (e.g., "sparse_matching", "depth_estimation").

        Returns:
            Context manager for torch.profiler.record_function.
        """
        return record_function(name)

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

        return analyze_profile(self.prof)


def profile_pipeline(config: PipelineConfig, frame: int = 0) -> ProfileReport:
    """Convenience function to profile a single-frame pipeline run.

    Note: This is a placeholder function. Full implementation would require
    running the actual pipeline with profiler instrumentation. For now,
    this function demonstrates the intended usage pattern.

    Args:
        config: Pipeline configuration.
        frame: Frame index to profile (default 0).

    Returns:
        ProfileReport with performance metrics.

    Raises:
        NotImplementedError: This is a placeholder for future integration.
    """
    # Future implementation would:
    # 1. Create PipelineProfiler
    # 2. Run single-frame pipeline within profiler context
    # 3. Wrap each pipeline stage with record_function labels
    # 4. Return ProfileReport
    #
    # For now, raise NotImplementedError to indicate this is a stub.
    raise NotImplementedError(
        "profile_pipeline requires integration with pipeline.Pipeline class. "
        "Use PipelineProfiler directly via context manager instead."
    )
