"""Pipeline orchestration package for multi-frame reconstruction.

Provides the pipeline context, builder, and helper functions for running
the full reconstruction pipeline.
"""

from .builder import build_pipeline_context, setup_pipeline
from .context import PipelineContext
from .runner import Pipeline, process_frame, run_pipeline
from .visualization import run_visualization_pass

__all__ = [
    "Pipeline",
    "PipelineContext",
    "build_pipeline_context",
    "setup_pipeline",
    "process_frame",
    "run_pipeline",
    "run_visualization_pass",
]
