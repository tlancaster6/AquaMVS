"""Pipeline orchestration package for multi-frame reconstruction.

Provides the pipeline context, builder, and helper functions for running
the full reconstruction pipeline.
"""

from .builder import build_pipeline_context, setup_pipeline
from .context import PipelineContext

__all__ = [
    "PipelineContext",
    "build_pipeline_context",
    "setup_pipeline",
]
