"""Benchmark runner for comparing pipeline execution pathways."""

import copy
import gc
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..config import PipelineConfig
from ..pipeline.visualization import run_visualization_pass
from ..profiling.analyzer import ProfileReport
from ..profiling.profiler import PipelineProfiler, set_active_profiler

logger = logging.getLogger(__name__)


@dataclass
class PathwayResult:
    """Results from a single pathway execution.

    Attributes:
        pathway_name: Human-readable name for this pathway configuration.
        timing: ProfileReport with per-stage wall times and memory metrics.
        point_count: Number of points in the fused cloud (0 if no cloud).
        cloud_density: Points per m² (bounding-box XY area proxy, 0 if no cloud).
        outlier_removal_pct: Percentage of points removed as outliers.
        stages_run: Names of pipeline stages that actually executed.
    """

    pathway_name: str
    timing: ProfileReport
    point_count: int = 0
    cloud_density: float = 0.0
    outlier_removal_pct: float = 0.0
    stages_run: list[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Results from a complete benchmark run.

    Attributes:
        results: Per-pathway results in run order.
        config_path: Path string of the PipelineConfig that was benchmarked.
        output_dir: Populated from config.output_dir — used by CLI for report saving.
        frame: Frame index that was benchmarked.
    """

    results: list[PathwayResult] = field(default_factory=list)
    config_path: str = ""
    output_dir: str = ""
    frame: int = 0


def build_pathways(
    base_config: PipelineConfig,
    extractors: list[str] | None = None,
    with_clahe: bool = False,
) -> list[tuple[str, PipelineConfig]]:
    """Build list of (name, config) tuples for each pathway to benchmark.

    Always includes 3 base pathways:

    1. RoMa full    — RoMa dense matcher,     pipeline_mode=full
    2. LG+SP full   — LightGlue + SuperPoint, pipeline_mode=full
    3. LG+SP sparse — LightGlue + SuperPoint, pipeline_mode=sparse

    RoMa sparse is excluded because it pays the same dense-matching cost as
    RoMa full but discards most of the warp field to extract sparse keypoints.

    With ``extractors``: adds LightGlue variants per extractor name.
    With ``with_clahe``: adds CLAHE-on variants for all LightGlue paths.

    Args:
        base_config: Loaded PipelineConfig to use as the template.
        extractors: Optional list of extractor names (e.g., ["aliked", "disk"]).
            Each creates additional LightGlue+sparse and LightGlue+full variants.
        with_clahe: If True, add CLAHE-enabled variants for all LightGlue paths.

    Returns:
        List of (pathway_name, modified_config) tuples.
    """
    pathways: list[tuple[str, PipelineConfig]] = []

    def _make_config(
        matcher: str,
        mode: str,
        extractor: str = "superpoint",
        clahe: bool = False,
    ) -> PipelineConfig:
        cfg = copy.deepcopy(base_config)
        cfg.matcher_type = matcher
        cfg.pipeline_mode = mode
        if matcher == "lightglue":
            cfg.sparse_matching.extractor_type = extractor
            cfg.sparse_matching.clahe_enabled = clahe
        return cfg

    # --- 3 base pathways ---
    # RoMa full runs first: it has the highest peak GPU memory and benefits
    # from a clean CUDA allocator with no prior fragmentation.
    #
    # RoMa sparse is excluded: it pays the same dense-matching cost as RoMa
    # full but discards most of the warp field to extract sparse keypoints.
    # Use LG+SP sparse for fast sparse reconstruction instead.
    pathways.append(("RoMa full", _make_config("roma", "full")))
    pathways.append(("LG+SP full", _make_config("lightglue", "full", "superpoint")))
    pathways.append(("LG+SP sparse", _make_config("lightglue", "sparse", "superpoint")))

    # --- Extra extractor variants ---
    if extractors:
        for ext in extractors:
            if ext.lower() == "superpoint":
                # Already covered by base pathways — skip duplicates
                continue
            pathways.append(
                (f"LG+{ext.upper()} sparse", _make_config("lightglue", "sparse", ext))
            )
            pathways.append(
                (f"LG+{ext.upper()} full", _make_config("lightglue", "full", ext))
            )

    # --- CLAHE variants ---
    if with_clahe:
        # Collect the current LightGlue pathways (before adding more) and add CLAHE versions
        base_count = len(pathways)
        for i in range(base_count):
            name, cfg = pathways[i]
            if cfg.matcher_type == "lightglue":
                clahe_cfg = copy.deepcopy(cfg)
                clahe_cfg.sparse_matching.clahe_enabled = True
                pathways.append((f"{name}+CLAHE", clahe_cfg))

    return pathways


def _read_single_frame(
    config: PipelineConfig,
    frame: int,
) -> "dict[str, object]":
    """Read a single frame from the configured video/image source.

    Args:
        config: PipelineConfig with camera_input_map set.
        frame: Frame index to read.

    Returns:
        Dictionary mapping camera name to raw BGR NumPy array.
    """
    import numpy as np
    from aquacal.io.video import VideoSet

    from ..io import ImageDirectorySet, detect_input_type

    input_type = detect_input_type(config.camera_input_map)
    if input_type == "images":
        context_manager = ImageDirectorySet(config.camera_input_map)
    else:
        context_manager = VideoSet(config.camera_input_map)

    with context_manager as source:
        raw_images: dict[str, np.ndarray] = source.read_frame(frame)

    return raw_images


def run_benchmark(
    config_path: Path,
    frame: int = 0,
    extractors: list[str] | None = None,
    with_clahe: bool = False,
) -> BenchmarkResult:
    """Run benchmark comparison across all pathways.

    For each pathway:

    1. Deep-copy and modify config for the pathway (matcher, mode, extractor, CLAHE).
    2. Set output directory to ``{base_output}/benchmark/{pathway_name}/``.
    3. Create a FRESH ``PipelineProfiler()`` instance for this pathway.
       (Do NOT reuse a single profiler — ``profiler.snapshots`` is a plain dict
       so reusing it overwrites same-named stage keys across pathways.)
    4. Set active profiler via ``set_active_profiler(fresh_profiler)``.
    5. Read frame, build context, run ``process_frame``.
    6. Collect profiler report + relative metrics.
    7. Deactivate profiler via ``set_active_profiler(None)``.

    Args:
        config_path: Path to the pipeline config YAML (same as ``aquamvs run``).
        frame: Frame index to benchmark (default 0).
        extractors: Additional LightGlue extractor variants (e.g., ["aliked", "disk"]).
        with_clahe: Add CLAHE-on variants for LightGlue pathways.

    Returns:
        BenchmarkResult with per-pathway timing, point counts, and cloud density.
    """
    from ..pipeline.builder import build_pipeline_context
    from ..pipeline.runner import process_frame
    from .metrics import compute_relative_metrics

    # Reduce CUDA memory fragmentation across sequential pathway runs.
    # Without this, reserved-but-unallocated memory from earlier pathways
    # can prevent large contiguous allocations in later ones (e.g. RoMa full).
    if torch.cuda.is_available():
        _alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments" not in _alloc_conf:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                f"{_alloc_conf},expandable_segments:True".lstrip(",")
            )

    config_path = Path(config_path)
    base_config = PipelineConfig.from_yaml(config_path)
    base_output_dir = Path(base_config.output_dir)

    # Read frame once (shared across pathways — same raw images for all)
    logger.info("Reading frame %d from %s", frame, config_path)
    raw_images = _read_single_frame(base_config, frame)

    pathways = build_pathways(base_config, extractors=extractors, with_clahe=with_clahe)
    logger.info("Running %d pathway(s)...", len(pathways))

    benchmark_result = BenchmarkResult(
        config_path=str(config_path),
        output_dir=str(base_output_dir),
        frame=frame,
    )

    for pathway_name, pathway_cfg in pathways:
        # Redirect pathway output to a benchmark subdirectory
        safe_name = pathway_name.replace("+", "_").replace(" ", "_")
        pathway_cfg.output_dir = str(base_output_dir / "benchmark" / safe_name)
        pathway_cfg.runtime.viz_enabled = True

        logger.info("  Pathway: %s -> %s", pathway_name, pathway_cfg.output_dir)

        ctx = build_pipeline_context(pathway_cfg)

        # Fresh profiler per pathway — isolated snapshots dict
        profiler = PipelineProfiler()

        with profiler:
            set_active_profiler(profiler)
            try:
                process_frame(frame, raw_images, ctx)
            except Exception:
                logger.exception("  Pathway %s: processing failed", pathway_name)
            finally:
                set_active_profiler(None)

        timing = profiler.get_report()

        # Viz pass (outside profiler so render time doesn't pollute metrics)
        if pathway_cfg.runtime.viz_enabled:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            run_visualization_pass(pathway_cfg, ctx)

        # Collect relative metrics from output
        pathway_output_dir = Path(pathway_cfg.output_dir)
        rel_metrics = compute_relative_metrics(pathway_output_dir)

        pw_result = PathwayResult(
            pathway_name=pathway_name,
            timing=timing,
            point_count=int(rel_metrics["point_count"]),
            cloud_density=rel_metrics["cloud_density"],
            outlier_removal_pct=rel_metrics["outlier_removal_pct"],
            stages_run=list(timing.stages.keys()),
        )
        benchmark_result.results.append(pw_result)
        logger.info(
            "  Pathway %s: %.1f s total, %d points",
            pathway_name,
            timing.total_time_ms / 1000.0,
            pw_result.point_count,
        )

        # Free GPU memory before next pathway loads different models
        del ctx
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return benchmark_result


__all__ = ["run_benchmark", "BenchmarkResult", "PathwayResult", "build_pathways"]
