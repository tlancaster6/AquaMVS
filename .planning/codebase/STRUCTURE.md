# Codebase Structure

**Analysis Date:** 2026-02-14

## Directory Layout

```
AquaMVS/
├── src/aquamvs/                    # Main package
│   ├── __init__.py                 # Public API exports
│   ├── cli.py                      # Command-line interface
│   ├── config.py                   # Configuration dataclasses
│   ├── pipeline.py                 # Pipeline orchestration
│   ├── calibration.py              # AquaCal integration and undistortion
│   ├── triangulation.py            # Sparse 3D reconstruction
│   ├── coloring.py                 # Best-view color assignment
│   ├── masks.py                    # ROI mask loading and application
│   ├── fusion.py                   # Depth map fusion
│   ├── surface.py                  # Mesh reconstruction
│   ├── projection/                 # Geometric models
│   │   ├── __init__.py
│   │   ├── protocol.py             # ProjectionModel protocol definition
│   │   └── refractive.py           # RefractiveProjectionModel implementation
│   ├── features/                   # Feature extraction and matching
│   │   ├── __init__.py
│   │   ├── extraction.py           # SuperPoint/ALIKED/DISK feature extractors
│   │   ├── matching.py             # LightGlue matcher
│   │   ├── pairs.py                # Camera pair selection strategy
│   │   └── roma.py                 # RoMa v2 dense matcher integration
│   ├── dense/                      # Dense stereo reconstruction
│   │   ├── __init__.py
│   │   ├── plane_sweep.py          # Plane-sweep stereo algorithm
│   │   ├── cost.py                 # Cost volume computation (NCC, SSIM)
│   │   └── roma_depth.py           # RoMa warp-to-depth conversion
│   ├── visualization/              # Diagnostic output rendering
│   │   ├── __init__.py
│   │   ├── depth.py                # Depth map colormaps
│   │   ├── features.py             # Feature keypoint overlays
│   │   ├── scene.py                # 3D point cloud and mesh rendering
│   │   ├── rig.py                  # Camera rig diagram
│   │   └── summary.py              # Time-series gallery and eval plots
│   ├── evaluation/                 # Benchmark metrics and alignment
│   │   ├── __init__.py
│   │   ├── alignment.py            # ICP registration
│   │   └── metrics.py              # Evaluation metrics (cloud distance, etc.)
│   └── benchmark/                  # Comparative benchmarking suite
│       ├── __init__.py
│       ├── runner.py               # Parameter sweep orchestration
│       ├── metrics.py              # Per-configuration metrics aggregation
│       ├── report.py               # HTML report generation
│       └── visualization.py        # Benchmark result plots
├── tests/                          # Test suite
│   ├── conftest.py                 # Shared fixtures (device parametrization)
│   ├── test_calibration.py         # Undistortion, calibration loading
│   ├── test_config.py              # Configuration validation
│   ├── test_cli.py                 # CLI commands
│   ├── test_coloring.py            # Color assignment
│   ├── test_masks.py               # Mask operations
│   ├── test_integration.py         # End-to-end pipeline smoke tests
│   ├── test_roma_import.py         # RoMa availability check
│   ├── test_projection/            # Projection model tests
│   │   ├── test_protocol.py        # Protocol runtime checking
│   │   ├── test_refractive.py      # RefractiveProjectionModel unit tests
│   │   └── test_cross_validation.py # Cross-validation vs AquaCal NumPy
│   ├── test_features/              # Feature extraction/matching tests
│   │   ├── test_extraction.py
│   │   ├── test_matching.py
│   │   ├── test_pairs.py
│   │   └── test_roma.py            # RoMa matcher tests
│   ├── test_dense/                 # Dense stereo tests
│   │   ├── test_cost.py
│   │   ├── test_plane_sweep.py
│   │   ├── test_depth_extraction.py
│   │   └── test_roma_depth.py
│   ├── test_fusion/                # Fusion tests
│   │   ├── test_fusion.py
│   │   └── test_consistency.py     # Geometric filtering tests
│   ├── test_triangulation.py       # Sparse cloud generation tests
│   ├── test_surface.py             # Mesh reconstruction tests
│   ├── test_pipeline.py            # Pipeline orchestration tests
│   ├── test_benchmark/             # Benchmark suite tests
│   │   ├── test_runner.py
│   │   ├── test_metrics.py
│   │   ├── test_report.py
│   │   └── test_visualization.py
│   ├── test_evaluation/            # Evaluation metrics tests
│   │   ├── test_alignment.py
│   │   └── test_metrics.py
│   └── test_visualization/         # Visualization tests
│       ├── test_depth_viz.py
│       ├── test_features_viz.py
│       ├── test_scene_viz.py
│       ├── test_rig_viz.py
│       └── test_summary_viz.py
├── .claude/                        # Agent infrastructure (ignored)
├── .planning/                      # Planning docs (ignored)
├── dev/                            # Development docs (ignored)
├── pyproject.toml                  # Package metadata and dependencies
├── README.md                       # Project overview
├── LICENSE                         # MIT license
└── CLAUDE.md                       # Project instructions for Claude
```

## Directory Purposes

**`src/aquamvs/`:**
- Purpose: Main package implementation
- Contains: All pipeline modules and subpackages
- Key files: `__init__.py` (public API), `pipeline.py` (orchestration), `config.py` (configuration)

**`src/aquamvs/projection/`:**
- Purpose: Geometric projection models (pinhole, refractive, etc.)
- Contains: ProjectionModel protocol, RefractiveProjectionModel implementing Snell's law
- Key files: `protocol.py` (interface definition), `refractive.py` (implementation)

**`src/aquamvs/features/`:**
- Purpose: Sparse feature-based matching pipeline
- Contains: Feature extraction (SuperPoint, ALIKED, DISK), LightGlue matching, RoMa v2 integration, pair selection
- Key files: `extraction.py`, `matching.py`, `roma.py`, `pairs.py`

**`src/aquamvs/dense/`:**
- Purpose: Dense depth map computation
- Contains: Plane-sweep stereo, cost volume construction, RoMa warp aggregation
- Key files: `plane_sweep.py` (main algorithm), `cost.py` (metrics), `roma_depth.py` (warp conversion)

**`src/aquamvs/visualization/`:**
- Purpose: Diagnostic image and 3D rendering
- Contains: Depth colormaps, feature overlays, 3D point cloud/mesh visualization, camera rig diagrams
- Key files: `depth.py`, `features.py`, `scene.py`, `rig.py`

**`src/aquamvs/evaluation/`:**
- Purpose: Benchmark metrics and ground-truth alignment
- Contains: ICP registration, cloud-to-cloud distance, height-field difference, reprojection error
- Key files: `alignment.py`, `metrics.py`

**`src/aquamvs/benchmark/`:**
- Purpose: Comparative benchmark suite for feature/matching configurations
- Contains: Parameter sweep, result aggregation, HTML reporting, benchmark visualizations
- Key files: `runner.py` (orchestration), `report.py` (report generation)

**`tests/`:**
- Purpose: Test coverage for all modules
- Contains: Unit tests (per module), integration tests (end-to-end), cross-validation tests (vs AquaCal)
- Structure: Mirrors src/ package structure with test_ prefix

## Key File Locations

**Entry Points:**
- `src/aquamvs/cli.py`: CLI commands (init, run, export-refs, benchmark)
- `src/aquamvs/pipeline.py`: Pipeline orchestration (setup_pipeline, process_frame, run_pipeline)

**Configuration:**
- `src/aquamvs/config.py`: All configuration dataclasses (PipelineConfig, FrameSamplingConfig, etc.)
- `pyproject.toml`: Package metadata, dependencies, CLI script definition

**Core Logic:**
- `src/aquamvs/projection/refractive.py`: Snell's law projection model (torch implementation)
- `src/aquamvs/triangulation.py`: Ray triangulation (least-squares solver)
- `src/aquamvs/dense/plane_sweep.py`: Plane-sweep cost volume construction
- `src/aquamvs/fusion.py`: Depth map fusion (geometric filtering + backprojection)
- `src/aquamvs/surface.py`: Surface reconstruction (Poisson, BPA, heightfield)

**Testing:**
- `tests/conftest.py`: Shared fixtures (device parametrization)
- `tests/test_projection/test_cross_validation.py`: Cross-validation against AquaCal NumPy implementation

## Naming Conventions

**Files:**
- Modules: `snake_case.py` (e.g., `plane_sweep.py`, `roma_depth.py`)
- Packages: Directory names lowercase, `__init__.py` provides public API

**Classes:**
- Format: `PascalCase` (e.g., `RefractiveProjectionModel`, `PipelineConfig`, `CalibrationData`)

**Functions:**
- Format: `snake_case` (e.g., `plane_sweep_stereo`, `extract_features`, `fuse_depth_maps`)

**Variables:**
- Format: `snake_case` (e.g., `depth_maps`, `projection_models`, `confidence_map`)
- Suffixes: `_ref` (reference camera), `_src` (source camera), `_map` (2D arrays), `_cloud` (3D point sets)

**Constants:**
- Format: `UPPER_SNAKE_CASE` (e.g., `VIDEO_EXTENSIONS`, `VALID_VIZ_STAGES`)

## Where to Add New Code

**New Feature Matcher:**
- Implementation: `src/aquamvs/features/` (new module or extend `matching.py`)
- Tests: `tests/test_features/test_{matcher_name}.py`
- Integration: Update `features/__init__.py` exports, add config option to `FeatureExtractionConfig` or `MatchingConfig`

**New Dense Matching Method:**
- Implementation: `src/aquamvs/dense/` (new module, e.g., `cvpr_method.py`)
- Tests: `tests/test_dense/test_{method_name}.py`
- Integration: Add option to `DenseMatchingConfig`, update `pipeline.py` dispatcher

**New Projection Model:**
- Implementation: `src/aquamvs/projection/{model_name}.py`
- Protocol: Must implement `ProjectionModel` protocol from `protocol.py`
- Tests: `tests/test_projection/test_{model_name}.py`
- Integration: Add instantiation to `setup_pipeline()` based on calibration type

**New Surface Reconstruction Method:**
- Implementation: Function in `src/aquamvs/surface.py` following pattern `reconstruct_{method_name}()`
- Tests: `tests/test_surface.py` (add test function)
- Integration: Add condition to `surface.py` dispatcher, add config option to `SurfaceConfig`

**New Visualization:**
- Implementation: New module in `src/aquamvs/visualization/{viz_name}.py`
- Public function: `render_{viz_name}()` returning None (outputs written to disk)
- Tests: `tests/test_visualization/test_{viz_name}.py`
- Integration: Add to `_should_viz()` check in `pipeline.py` if per-frame, or conditional calling

**Utilities:**
- Shared helpers for cross-cutting logic: Extend existing module or add `src/aquamvs/utils/{topic}.py`
- Example: Color normalization helpers in `coloring.py`

**Evaluation Metric:**
- Implementation: Function in `src/aquamvs/evaluation/metrics.py`
- Tests: `tests/test_evaluation/test_metrics.py`
- Integration: Add to `EvaluationConfig` if configurable, update benchmark runner

## Special Directories

**`.planning/codebase/`:**
- Purpose: Codebase analysis documents (ARCHITECTURE.md, STRUCTURE.md, etc.)
- Generated: Yes (by /gsd:map-codebase)
- Committed: Yes (git tracking)

**`dev/`:**
- Purpose: Development workflow docs (DESIGN.md, TASKS.md, KNOWLEDGE_BASE.md, CHANGELOG.md, tasks/, handoffs/)
- Generated: Yes (by agents during development)
- Committed: No (gitignored)

**`.claude/`:**
- Purpose: Agent infrastructure and hooks
- Generated: No (checked in)
- Committed: Yes

**`.planning/`:**
- Purpose: GSD planning output directory (created by /gsd orchestrator)
- Generated: Yes
- Committed: No (gitignored)

## Module-Level Public API

Each package must export its public interface in `__init__.py`. Example from `src/aquamvs/__init__.py`:

```python
from .projection import ProjectionModel, RefractiveProjectionModel
from .features import extract_features, match_all_pairs
from .dense import plane_sweep_stereo, extract_depth
from .fusion import fuse_depth_maps, filter_depth_map
from .surface import reconstruct_surface, reconstruct_poisson
from .pipeline import PipelineContext, setup_pipeline, process_frame, run_pipeline

__all__ = [
    "ProjectionModel",
    "RefractiveProjectionModel",
    "extract_features",
    "match_all_pairs",
    "plane_sweep_stereo",
    "extract_depth",
    "fuse_depth_maps",
    "filter_depth_map",
    "reconstruct_surface",
    "reconstruct_poisson",
    "PipelineContext",
    "setup_pipeline",
    "process_frame",
    "run_pipeline",
]
```

Always add new public symbols to `__init__.py` and `__all__`. Never import private/internal functions directly into packages.

## Output Directory Structure

Pipeline creates frame-specific subdirectories under `output_dir`:

```
output_dir/
├── config.yaml                     # Copy of PipelineConfig (saved at setup)
├── frame_000000/                   # Per-frame outputs
│   ├── sparse/
│   │   └── sparse_cloud.pt         # Unfiltered triangulated points (optional)
│   ├── point_cloud/
│   │   ├── sparse.ply              # Filtered sparse cloud (sparse mode)
│   │   └── fused.ply               # Fused dense cloud (dense mode)
│   ├── mesh/
│   │   └── surface.ply             # Reconstructed mesh
│   ├── depth_maps/                 # (optional, removed if keep_intermediates=False)
│   │   ├── {camera_name}.npz       # (depth_map, confidence_map) tuple
│   │   └── ...
│   └── viz/                        # (optional, if visualization enabled)
│       ├── depth_{camera}.png      # Depth colormaps
│       ├── features_{camera}.png   # Feature keypoints
│       ├── matches_{ref}_{src}.png # Match overlays
│       ├── scene_*.png             # 3D renders from viewpoints
│       └── rig.png                 # Camera rig diagram
├── frame_000100/                   # Next frame
│   └── ...
└── summary/                        # (optional)
    ├── timeseries_gallery.png      # Height map sequence
    └── evaluation_summary.html     # Benchmark report
```

---

*Structure analysis: 2026-02-14*
