# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-14)

**Core value:** Accurate refractive multi-view stereo reconstruction — cameras in air, geometry underwater, Snell's law bridging the two.
**Current focus:** Phase 2 complete — ready for Phase 3 planning

## Current Position

Phase: 03 (Pipeline Decomposition and Modularization)
Plan: 3 of 3 plans complete
Status: Complete — phase 03 finished
Last activity: 2026-02-15 - Completed quick task 2: Fix pre-commit lint issues

Progress: [████████░░] 80% (4 of 5 phases complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 9
- Average duration: 7.1 min
- Total execution time: 1.1 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01    | 2     | 20min | 10min    |
| 01.1  | 2     | 17min | 8.5min   |
| 02    | 2     | 16min | 8min     |
| 03    | 3     | 16min | 5.3min   |

**Recent Trend:**
- Last 5 plans: 02-02 (11min), 03-01 (4min), 03-02 (4min), 03-03 (8min)
- Trend: Extraction/scaffolding fast (4min), integration moderate (8-11min)
| Phase 03 P01 | 4 | 2 tasks | 6 files |
| Phase 03 P02 | 4 | 2 tasks | 7 files |
| Phase 03 P03 | 8 | 2 tasks | 5 files |
| Phase 03 P03 | 8 | 2 tasks | 5 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- PyTorch over NumPy for internal math (GPU support, differentiability)
- Dual pathway (LightGlue sparse + RoMa dense) for different accuracy/speed tradeoffs
- Package as both CLI + library to serve pipeline users and custom workflow developers

**Phase 01 Plan 01 (2026-02-14):**
- PyTorch as user-managed prerequisite with import-time check (not in pyproject.toml)
- LightGlue pinned to commit edb2b83 (v0.2 release) via git URL
- RoMa v2 pinned to user fork at tlancaster6/RoMaV2 (dataclasses metadata fix)
- AquaCal as standard PyPI dependency (aquacal>=0.1.0)
- prereq-docs strategy: Document LightGlue and RoMa as manual install prerequisites for PyPI compatibility

**Phase 01 Plan 02 (2026-02-14):**
- Matrix testing across Ubuntu and Windows with Python 3.10, 3.11, 3.12 (6 combinations)
- Trusted Publishing (OIDC) eliminates API token management for PyPI uploads
- Three-stage publish pipeline: build -> TestPyPI -> PyPI with manual approval gate

**Phase 01.1 Plan 01 (2026-02-14):**
- Lazy imports in CLI handlers to avoid loading cv2/Open3D at parse time
- Circular buffer for memory-efficient temporal median filtering
- Auto-compute normals for STL export (Open3D requirement)
- Format detection from output_path suffix for mesh export

**Phase 01.1 Plan 02 (2026-02-14):**
- Outlier removal enabled by default (nb_neighbors=20, std_ratio=2.0)
- Consistency map colormap normalized by number of source cameras, not per-frame max
- Viridis colormap for perceptual uniformity and colorblind accessibility
- Auto-detection of input type (image directory vs video file) based on path.is_dir()
- ImageDirectorySet validates matching filenames and frame counts across cameras

**Phase 02 Plan 01 (2026-02-14):**
- Consolidated 14 dataclasses into 6 Pydantic models grouped by pipeline stage
- Automatic error collection with YAML-path formatting (not fail-on-first)
- Extra fields produce warnings, not errors (forward-compatible)
- Backward-compatible YAML migration layer with INFO logging
- Old class names as aliases to prevent import errors

**Phase 02 Plan 02 (2026-02-14):**
- Updated all function signatures to use grouped config classes (SparseMatchingConfig, ReconstructionConfig, RuntimeConfig)
- Migrated all config field access paths in pipeline and downstream modules
- Added tqdm progress bars to frame loop and plane sweep stereo loop
- CLI --quiet flag to suppress progress bars
- Progress bars auto-disable in non-TTY contexts (CI, pipes, logs)

**Phase 03 Plan 01 (2026-02-14):**
- FrameSource protocol abstracts VideoSet and ImageDirectorySet with iterate_frames() method
- CalibrationProvider protocol defined; existing CalibrationData already satisfies it structurally
- ensure_refractive_params() provides refraction-naive fallback (n_air=n_water=1.0) with warning
- build_pipeline_context() replaces setup_pipeline (alias preserved for backward compatibility)
- Pipeline package structure: interfaces.py, context.py, builder.py, helpers.py

**Phase 03 Plan 02 (2026-02-14):**
- 6 stage modules extracted from monolithic process_frame (undistortion, sparse_matching, dense_matching, depth_estimation, fusion, surface)
- Pure function stage design: run_X_stage(inputs, ctx, frame_dir, frame_idx) -> outputs
- Stages are internal-only (not exported from pipeline package)
- Each execution path (lightglue+sparse, lightglue+full, roma+sparse, roma+full) traceable through distinct stage functions
- Visualization and I/O embedded in owning stages, gated by _should_viz and config flags

**Phase 03 Plan 03 (2026-02-15):**
- Pipeline class is primary programmatic API: Pipeline(config).run()
- process_frame orchestrates all 4 execution paths via stage function composition
- AquaCal VideoSet isolated to runner.py only (REF-03 requirement satisfied)
- Top-level import works: from aquamvs import Pipeline
- Test imports and patches updated to new module locations (builder, stages, runner, helpers)

### Roadmap Evolution

- Phase 1.1 inserted after Phase 1: Last-Minute Feature Additions (URGENT)

### Pending Todos

1. **Mint a Zenodo DOI** (general) — Link repo to Zenodo, add `.zenodo.json`, get citable DOI
2. **Build portfolio website for Aqua libraries** (general) — Showcase site for AquaCal/AquaMVS/AquaPose targeting employers; static hosting, 3D viewers, comparison sliders

### Blockers/Concerns

**Phase 1 Dependencies (RESOLVED in 01-01):**
- ✓ LightGlue: git dependency pin to edb2b83, documented as prerequisite
- ✓ RoMa v2: git dependency pin to user fork, documented as prerequisite
- ✓ AquaCal: standard PyPI dependency (aquacal>=0.1.0)
- ✓ PyPI compatibility: prereq-docs strategy allows PyPI upload
approved
**Phase 3 Refactoring:**
- Backward compatibility scope needs definition — audit which APIs are public vs internal to minimize breakage

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 1 | Add slow-test workflow, adopt Ruff+pre-commit, add coverage reporting, add docs workflow with Sphinx scaffolding | 2026-02-14 | c69270d | [1-add-slow-test-workflow-adopt-ruff-pre-co](./quick/1-add-slow-test-workflow-adopt-ruff-pre-co/) |
| 2 | Fix pre-commit and CI test failures (lint, imports, Open3D headless) | 2026-02-15 | 8fabc94 | [2-fix-pre-commit-lint-issues-noqa-directiv](./quick/2-fix-pre-commit-lint-issues-noqa-directiv/) |

## Session Continuity

Last session: 2026-02-15 (phase 03 execution)
Stopped at: Completed 03-03-PLAN.md (Phase 03 complete)
Resume file: None
