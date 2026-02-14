# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-14)

**Core value:** Accurate refractive multi-view stereo reconstruction — cameras in air, geometry underwater, Snell's law bridging the two.
**Current focus:** Phase 2 complete — ready for Phase 3 planning

## Current Position

Phase: 03 (Pipeline Decomposition and Modularization)
Plan: 1 of 3 plans complete
Status: In progress — executing phase 03
Last activity: 2026-02-14 - Phase 03 Plan 01 complete

Progress: [██████░░░░] 60% (3 of 5 phases complete, phase 03 1/3)

## Performance Metrics

**Velocity:**
- Total plans completed: 7
- Average duration: 8.6 min
- Total execution time: 1.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01    | 2     | 20min | 10min    |
| 01.1  | 2     | 17min | 8.5min   |
| 02    | 2     | 16min | 8min     |
| 03    | 1     | 4min  | 4min     |

**Recent Trend:**
- Last 5 plans: 01.1-02 (12min), 02-01 (5min), 02-02 (11min), 03-01 (4min)
- Trend: Refactoring/extraction tasks very fast (4min), comprehensive updates moderate (11min)
| Phase 03 P01 | 4 | 2 tasks | 6 files |

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

### Roadmap Evolution

- Phase 1.1 inserted after Phase 1: Last-Minute Feature Additions (URGENT)

### Pending Todos

1. **Mint a Zenodo DOI** (general) — Link repo to Zenodo, add `.zenodo.json`, get citable DOI

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

## Session Continuity

Last session: 2026-02-14 (phase 03 execution)
Stopped at: Completed 03-01-PLAN.md
Resume file: None
