# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-14)

**Core value:** Accurate refractive multi-view stereo reconstruction — cameras in air, geometry underwater, Snell's law bridging the two.
**Current focus:** Phase 2 complete — ready for Phase 3 planning

## Current Position

Phase: 04 (Documentation and Examples)
Plan: 7 of 7 plans complete
Status: Complete
Last activity: 2026-02-15 - Completed quick task 3: Fix test failures from Pydantic config migration

Progress: [████████░░] 80% (4 of 5 phases complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 15
- Average duration: 5.4 min
- Total execution time: 1.6 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01    | 2     | 20min | 10min    |
| 01.1  | 2     | 17min | 8.5min   |
| 02    | 2     | 16min | 8min     |
| 03    | 3     | 16min | 5.3min   |
| 04    | 7     | 24min | 3.4min   |

**Recent Trend:**
- Last 5 plans: 04-03 (7min), 04-04 (4min), 04-05 (1min), 04-06 (3min), 04-07 (3min)
- Trend: Documentation tasks fast (1-7min), gap closure tasks very fast (3min)

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 03 P01 | 4 | 2 tasks | 6 files |
| Phase 03 P02 | 4 | 2 tasks | 7 files |
| Phase 03 P03 | 8 | 2 tasks | 5 files |
| Phase 04 P01 | 2 | 2 tasks | 10 files |
| Phase 04 P02 | 4 | 2 tasks | 10 files |
| Phase 04 P03 | 7 | 2 tasks | 4 files |
| Phase 04 P04 | 4 | 2 tasks | 3 files |
| Phase 04 P05 | 1 | 2 tasks | 3 files |
| Phase 04 P06 | 3 | 3 tasks | 5 files |
| Phase 04 P07 | 3 | 2 tasks | 1 files |

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

**Phase 04 Plan 03 (2026-02-15):**
- Theory section structured in three stages: refractive geometry, dense stereo, fusion
- Mermaid diagrams for visual representation (requires sphinxcontrib-mermaid extension)
- Comprehensive math derivations with RST math directives for equations
- Cross-references to API docs to connect theory with implementation

**Phase 04 Plan 01 (2026-02-15):**
- Furo theme over sphinx_rtd_theme for modern aesthetics and better mobile support
- ReadTheDocs custom build.commands to install PyTorch CPU and git prerequisites before package install
- Documentation structure: Getting Started, User Guide, Theory, API Reference sections
- Installation guide covers Windows, Linux, macOS, GPU/CPU configurations with troubleshooting
- [Phase 04]: Visual-first README with hero image, badges, and minimal quickstart snippet
- [Phase 04]: Alabaster theme for Sphinx docs (furo not installed)

**Phase 04 Plan 04 (2026-02-15):**
- Dual format tutorials (Jupyter + CLI guide) serve both API users (Python programmers) and CLI users (command-line workflow)
- Downloadable notebook instead of rendered in Sphinx (nbsphinx requires pandoc which complicates CI)
- Placeholder example dataset URL (dataset not yet published to Zenodo/GitHub Releases)

**Phase 04 Plan 05 (2026-02-15):**
- Placeholder hero image generated with matplotlib (conceptual pipeline diagram) until real reconstruction results available
- Example dataset README describes structure but actual data files not committed (added to .gitignore)
- Sphinx build verification confirms all Phase 4 documentation integrates correctly (zero warnings)

**Phase 04 Plan 06 (2026-02-15):**
- GitHub Releases used for example dataset distribution instead of Zenodo (can migrate to Zenodo for DOI later)
- Dataset packaging script automates assembly from AquaCal raw data into distributable archive
- Citation section updated to reference GitHub with note about future Zenodo DOI
- All documentation placeholder URLs replaced with actual GitHub Releases download link
- Dataset structure validation and README generation

**Phase 04 Plan 07 (2026-02-15):**
- Requirements files need bare package specifiers, not 'pip install' command prefix
- ReadTheDocs project connected to GitHub with webhook integration for automatic rebuilds
- Initial build failure fixed via requirements-prereqs.txt format correction
- Deployment pending rebuild after push (90% complete)

### Roadmap Evolution

- Phase 1.1 inserted after Phase 1: Last-Minute Feature Additions (URGENT)

### Pending Todos

1. **Mint a Zenodo DOI** (general) — Link repo to Zenodo, add `.zenodo.json`, get citable DOI
2. **Build portfolio website for Aqua libraries** (general) — Showcase site for AquaCal/AquaMVS/AquaPose targeting employers; static hosting, 3D viewers, comparison sliders
3. **Write project retrospective** (general) — Lessons learned (agentic coding, new tech stacks, workflow pain points, future directions)

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
| 3 | Fix test failures from Pydantic config migration (mock patch paths) | 2026-02-15 | 53ff2f8 | [3-fix-test-failures-from-pydantic-config-m](./quick/3-fix-test-failures-from-pydantic-config-m/) |

## Session Continuity

Last session: 2026-02-15 (quick task execution)
Stopped at: Completed quick-3 (Fix test failures from Pydantic config migration)
Resume file: None
