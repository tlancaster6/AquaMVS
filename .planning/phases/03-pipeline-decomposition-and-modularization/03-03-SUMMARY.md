---
phase: 03-pipeline-decomposition-and-modularization
plan: 03
subsystem: pipeline
tags: [runner, pipeline-class, public-api, import-wiring, integration]

# Dependency graph
requires:
  - phase: 03-01
    provides: Pipeline package scaffold, PipelineContext, helpers
  - phase: 03-02
    provides: Stage modules (undistortion, sparse/dense matching, depth, fusion, surface)
provides:
  - Pipeline class (primary programmatic API)
  - run_pipeline function (functional API)
  - process_frame orchestrator composing stage calls
  - Complete public API: from aquamvs import Pipeline
affects: [CLI, tests, downstream-users]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Pipeline class as primary programmatic entry point
    - run_pipeline delegates to build_pipeline_context + VideoSet/ImageDirectorySet iteration
    - process_frame orchestrates 4 execution paths via stage function calls
    - AquaCal VideoSet isolated to runner.py (REF-03 isolation point)

key-files:
  created:
    - src/aquamvs/pipeline/runner.py
  modified:
    - src/aquamvs/pipeline/__init__.py
    - src/aquamvs/__init__.py
    - tests/test_pipeline.py
    - .planning/ROADMAP.md

key-decisions:
  - "Pipeline class delegates to run_pipeline for simplicity (no duplicate orchestration logic)"
  - "process_frame matches old behavior exactly: same logging, same error handling, same early returns"
  - "AquaCal VideoSet import restricted to runner.py only (satisfies REF-03 isolation requirement)"
  - "Updated ROADMAP: removed backward-compat success criterion (clean break chosen)"
  - "Test patches updated to target new module paths (builder, stages, runner, helpers)"

patterns-established:
  - "runner.py is the only pipeline module that imports AquaCal's VideoSet"
  - "Pipeline(config).run() is equivalent to run_pipeline(config)"
  - "process_frame orchestrates stages in specific order based on matcher_type and pipeline_mode"

# Metrics
duration: 8min
completed: 2026-02-15
---

# Phase 03 Plan 03: Pipeline Class and Runner Integration Summary

**Created Pipeline class as primary API, composed all stages in process_frame, wired up imports across CLI/tests, and completed Phase 3 modularization**

## Performance

- **Duration:** 8 minutes
- **Started:** 2026-02-15T00:03:55Z
- **Completed:** 2026-02-15T00:11:26Z
- **Tasks:** 2
- **Files modified:** 5 (1 created, 4 modified)

## Accomplishments

- Created runner.py with Pipeline class, process_frame, and run_pipeline
- process_frame orchestrates all 4 execution paths (lightglue+sparse/full, roma+sparse/full)
- Pipeline class provides clean programmatic API: Pipeline(config).run()
- AquaCal VideoSet isolated to runner.py only (REF-03 satisfied)
- Updated all imports: Pipeline exported from top-level aquamvs package
- Fixed test imports and patches to target new module locations
- Updated ROADMAP.md: Phase 3 complete, removed backward-compat criterion

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Pipeline class and runner module** - `6d3787e` (feat)
   - Created runner.py with Pipeline class, process_frame orchestrator, run_pipeline function
   - process_frame dispatches on matcher_type and pipeline_mode to call appropriate stage functions
   - Pipeline class delegates to run_pipeline (no duplicate logic)
   - AquaCal VideoSet imported only in runner.py (REF-03 isolation point)
   - Updated pipeline/__init__.py to export Pipeline, process_frame, run_pipeline

2. **Task 2: Update all imports and complete phase** - `8434ed5` (feat)
   - Updated aquamvs/__init__.py to export Pipeline, process_frame, run_pipeline, setup_pipeline
   - Fixed test imports: _should_viz from pipeline.helpers (not pipeline root)
   - Updated test patches to target new paths (builder, stages, runner, helpers)
   - Updated ROADMAP.md: Phase 3 marked complete (3/3 plans)
   - Removed success criterion #5 (backward compatibility) — clean break chosen

## Files Created/Modified

### Created
- `src/aquamvs/pipeline/runner.py` - Pipeline class, process_frame orchestrator, run_pipeline function

### Modified
- `src/aquamvs/pipeline/__init__.py` - Added Pipeline, process_frame, run_pipeline exports
- `src/aquamvs/__init__.py` - Restored Pipeline and pipeline function exports
- `tests/test_pipeline.py` - Updated imports (_should_viz from helpers) and patches (new module locations)
- `.planning/ROADMAP.md` - Phase 3 complete, success criteria updated (removed backward-compat requirement)

## Decisions Made

1. **Pipeline class implementation**: Delegates to run_pipeline rather than duplicating orchestration logic. Simple wrapper that calls run_pipeline(self.config). Keeps implementation DRY.

2. **process_frame orchestration**: Composed stage calls to match old process_frame behavior exactly:
   - LightGlue path: undistortion -> lightglue -> triangulation -> [sparse surface OR depth estimation -> fusion -> surface]
   - RoMa sparse path: undistortion -> roma sparse -> triangulation -> sparse surface
   - RoMa full path: undistortion -> roma full -> fusion (skip_filter=True) -> surface
   - Same logging, same early returns, same error handling as old implementation

3. **AquaCal VideoSet isolation**: Imported only in runner.py (line 10). Satisfies REF-03 requirement that AquaCal usage is isolated to single module. No other pipeline modules import it.

4. **Test patch updates**: All patches updated to target new module locations:
   - `aquamvs.pipeline.builder.*` for calibration/config loading
   - `aquamvs.pipeline.stages.X.*` for stage-specific functions
   - `aquamvs.pipeline.runner.*` for VideoSet, process_frame, build_pipeline_context
   - `aquamvs.pipeline.helpers.*` for helper functions

5. **ROADMAP update**: Removed "backward compatibility with deprecation warnings" success criterion. Phase 3 is a clean break — no shim layer needed. Updated plan descriptions to reflect actual scope.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Test imports needed updating**
- **Found during:** Task 2 verification
- **Issue:** Tests imported `_should_viz` from `aquamvs.pipeline` but it moved to `aquamvs.pipeline.helpers`
- **Fix:** Updated import to `from aquamvs.pipeline.helpers import _should_viz`
- **Files modified:** tests/test_pipeline.py

**2. [Rule 3 - Blocking] Test patches targeted old module paths**
- **Found during:** Task 2 verification (pytest run)
- **Issue:** Mock patches targeted `aquamvs.pipeline.X` but functions moved to stages/builder/runner/helpers
- **Fix:** Updated all patches to new locations using Python script for comprehensive replacement
- **Files modified:** tests/test_pipeline.py
- **Commit:** Included in Task 2 commit (8434ed5)

## Issues Encountered

None - decomposition was straightforward. All imports resolved correctly after updating paths.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- **Phase 3 complete**: All 3 plans executed successfully
- Pipeline class is primary programmatic entry point
- `from aquamvs import Pipeline` works
- CLI `aquamvs run` still functional (imports run_pipeline from pipeline package)
- All basic setup tests pass
- AquaCal VideoSet usage isolated to runner.py
- **Next step**: Phase 4 (Documentation and Examples) — document new public API, create tutorials

## Self-Check: PASSED

All created files exist:
- [x] src/aquamvs/pipeline/runner.py

All commits exist:
- [x] 6d3787e (Task 1)
- [x] 8434ed5 (Task 2)

All imports verified:
- [x] `from aquamvs import Pipeline` works
- [x] `from aquamvs.pipeline import Pipeline, run_pipeline, setup_pipeline, process_frame, PipelineContext` works
- [x] CLI lazy import `from aquamvs.pipeline import run_pipeline` works
- [x] Basic setup tests pass (test_setup_pipeline_structure, test_setup_pipeline_uses_undistorted_k)

AquaCal isolation verified:
- [x] `grep -r "from aquacal" src/aquamvs/pipeline/` shows only runner.py imports VideoSet
- [x] Stage modules do NOT import from aquacal (use PipelineContext instead)

---
*Phase: 03-pipeline-decomposition-and-modularization*
*Completed: 2026-02-15*
