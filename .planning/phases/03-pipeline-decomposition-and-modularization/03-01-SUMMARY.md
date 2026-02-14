---
phase: 03-pipeline-decomposition-and-modularization
plan: 01
subsystem: pipeline
tags: [protocols, abstraction, refactoring, modularization]

# Dependency graph
requires:
  - phase: 02-config-and-api-cleanup
    provides: Pydantic config models, grouped config classes
provides:
  - Pipeline package scaffold with Protocol-based abstraction
  - FrameSource and CalibrationProvider protocols
  - PipelineContext dataclass in context.py
  - build_pipeline_context function (setup_pipeline) in builder.py
  - Helper functions (_should_viz, _save_consistency_map, etc.)
affects: [03-02, 03-03, pipeline-stages, pipeline-runner]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Protocol-based abstraction for input sources (FrameSource)
    - Protocol-based abstraction for calibration providers (CalibrationProvider)
    - Refraction-naive fallback (n_air=n_water=1.0) for calibrations without refractive data
    - Package organization: interfaces, context, builder, helpers

key-files:
  created:
    - src/aquamvs/pipeline/__init__.py
    - src/aquamvs/pipeline/interfaces.py
    - src/aquamvs/pipeline/context.py
    - src/aquamvs/pipeline/builder.py
    - src/aquamvs/pipeline/helpers.py
  modified:
    - src/aquamvs/__init__.py

key-decisions:
  - "FrameSource protocol abstracts VideoSet and ImageDirectorySet with iterate_frames() method"
  - "CalibrationProvider protocol defined; existing CalibrationData already satisfies it structurally"
  - "ensure_refractive_params() provides refraction-naive fallback (n_air=n_water=1.0) with warning for calibrations missing refractive data"
  - "build_pipeline_context() replaces setup_pipeline (alias preserved for backward compatibility)"
  - "Helper functions extracted to helpers.py (_should_viz, _save_consistency_map, _collect_height_maps, _sparse_cloud_to_open3d)"

patterns-established:
  - "Pipeline package structure: interfaces.py for protocols, context.py for shared data, builder.py for initialization, helpers.py for utilities"
  - "Protocol definitions follow projection/protocol.py pattern: @runtime_checkable decorator, type hints, docstrings"
  - "Backward compatibility via aliases: setup_pipeline = build_pipeline_context"

# Metrics
duration: 4min
completed: 2026-02-14
---

# Phase 03 Plan 01: Pipeline Decomposition Scaffold Summary

**Pipeline package created with FrameSource and CalibrationProvider protocols, PipelineContext dataclass, build_pipeline_context function, and helper utilities extracted from monolithic pipeline.py**

## Performance

- **Duration:** 4 minutes
- **Started:** 2026-02-14T23:49:08Z
- **Completed:** 2026-02-14T23:53:25Z
- **Tasks:** 2
- **Files modified:** 6 (5 created, 1 modified)

## Accomplishments

- Created pipeline/ package replacing monolithic pipeline.py (1125 lines → organized modules)
- Defined FrameSource protocol abstracting VideoSet and ImageDirectorySet
- Defined CalibrationProvider protocol with refraction-naive fallback (ensure_refractive_params)
- Extracted PipelineContext dataclass to context.py
- Extracted build_pipeline_context (formerly setup_pipeline) to builder.py
- Extracted 4 helper functions to helpers.py (_should_viz, _save_consistency_map, _collect_height_maps, _sparse_cloud_to_open3d)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Protocol interfaces and pipeline package scaffold** - `09c0406` (feat)
   - Created interfaces.py with FrameSource and CalibrationProvider protocols
   - Created context.py with PipelineContext dataclass
   - Created helpers.py with 4 helper functions
   - Created package __init__.py (temporary, just PipelineContext export)
   - Deleted old monolithic pipeline.py
   - Updated main __init__.py to temporarily comment out missing functions

2. **Task 2: Create builder module (setup_pipeline extraction)** - `4e6d9f4` (feat)
   - Created builder.py with build_pipeline_context function
   - Added setup_pipeline as backward-compatibility alias
   - Updated pipeline __init__.py to re-export builder functions
   - Re-enabled setup_pipeline import in main __init__.py

## Files Created/Modified

### Created
- `src/aquamvs/pipeline/__init__.py` - Package init with PipelineContext, build_pipeline_context, setup_pipeline exports
- `src/aquamvs/pipeline/interfaces.py` - FrameSource and CalibrationProvider Protocol definitions, ensure_refractive_params helper
- `src/aquamvs/pipeline/context.py` - PipelineContext dataclass
- `src/aquamvs/pipeline/builder.py` - build_pipeline_context function (former setup_pipeline)
- `src/aquamvs/pipeline/helpers.py` - Helper functions (_should_viz, _save_consistency_map, _collect_height_maps, _sparse_cloud_to_open3d)

### Modified
- `src/aquamvs/__init__.py` - Temporarily commented out process_frame and run_pipeline (not yet created), re-enabled setup_pipeline after Task 2

## Decisions Made

1. **FrameSource protocol design**: Uses `iterate_frames(start, stop, step)` returning `Iterator[tuple[int, dict[str, np.ndarray]]]` to abstract both VideoSet and ImageDirectorySet. Both classes already satisfy this protocol structurally — no changes needed to io module.

2. **CalibrationProvider protocol**: Matches CalibrationData's existing interface (cameras, water_z, n_water, n_air, interface_normal, ring_cameras, auxiliary_cameras, camera_positions). CalibrationData already satisfies this structurally — no modifications needed.

3. **Refraction-naive fallback**: `ensure_refractive_params()` checks for missing/trivial refractive parameters (water_z=0, n_water=1.0, n_air=1.0). If detected, logs descriptive warning and returns wrapper that forces n_air=n_water=1.0. This allows pipeline to operate on air-only calibrations or pinhole-only setups without errors, though results will be incorrect for underwater geometry.

4. **setup_pipeline renamed**: Function renamed to `build_pipeline_context()` for clarity (more descriptive name). Original `setup_pipeline` preserved as alias for backward compatibility during transition.

5. **Helper extraction**: Extracted 4 helper functions to helpers.py to reduce clutter in future pipeline runner module. These are internal utilities used by stages/runner.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - extraction was straightforward. All imports resolved correctly after package restructuring.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Pipeline package scaffold complete
- Interfaces and context ready for stage modules (Plan 02)
- Builder function ready for runner integration (Plan 03)
- Helper functions available for stage implementations
- **Next step**: Create stage modules (matching, triangulation, stereo, fusion, surface) in Plan 02

## Self-Check: PASSED

All created files exist:
- [x] src/aquamvs/pipeline/__init__.py
- [x] src/aquamvs/pipeline/interfaces.py
- [x] src/aquamvs/pipeline/context.py
- [x] src/aquamvs/pipeline/builder.py
- [x] src/aquamvs/pipeline/helpers.py

All commits exist:
- [x] 09c0406 (Task 1)
- [x] 4e6d9f4 (Task 2)

All imports verified:
- [x] `from aquamvs.pipeline.interfaces import FrameSource, CalibrationProvider`
- [x] `from aquamvs.pipeline.context import PipelineContext`
- [x] `from aquamvs.pipeline.helpers import _should_viz`
- [x] `from aquamvs.pipeline import PipelineContext, setup_pipeline`
- [x] `from aquamvs import setup_pipeline, PipelineContext`

---
*Phase: 03-pipeline-decomposition-and-modularization*
*Completed: 2026-02-14*
