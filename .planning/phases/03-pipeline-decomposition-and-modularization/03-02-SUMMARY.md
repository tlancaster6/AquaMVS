---
phase: 03-pipeline-decomposition-and-modularization
plan: 02
subsystem: pipeline
tags: [stage-extraction, modularization, refactoring, execution-paths]

# Dependency graph
requires:
  - phase: 03-01
    provides: Pipeline package scaffold, PipelineContext, helpers
provides:
  - 6 stage modules under pipeline/stages/ covering all execution paths
  - Pure function stage design with (inputs, context) -> outputs pattern
  - Embedded visualization and I/O within stage modules
affects: [03-03, pipeline-runner]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Pure function stage design (no state, explicit inputs/outputs)
    - Stages delegate to domain modules (features/, dense/, fusion/, triangulation/, surface/)
    - Visualization and I/O embedded in owning stages, gated by config
    - Internal-only stage package (not exported from pipeline)

key-files:
  created:
    - src/aquamvs/pipeline/stages/__init__.py
    - src/aquamvs/pipeline/stages/undistortion.py
    - src/aquamvs/pipeline/stages/sparse_matching.py
    - src/aquamvs/pipeline/stages/dense_matching.py
    - src/aquamvs/pipeline/stages/depth_estimation.py
    - src/aquamvs/pipeline/stages/fusion.py
    - src/aquamvs/pipeline/stages/surface.py
  modified: []

key-decisions:
  - "Stage modules are internal-only (not exported from pipeline package)"
  - "Each execution path (lightglue+sparse, lightglue+full, roma+sparse, roma+full) traceable through distinct stage functions"
  - "Stage functions receive PipelineContext and return explicit outputs (no side effects except logging/I/O)"
  - "Visualization and I/O operations embedded in owning stages, gated by _should_viz and config flags"
  - "Stages use absolute imports for domain modules, relative imports for pipeline siblings"

patterns-established:
  - "Stage function signature: run_X_stage(inputs, ctx, frame_dir, frame_idx) -> outputs"
  - "No stage imports from other stages (avoid coupling)"
  - "Helper functions (_should_viz, _save_consistency_map, _sparse_cloud_to_open3d) shared via ..helpers"

# Metrics
duration: 4min
completed: 2026-02-14
---

# Phase 03 Plan 02: Pipeline Stage Module Extraction Summary

**Extracted all process_frame logic into 6 distinct stage modules (undistortion, sparse_matching, dense_matching, depth_estimation, fusion, surface) under pipeline/stages/ with pure function design and embedded visualization/I/O**

## Performance

- **Duration:** 4 minutes
- **Started:** 2026-02-14T23:56:04Z
- **Completed:** 2026-02-14T00:00:30Z
- **Tasks:** 2
- **Files modified:** 7 (7 created, 0 modified)

## Accomplishments

- Created pipeline/stages/ package with 7 files (6 stage modules + __init__.py)
- Extracted undistortion + color normalization stage (run_undistortion_stage)
- Extracted LightGlue sparse matching stage (run_lightglue_path, run_triangulation)
- Extracted RoMa dense matching stage (run_roma_full_path, run_roma_sparse_path)
- Extracted depth estimation stage (run_depth_estimation with plane sweep stereo)
- Extracted fusion stage (run_fusion_stage with consistency filtering + fusion + outlier removal)
- Extracted surface stage (run_surface_stage, run_sparse_surface_stage for both full and sparse modes)
- All 4 execution paths now traceable through distinct stage function calls

## Task Commits

Each task was committed atomically:

1. **Task 1: Create undistortion, sparse matching, and dense matching stage modules** - `61e7546` (feat)
   - stages/__init__.py: Internal package marker with docstring, no exports
   - stages/undistortion.py: Undistort images, apply color normalization if enabled, return undistorted numpy + tensors + camera_centers
   - stages/sparse_matching.py: LightGlue feature extraction + masking + matching + feature viz + save features/matches + triangulation + sparse cloud filtering + depth ranges
   - stages/dense_matching.py: RoMa full path (run_roma_all_pairs -> roma_warps_to_depth_maps) and sparse path (match_all_pairs_roma)

2. **Task 2: Create depth estimation, fusion, and surface stage modules** - `cf96d8b` (feat)
   - stages/depth_estimation.py: Plane sweep stereo for all ring cameras with tqdm progress + depth extraction + masking + save + depth viz
   - stages/fusion.py: Geometric consistency filtering (skip if skip_filter=True for roma+full) + depth map fusion + intermediate cleanup + save point cloud + outlier removal
   - stages/surface.py: Surface reconstruction + mesh coloring (ring cameras only) + save mesh + scene viz + rig viz for both full mode (run_surface_stage) and sparse mode (run_sparse_surface_stage)

## Files Created/Modified

### Created
- `src/aquamvs/pipeline/stages/__init__.py` - Internal package marker (no exports)
- `src/aquamvs/pipeline/stages/undistortion.py` - Undistortion + color normalization stage
- `src/aquamvs/pipeline/stages/sparse_matching.py` - LightGlue extraction + matching + triangulation
- `src/aquamvs/pipeline/stages/dense_matching.py` - RoMa full and sparse paths
- `src/aquamvs/pipeline/stages/depth_estimation.py` - Plane sweep stereo for ring cameras
- `src/aquamvs/pipeline/stages/fusion.py` - Consistency filtering + fusion + outlier removal
- `src/aquamvs/pipeline/stages/surface.py` - Surface reconstruction + visualization (full and sparse)

### Modified
None - pure extraction

## Decisions Made

1. **Internal-only stage package**: Stage modules are NOT exported from pipeline package __init__.py. They are implementation details of the pipeline runner (to be created in Plan 03). Users interact with run_pipeline, not individual stages.

2. **Pure function design**: All stage functions follow pattern `run_X_stage(inputs, ctx, frame_dir, frame_idx) -> outputs`. No mutable state, explicit inputs/outputs. This enables:
   - Easy testing (pass mock inputs, verify outputs)
   - Clear data flow (no hidden dependencies)
   - Potential parallelization (stages don't share state)

3. **Embedded visualization and I/O**: Rather than separate viz stages, each stage owns its visualization/I/O operations:
   - sparse_matching.py: feature overlays, save features/matches
   - dense_matching.py: depth map viz, save depth maps
   - depth_estimation.py: depth map viz, save depth maps
   - fusion.py: save consistency maps, save point cloud
   - surface.py: scene viz, rig viz, save mesh
   - All gated by _should_viz(config, stage) and config.runtime.save_* flags

4. **No cross-stage imports**: Stages do NOT import from each other (e.g., dense_matching doesn't import from sparse_matching). This avoids coupling and keeps dependencies explicit through function parameters.

5. **Delegation to domain modules**: Stages are thin orchestration layers. They call domain functions from features/, dense/, fusion/, triangulation/, surface/ and handle I/O/viz. No logic duplication.

6. **Execution path traceability**: Each of the 4 execution paths is now clear:
   - **LightGlue + sparse**: run_lightglue_path -> run_triangulation -> run_sparse_surface_stage
   - **LightGlue + full**: run_lightglue_path -> run_triangulation -> run_depth_estimation -> run_fusion_stage -> run_surface_stage
   - **RoMa + sparse**: run_roma_sparse_path -> run_triangulation -> run_sparse_surface_stage
   - **RoMa + full**: run_roma_full_path -> run_fusion_stage (skip_filter=True) -> run_surface_stage

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - extraction was straightforward. All imports resolved correctly.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All stage modules created and verified (import successful, no circular dependencies)
- Stage functions ready for pipeline runner integration (Plan 03)
- Execution paths clearly separated and traceable
- Visualization and I/O operations embedded and configurable
- **Next step**: Create process_frame and run_pipeline in Plan 03 to orchestrate stage calls

## Self-Check: PASSED

All created files exist:
- [x] src/aquamvs/pipeline/stages/__init__.py
- [x] src/aquamvs/pipeline/stages/undistortion.py
- [x] src/aquamvs/pipeline/stages/sparse_matching.py
- [x] src/aquamvs/pipeline/stages/dense_matching.py
- [x] src/aquamvs/pipeline/stages/depth_estimation.py
- [x] src/aquamvs/pipeline/stages/fusion.py
- [x] src/aquamvs/pipeline/stages/surface.py

All commits exist:
- [x] 61e7546 (Task 1)
- [x] cf96d8b (Task 2)

All imports verified:
- [x] All 8 stage functions import successfully
- [x] No circular imports detected
- [x] Stages delegate to domain modules (features, dense, fusion, triangulation, surface)

---
*Phase: 03-pipeline-decomposition-and-modularization*
*Completed: 2026-02-14*
