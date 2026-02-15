---
phase: quick-3
plan: 01
subsystem: testing
tags: [pytest, mocking, mock.patch, pydantic]

# Dependency graph
requires:
  - phase: 02-config-modernization
    provides: Pydantic config classes and pipeline refactoring
provides:
  - All pipeline tests passing after Pydantic config migration
affects: [testing, ci]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Mock patch paths must target where functions are imported (used), not where defined

key-files:
  created: []
  modified:
    - tests/test_pipeline.py

key-decisions:
  - "Added separate mock for save_point_cloud in surface stage (sparse mode) vs fusion stage (full mode)"

patterns-established:
  - "Mock.patch() targeting pattern: patch where function is bound in namespace after import, not its definition location"

# Metrics
duration: 14min
completed: 2026-02-15
---

# Quick Task 3: Fix Test Failures from Pydantic Config Migration Summary

**Corrected mock patch paths in 6 failing pipeline tests caused by incorrect namespace targeting after Phase 2 refactoring**

## Performance

- **Duration:** 14 min
- **Started:** 2026-02-15T05:20:08Z
- **Completed:** 2026-02-15T05:34:13Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Fixed 6 failing tests in test_pipeline.py by correcting mock patch paths
- All 42 pipeline tests now passing
- Restored CI green status after Pydantic config migration

## Task Commits

1. **Task 1: Fix mock patch paths and add sparse mode save_pcd mock** - `53ff2f8` (test)

## Files Created/Modified
- `tests/test_pipeline.py` - Fixed 4 mock patch paths to target import locations

## Decisions Made

**Mock patch strategy for functions imported across modules:**
- Patched `_collect_height_maps` at `aquamvs.pipeline.runner` (where imported), not `aquamvs.pipeline.helpers` (where defined)
- Patched `_sparse_cloud_to_open3d` at `aquamvs.pipeline.stages.surface` (where used)
- Patched `load_all_masks` at `aquamvs.pipeline.builder` (where imported)
- Added separate `save_point_cloud` patches for fusion stage (full mode, saves fused.ply) and surface stage (sparse mode, saves sparse.ply)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

**Initial fix incomplete:** After fixing the first 4 mock patch paths, 2 sparse mode tests still failed. Investigation revealed that `save_point_cloud` is called from different modules in sparse vs full mode:
- Full mode: `fusion.py` saves fused.ply
- Sparse mode: `surface.py` saves sparse.ply

**Resolution:** Added second mock patch for `aquamvs.pipeline.stages.surface.save_point_cloud` alongside existing `aquamvs.pipeline.stages.fusion.save_point_cloud` patch. Updated sparse mode test assertions to check the correct mock (`save_pcd_sparse` instead of `save_pcd`).

## Next Phase Readiness

All pipeline tests passing. Integration test failure (`test_end_to_end_reconstruction`) is pre-existing and unrelated to this fix.

Full test suite: 617 passed, 1 failed (pre-existing), 3 skipped

## Self-Check: PASSED

- ✓ File exists: tests/test_pipeline.py
- ✓ Commit exists: 53ff2f8
- ✓ All 42 pipeline tests pass

---
*Phase: quick-3*
*Completed: 2026-02-15*
