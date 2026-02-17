---
phase: 07-post-qa-bug-triage
plan: 02
subsystem: profiling
tags: [threading, profiler, timed_stage, pipeline, thread-local]

# Dependency graph
requires:
  - phase: 05-performance
    provides: PipelineProfiler and timed_stage context manager
provides:
  - Thread-local profiler registry (set_active_profiler, get_active_profiler)
  - Updated timed_stage that delegates to PipelineProfiler when active
  - Fixed profile_pipeline wiring via set_active_profiler
affects: [07-03-PLAN, benchmark-command, profiling]

# Tech tracking
tech-stack:
  added: [threading.local (stdlib)]
  patterns: [thread-local registry pattern for zero-intrusion pipeline instrumentation]

key-files:
  created: []
  modified:
    - src/aquamvs/profiling/profiler.py
    - src/aquamvs/profiling/__init__.py
    - tests/test_config.py

key-decisions:
  - "Thread-local storage (threading.local) chosen for profiler registry — ensures per-thread isolation and avoids global mutable state"
  - "timed_stage else-branch identical to old implementation — backward-compatible, no behavior change for normal aquamvs run usage"
  - "profile_pipeline wraps process_frame in try/finally to ensure set_active_profiler(None) always runs even on exception"

patterns-established:
  - "Thread-local registry pattern: _profiler_local = threading.local() with set/get helpers — reusable for other per-thread state"
  - "Zero-intrusion instrumentation: pipeline stage files require no changes to gain profiling capability"

# Metrics
duration: 13min
completed: 2026-02-17
---

# Phase 07 Plan 02: Profiler Wiring Summary

**Thread-local registry pattern bridges timed_stage and PipelineProfiler without touching any pipeline stage files, fixing the empty profiler report table QA issue**

## Performance

- **Duration:** 13 min
- **Started:** 2026-02-17T20:08:47Z
- **Completed:** 2026-02-17T20:22:24Z
- **Tasks:** 1
- **Files modified:** 3

## Accomplishments

- Added thread-local profiler registry (`set_active_profiler`, `get_active_profiler`) to `profiler.py`
- Updated `timed_stage` to delegate to `profiler.stage(name)` when an active profiler is set, falling back to log-only behavior when none is active
- Fixed `profile_pipeline` to wire the profiler via `set_active_profiler` with `try/finally` safety
- Exported new registry functions from `profiling/__init__.py`
- Auto-fixed breaking config test regression caused by plan 01's quality-preset behavioral change

## Task Commits

The plan 02 changes were committed as part of commit `5b5c3ee` (plan 01 execution bundled them together):

1. **Task 1: Thread-local profiler registry + timed_stage update** - `5b5c3ee` (fix: config cleanup — deprecated keys, quality presets to init-time)

Note: The profiler wiring changes were committed together with plan 01's config changes in a single commit. All verifications pass against HEAD.

## Files Created/Modified

- `src/aquamvs/profiling/profiler.py` - Added `threading` import, `_profiler_local`, `set_active_profiler`, `get_active_profiler`; updated `timed_stage` to delegate to profiler when active; fixed `profile_pipeline` to use `set_active_profiler`
- `src/aquamvs/profiling/__init__.py` - Added `set_active_profiler` and `get_active_profiler` to imports and `__all__`
- `tests/test_config.py` - Updated preset tests to reflect new behavior (preset stored but not auto-applied at runtime; `apply_preset()` still works explicitly)

## Decisions Made

- Thread-local storage (`threading.local`) chosen for profiler registry — ensures per-thread isolation, safe for concurrent pipeline runs
- `timed_stage` else-branch is identical to the pre-plan implementation — backward-compatible; `aquamvs run` unaffected
- `profile_pipeline` uses `try/finally` to guarantee `set_active_profiler(None)` always runs, preventing profiler leakage between calls

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated quality-preset tests broken by plan 01's behavioral change**
- **Found during:** Task 1 (running full test suite verification)
- **Issue:** Plan 01 changed `auto_apply_preset` validator from silently applying presets to emitting a deprecation warning. Tests `test_fast_preset_sets_expected_values`, `test_balanced_preset_sets_expected_values`, `test_quality_preset_sets_expected_values`, `test_explicit_values_override_preset`, `test_partial_override_preset`, `test_preset_round_trip_yaml`, and `test_preset_from_string` still expected auto-application behavior (`1 failed, 106 passed`)
- **Fix:** Updated all affected tests to call `apply_preset()` explicitly instead of relying on auto-application. Added `test_quality_preset_not_auto_applied` to positively verify the new behavior. Updated round-trip and string tests to expect default values (not preset values) after construction
- **Files modified:** `tests/test_config.py`
- **Verification:** `pytest tests/test_config.py` — 51 passed; `pytest tests/` — 600 passed
- **Committed in:** `5b5c3ee` (bundled with plan 01 commit — already in HEAD)

---

**Total deviations:** 1 auto-fixed (Rule 1 — Bug)
**Impact on plan:** Required to get tests from 1 failure to 600 passed. The test behavioral contracts were outdated after plan 01's intentional breaking change to preset handling.

## Issues Encountered

- Plan 02 changes were already committed to HEAD as part of the plan 01 execution bundle (commit `5b5c3ee`). The edits applied correctly (idempotent), all verifications passed, and 600 tests pass.

## Next Phase Readiness

- Plan 03 (benchmark command) can now call `set_active_profiler(profiler)` before running the pipeline and `PipelineProfiler.snapshots` will be populated with per-stage timing data
- No changes required in any pipeline stage file — zero-intrusion wiring complete
- `profile_pipeline()` also fixed as a side effect (usable for `aquamvs profile` until plan 03 replaces it)

## Self-Check: PASSED

- `src/aquamvs/profiling/profiler.py` — FOUND
- `src/aquamvs/profiling/__init__.py` — FOUND
- Commit `5b5c3ee` — FOUND

---
*Phase: 07-post-qa-bug-triage*
*Completed: 2026-02-17*
