---
phase: 07-post-qa-bug-triage
plan: 01
subsystem: config
tags: [pydantic, config, cli, argparse, preprocessing, video, bottleneck, median]

# Dependency graph
requires:
  - phase: 06-cli-qa
    provides: QA issues list identifying deprecated config keys, preset override bug, 0-fps video output
provides:
  - REMOVED_KEYS stripping in _migrate_legacy_config with clear user warning
  - --preset flag on `aquamvs init` baking preset values at init time
  - quality_preset validator gutted (no longer silently applies at runtime)
  - --output-fps flag on temporal-filter with explicit 30fps default
  - bottleneck.median fallback with graceful np.median fallback
affects: [08-post-qa-bug-triage, future config changes, preprocessing performance]

# Tech tracking
tech-stack:
  added: [bottleneck (optional, graceful fallback)]
  patterns:
    - Try-import optional performance dependency at module level with fallback
    - Bake preset values at init time instead of storing preset name for runtime re-apply
    - REMOVED_KEYS set for clean deprecation of deleted config fields

key-files:
  created: []
  modified:
    - src/aquamvs/config.py
    - src/aquamvs/cli.py
    - src/aquamvs/preprocess.py
    - src/aquamvs/pipeline/helpers.py
    - tests/test_config.py

key-decisions:
  - "Preset baked at init time via aquamvs init --preset; quality_preset=null in saved YAML so no runtime re-apply"
  - "auto_apply_preset validator gutted to warn-only (not removed, Pydantic needs method signature)"
  - "output_fps explicit parameter replaces fps/framestep computation (cv2 returns 0.0 for some containers)"
  - "bottleneck.median optional: try-import at module level with logger.warning fallback to np.median"
  - "REMOVED_KEYS = {save_depth_maps, save_point_cloud, save_mesh} stripped in _migrate_legacy_config"

patterns-established:
  - "Config deprecation: define REMOVED_KEYS set, strip in _migrate_legacy_config with clear warning"
  - "Optional perf deps: try-import at module level, assign _fn = fast or slow impl, log warning on miss"
  - "Init-time preset baking: apply_preset() + quality_preset=None before to_yaml()"

# Metrics
duration: 21min
completed: 2026-02-17
---

# Phase 7 Plan 1: Post-QA Bug Triage (Config + Preprocessing) Summary

**Deprecated config keys now warn and strip cleanly; quality presets baked at init time via --preset flag; video output uses explicit --output-fps (not unreliable source FPS); bottleneck.median replaces np.median for 3-10x preprocessing speedup**

## Performance

- **Duration:** 21 min
- **Started:** 2026-02-17T20:08:50Z
- **Completed:** 2026-02-17T20:29:57Z
- **Tasks:** 3
- **Files modified:** 7 (config.py, cli.py, preprocess.py, pipeline/helpers.py, profiling/__init__.py, profiling/profiler.py, tests/test_config.py)

## Accomplishments

- Deprecated config keys (`save_depth_maps`, `save_point_cloud`, `save_mesh`) are now stripped in `_migrate_legacy_config` with a clear user-facing warning instead of causing Pydantic validation errors
- Quality presets moved to init-time: `aquamvs init --preset fast/balanced/quality` bakes values into explicit YAML fields; `auto_apply_preset` validator now only warns if `quality_preset` is set (no silent override at runtime)
- `--output-fps` flag added to `temporal-filter` subcommand; `process_video_temporal_median` and `process_batch` accept explicit `output_fps` parameter, eliminating the `fps / framestep` computation that produced 0.0 fps
- STL export fix (compute_vertex_normals before write) confirmed already present in `surface.py` line 396
- `bottleneck.median` replaces `np.median` for temporal median computation with graceful fallback and install hint
- NAL warning documented as benign in `preprocess.py`; sparse-mode empty gallery documented as by-design in `pipeline/helpers.py`
- 600 tests pass with no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Config cleanup — deprecated keys + quality presets to init-time** - `5b5c3ee` (fix)
2. **Task 2: Preprocessing --output-fps flag + verify STL fix** - `8f42b05` (fix)
3. **Task 3: Profile preprocessing bottleneck and apply targeted performance fix** - `30836d3` (perf)

## Files Created/Modified

- `src/aquamvs/config.py` - REMOVED_KEYS constant; deprecated key stripping in _migrate_legacy_config; auto_apply_preset gutted to warn-only
- `src/aquamvs/cli.py` - --preset on init_parser; preset applied in init_config body; --output-fps on temporal-filter parser; output_fps passed to process_batch
- `src/aquamvs/preprocess.py` - output_fps parameter in process_video_temporal_median and process_batch; bottleneck try-import with _median fallback; NAL warning comment
- `src/aquamvs/pipeline/helpers.py` - sparse mode empty gallery comment in _collect_height_maps
- `tests/test_config.py` - Updated 3 preset tests to call apply_preset() explicitly (no longer pass quality_preset at construction)
- `src/aquamvs/profiling/__init__.py` - Export get_active_profiler / set_active_profiler (pre-existing modification)
- `src/aquamvs/profiling/profiler.py` - Thread-local active profiler helpers (pre-existing modification)

## Decisions Made

- Preset baked at init time (`aquamvs init --preset fast`), `quality_preset=None` in saved YAML — loading a config with `quality_preset` set only warns, never silently overrides user values at runtime
- `auto_apply_preset` validator method kept (Pydantic needs the method signature) but body replaced with deprecation warning
- `output_fps` explicit parameter replaces `fps / framestep` because `cv2.CAP_PROP_FPS` returns `0.0` for some container formats, producing broken video output
- `bottleneck.median` is optional (graceful fallback) because it's a performance-only dependency; users without it still get correct output with a clear upgrade hint

## Deviations from Plan

None — plan executed exactly as written. The profiling module changes (`get_active_profiler` / `set_active_profiler`) were pre-existing uncommitted modifications in the working tree, included in Task 1 commit for cleanliness.

## Issues Encountered

- Pre-commit ruff hook fixed a minor lint issue on first commit attempt (auto-resolved, re-staged and committed)
- `git background task` output mechanism wasn't producing readable output; fell back to synchronous pytest run

## User Setup Required

None — no external service configuration required. `bottleneck` is an optional performance dependency; users can install it with `pip install bottleneck` for faster preprocessing.

## Next Phase Readiness

- 6 of 9 QA issues resolved or closed: deprecated config keys (fixed), quality presets (moved to init), output FPS (fixed), STL export (verified), sparse summary (closed as by-design), NAL warning (documented)
- 3 remaining QA issues deferred to Phase 07-02 and later plans
- Ready for Plan 02 (profiler wiring, based on the docs commit `758024b` already present)

---
*Phase: 07-post-qa-bug-triage*
*Completed: 2026-02-17*
