---
phase: quick-5
plan: 01
subsystem: pipeline
tags: [gpu-memory, visualization, two-pass, windows-crash-fix]
dependency-graph:
  requires: []
  provides: [two-pass-pipeline, deferred-viz, deferred-cleanup]
  affects: [pipeline, config, visualization, stages]
tech-stack:
  added: []
  patterns: [compute-then-viz, reload-from-disk, deferred-cleanup]
key-files:
  created:
    - src/aquamvs/pipeline/visualization.py
  modified:
    - src/aquamvs/config.py
    - src/aquamvs/pipeline/runner.py
    - src/aquamvs/pipeline/__init__.py
    - src/aquamvs/pipeline/stages/undistortion.py
    - src/aquamvs/pipeline/stages/dense_matching.py
    - src/aquamvs/pipeline/stages/depth_estimation.py
    - src/aquamvs/pipeline/stages/sparse_matching.py
    - src/aquamvs/pipeline/stages/fusion.py
    - src/aquamvs/pipeline/stages/surface.py
    - docs/cli_guide.md
    - tests/test_config.py
    - tests/test_pipeline.py
decisions:
  - Always save depth maps, point clouds, and meshes (removed save toggles)
  - Features viz forces save_features=True via config validator
  - Undistorted images saved to disk only when features viz is active
  - Intermediate cleanup deferred to after viz pass
metrics:
  duration: ~25
  completed: 2026-02-16T22:30:00Z
---

# Quick Task 5: Two-Pass Pipeline (Compute then Viz)

Split pipeline into compute pass and visualization pass to fix GPU memory crashes on Windows where Open3D's OpenGL context competes with CUDA.

## One-liner

Split `run_pipeline()` into a compute pass (all frames) followed by `torch.cuda.empty_cache()` and a viz pass that reloads artifacts from disk, eliminating Open3D/CUDA GPU memory conflicts.

## What Changed

### Config Simplification

**src/aquamvs/config.py:**
- Removed `save_depth_maps`, `save_point_cloud`, `save_mesh` toggles from `RuntimeConfig` — these outputs are now always saved (viz pass needs them on disk)
- Added `model_validator` on `PipelineConfig` to force `save_features=True` when features visualization is active (empty `viz_stages` or `"features"` in list)
- Updated docstrings to reflect removed fields

### New Viz Pass Module

**src/aquamvs/pipeline/visualization.py (NEW):**
- `run_visualization_pass(config, ctx)` — iterates frame directories, renders all viz stages
- `_run_frame_viz()` — orchestrates per-frame viz (depth, features, scene, rig)
- `_viz_depth()` — reloads `.npz` depth maps via `load_depth_map()`, calls `render_all_depth_maps()`
- `_viz_features()` — reloads undistorted PNGs + `.pt` features/matches, calls `render_all_features()`
- `_viz_scene()` — reloads `.ply` point cloud and mesh, calls `render_all_scenes()`
- `_viz_rig()` — uses `ctx.calibration` + optional point cloud overlay, calls `render_rig_diagram()`
- Summary viz (`render_timeseries_gallery`) moved here from runner
- All viz functions wrapped in try/except to isolate failures

### Runner Refactor

**src/aquamvs/pipeline/runner.py:**
- `run_pipeline()` now runs two sequential loops: compute pass → `cuda.empty_cache()` → viz pass
- Viz pass only runs when `config.runtime.viz_enabled` is True
- Added `_cleanup_intermediates()` for deferred deletion of depth maps and undistorted images after viz pass
- Summary viz removed from runner (moved to visualization module)
- Removed imports: `_collect_height_maps`, `_should_viz`
- Added imports: `torch`, `shutil`, `run_visualization_pass`

### Stage Cleanup (6 files)

**src/aquamvs/pipeline/stages/undistortion.py:**
- Added `frame_dir` parameter (optional, for saving undistorted images)
- When features viz is active, saves undistorted images to `frame_dir/undistorted/{cam}.png`

**src/aquamvs/pipeline/stages/dense_matching.py:**
- Removed `_should_viz` import and depth viz block (lines 74-90)
- Removed `if config.runtime.save_depth_maps:` guard — always saves

**src/aquamvs/pipeline/stages/depth_estimation.py:**
- Removed `_should_viz` import and depth viz block (lines 111-128)
- Removed `if config.runtime.save_depth_maps:` guard — always saves

**src/aquamvs/pipeline/stages/sparse_matching.py:**
- Removed `_should_viz` import and features viz block (lines 85-117)

**src/aquamvs/pipeline/stages/fusion.py:**
- Removed `if config.runtime.save_point_cloud:` guard — always saves
- Removed `keep_intermediates` cleanup — deferred to after viz pass

**src/aquamvs/pipeline/stages/surface.py:**
- Removed `_should_viz` import
- Removed `if config.runtime.save_mesh:` guard — always saves (both full and sparse paths)
- Removed `torch.cuda.empty_cache()` calls (moved to runner between passes)
- Removed scene viz block and rig viz block (both full and sparse paths)

### Docs & Tests

**docs/cli_guide.md:**
- Removed `save_depth_maps`, `save_point_cloud`, `save_mesh` from Output Control section
- Updated output tree annotations to say "always saved" instead of config toggles
- Updated data output descriptions

**tests/test_config.py:**
- Removed assertions for `save_depth_maps`, `save_point_cloud`, `save_mesh` fields
- Removed `save_depth_maps=False` from custom values test

**tests/test_pipeline.py:**
- Removed `test_skip_depth_maps`, `test_skip_point_cloud`, `test_skip_mesh` tests (toggles removed)
- Removed `test_cleanup_intermediates` (cleanup now deferred, not testable via process_frame)
- Updated `test_sparse_cloud_always_saved` to remove deleted config fields
- Replaced `test_viz_enabled_all_stages`, `test_viz_enabled_specific_stages`, `test_viz_error_does_not_crash_pipeline` with:
  - `test_viz_not_called_in_process_frame` — verifies viz does NOT run in compute pass
  - `test_viz_pass_calls_depth_viz` — verifies viz pass renders depth maps from disk
  - `test_viz_pass_error_does_not_crash` — verifies viz pass error isolation
- Replaced `test_sparse_mode_scene_viz` and `test_sparse_mode_rig_viz` with `test_sparse_mode_no_viz_in_process_frame`
- Updated summary viz tests to patch `run_visualization_pass` and `_collect_height_maps` at new module paths

**src/aquamvs/pipeline/__init__.py:**
- Added `run_visualization_pass` to exports and `__all__`

## Deviations from Plan

None — plan executed as written.

## Verification

- [x] `pytest tests/test_config.py` — 51 passed
- [x] `pytest tests/test_pipeline.py` — 37 passed
- [x] `pytest tests/` — 598 passed, 1 failed (pre-existing), 3 skipped
- [x] Pre-existing failure confirmed: `test_end_to_end_reconstruction` fails identically on old code (empty fused cloud on synthetic scene)
- [x] Todo captured for pre-existing failure

## Files Modified

| File | Lines Changed | Purpose |
|------|---------------|---------|
| src/aquamvs/config.py | -6 fields, +10 validator | Remove save toggles, add features viz validator |
| src/aquamvs/pipeline/visualization.py | +290 (new) | Viz pass: reload artifacts from disk, render all stages |
| src/aquamvs/pipeline/runner.py | +40, -25 | Two-pass loop, deferred cleanup |
| src/aquamvs/pipeline/__init__.py | +2 | Export run_visualization_pass |
| src/aquamvs/pipeline/stages/undistortion.py | +15, -3 | Save undistorted images for features viz |
| src/aquamvs/pipeline/stages/dense_matching.py | -20 | Strip viz, force save |
| src/aquamvs/pipeline/stages/depth_estimation.py | -22 | Strip viz, force save |
| src/aquamvs/pipeline/stages/sparse_matching.py | -33 | Strip features viz |
| src/aquamvs/pipeline/stages/fusion.py | -10 | Force save, remove inline cleanup |
| src/aquamvs/pipeline/stages/surface.py | -100 | Strip all viz, force saves, remove cuda cache |
| docs/cli_guide.md | ~15 | Update output docs for always-save |
| tests/test_config.py | -6 | Remove deleted field assertions |
| tests/test_pipeline.py | -80, +60 | Rewrite viz tests for two-pass architecture |

## Architecture

```
run_pipeline(config)
├── build_pipeline_context(config)
│
├── ── Compute Pass ──
│   └── for frame in frames:
│       └── process_frame(frame_idx, images, ctx)
│           ├── run_undistortion_stage()     # saves undistorted/ if features viz active
│           ├── run_lightglue_path()         # saves features/ if save_features
│           │   or run_roma_full_path()      # always saves depth_maps/
│           ├── run_triangulation()          # always saves sparse/
│           ├── run_depth_estimation()       # always saves depth_maps/
│           ├── run_fusion_stage()           # always saves point_cloud/
│           └── run_surface_stage()          # always saves mesh/
│
├── torch.cuda.empty_cache()                 # free GPU before OpenGL
│
├── ── Viz Pass ──
│   └── run_visualization_pass(config, ctx)
│       ├── for frame_dir in frame_*/:
│       │   ├── _viz_depth()      # reload .npz → render PNGs
│       │   ├── _viz_features()   # reload .png + .pt → render overlays
│       │   ├── _viz_scene()      # reload .ply → render 3D views
│       │   └── _viz_rig()        # ctx.calibration → render diagram
│       └── summary viz           # height maps → gallery
│
└── _cleanup_intermediates()                 # deferred depth_maps/ + undistorted/ deletion
```

## Self-Check: PASSED

**Created files:**
- FOUND: src/aquamvs/pipeline/visualization.py

**Modified files:**
- FOUND: src/aquamvs/config.py
- FOUND: src/aquamvs/pipeline/runner.py
- FOUND: src/aquamvs/pipeline/stages/undistortion.py
- FOUND: src/aquamvs/pipeline/stages/dense_matching.py
- FOUND: src/aquamvs/pipeline/stages/depth_estimation.py
- FOUND: src/aquamvs/pipeline/stages/sparse_matching.py
- FOUND: src/aquamvs/pipeline/stages/fusion.py
- FOUND: src/aquamvs/pipeline/stages/surface.py
- FOUND: docs/cli_guide.md
- FOUND: tests/test_config.py
- FOUND: tests/test_pipeline.py

**Test results:** 598 passed, 1 pre-existing failure, 3 skipped
