---
phase: 07-post-qa-bug-triage
verified: 2026-02-17T20:51:40Z
status: passed
score: 14/14 must-haves verified
re_verification: false
---

# Phase 7: Post-QA Bug Triage Verification Report

**Phase Goal:** Fix bugs and issues discovered during Phase 6 CLI QA execution, prioritizing medium-impact items that affect core functionality
**Verified:** 2026-02-17T20:51:40Z
**Status:** passed
**Re-verification:** No - initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Old configs with save_depth_maps/save_point_cloud/save_mesh keys load without crashing | VERIFIED | REMOVED_KEYS set at config.py:22; stripping loop in _migrate_legacy_config lines 515-532; covers top-level and runtime subdict |
| 2 | aquamvs init --preset fast generates config with preset values baked in, quality_preset null | VERIFIED | --preset on init_parser (cli.py:611-616); init_config accepts preset param; apply_preset + quality_preset=None; dispatched via preset=args.preset |
| 3 | Loading a YAML with quality_preset set does NOT silently override user values at runtime | VERIFIED | auto_apply_preset validator (config.py:413-428) warns only, returns self without calling apply_preset() |
| 4 | aquamvs temporal-filter --output-fps 30 writes video at 30 fps regardless of source FPS | VERIFIED | --output-fps on preprocess_parser (cli.py:728-733); temporal_filter_command passes output_fps=args.output_fps; preprocess.py uses param directly; fps/framestep computation gone |
| 5 | STL export works end-to-end | VERIFIED | compute_vertex_normals() called before STL write in surface.py:396 inside has_vertex_normals() guard |
| 6 | Preprocessing median uses bottleneck.median when available, graceful np.median fallback | VERIFIED | Try-import block at module level (preprocess.py:15-24); _median = bn.median or np.median; both call sites (lines 200, 262) use _median(frame_stack...); no np.median(frame_stack anywhere |
| 7 | When active profiler set, timed_stage() delegates to profiler.stage() | VERIFIED | get_active_profiler() checked (profiler.py:60); delegates to profiler.stage(name) when not None (line 62) |
| 8 | When no active profiler, timed_stage() behaves identically to before | VERIFIED | else branch (profiler.py:64-72) identical to pre-plan: CUDA sync + perf_counter + log |
| 9 | Pipeline stages require zero changes - timed_stage signature unchanged | VERIFIED | timed_stage(name: str, logger: logging.Logger) unchanged (profiler.py:46); no stage files modified in any plan |
| 10 | Thread-local storage ensures profiler state is isolated per thread | VERIFIED | _profiler_local = threading.local() (profiler.py:21); set/get via _profiler_local.instance with getattr fallback |
| 11 | aquamvs benchmark config.yaml runs all 4 pathways and prints comparison table | VERIFIED | build_pathways() produces 4 base pathways (runner.py:95-99); format_console_table() renders tabulate grid (report.py:65-128); CLI prints it (cli.py:423) |
| 12 | --extractors and --with-clahe flags add pathway variants | VERIFIED | --extractors (cli.py:681); --with-clahe (cli.py:687); build_pathways() handles both (runner.py:102-124) |
| 13 | Markdown report saved to {output_dir}/benchmark_{timestamp}.md | VERIFIED | save_markdown_report() creates timestamped file (report.py:208-225); CLI calls with Path(result.output_dir); result.output_dir from base_config.output_dir |
| 14 | Old benchmark module files and profile command removed | VERIFIED | benchmark dir has only __init__.py, metrics.py, report.py, runner.py; 6 old files deleted; synthetic_profile.py deleted; .benchmarks/*.py deleted; no profile_command or profile_parser in cli.py |

**Score:** 14/14 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| src/aquamvs/config.py | REMOVED_KEYS; deprecated key stripping; gutted auto_apply_preset | VERIFIED | REMOVED_KEYS at line 22; stripping in _migrate_legacy_config; auto_apply_preset warns only |
| src/aquamvs/cli.py | --preset on init, --output-fps on temporal-filter, new benchmark, profile removed | VERIFIED | All flags present; benchmark_command(args) substantive; output_fps passed through; no profile dispatch |
| src/aquamvs/preprocess.py | output_fps in both functions; bottleneck _median | VERIFIED | output_fps: int = 30 in both functions; _median via try-import; both call sites use _median(frame_stack...) |
| src/aquamvs/profiling/profiler.py | Thread-local registry; updated timed_stage; fixed profile_pipeline | VERIFIED | _profiler_local = threading.local(); set/get_active_profiler; timed_stage delegates; profile_pipeline uses try/finally |
| src/aquamvs/profiling/__init__.py | Exports set_active_profiler and get_active_profiler | VERIFIED | Both in imports and __all__ |
| src/aquamvs/benchmark/runner.py | run_benchmark, BenchmarkResult, PathwayResult, build_pathways; uses set_active_profiler | VERIFIED | All exports present; fresh profiler per pathway; process_frame called per pathway |
| src/aquamvs/benchmark/metrics.py | compute_relative_metrics with Open3D point count and density | VERIFIED | Optional Open3D import; fused_points.ply rglob; bounding-box XY density computation |
| src/aquamvs/benchmark/report.py | format_console_table, format_markdown_report, save_markdown_report | VERIFIED | All three functions; tabulate grid; markdown with system info; timestamped save |
| src/aquamvs/benchmark/__init__.py | Clean public API | VERIFIED | 7 exports: run_benchmark, BenchmarkResult, PathwayResult, build_pathways, format_console_table, format_markdown_report, save_markdown_report |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| cli.py | config.py | init_config passes preset to apply_preset before to_yaml | VERIFIED | init_config(preset=...) calls config.apply_preset(preset) then config.quality_preset = None |
| cli.py | preprocess.py | temporal_filter_command passes output_fps to process_batch | VERIFIED | process_batch(..., output_fps=args.output_fps, ...) at cli.py:467 |
| profiling/profiler.py | pipeline stages | timed_stage checks get_active_profiler before each stage | VERIFIED | profiler = get_active_profiler() at profiler.py:60; delegates to profiler.stage(name) when not None |
| benchmark/runner.py | profiling/profiler.py | set_active_profiler wires profiler before each pathway run | VERIFIED | set_active_profiler(profiler) inside loop; try/finally guarantees cleanup |
| cli.py | benchmark/runner.py | benchmark_command calls run_benchmark | VERIFIED | result = run_benchmark(config_path=..., frame=args.frame, ...) at cli.py:415 |
| benchmark/runner.py | pipeline/runner.py | process_frame called for each pathway | VERIFIED | from ..pipeline.runner import process_frame; called at runner.py:223 |

---

### Anti-Patterns Found

None. No TODO/FIXME/placeholder comments, no stub implementations, no empty returns in wired code paths.

Note: BenchmarkConfig = RuntimeConfig alias at config.py:716 is a pre-existing backward-compatibility
shim pointing to RuntimeConfig, not the deleted benchmark/config.py model. It is intentional.

---

### Human Verification Required

The following items are functional at code level but require real data to confirm end-to-end behavior.

#### 1. Benchmark command end-to-end run

**Test:** Run aquamvs benchmark config.yaml --frame 0 against a real dataset
**Expected:** Console table with 4 pathway rows; markdown report saved to {output_dir}/benchmark_YYYYMMDD_HHMMSS.md
**Why human:** Requires AquaCal calibration data and synchronized video input

#### 2. Deprecated config key warning

**Test:** Create a YAML with save_depth_maps: true and save_mesh: true; run aquamvs run config.yaml
**Expected:** Load succeeds; warning messages for both removed keys; no Pydantic validation error
**Why human:** Requires actual CLI invocation to observe warning output

#### 3. Temporal filter output FPS

**Test:** Run aquamvs temporal-filter video.mp4 --format mp4 --output-fps 30
**Expected:** Output .mp4 file plays at 30 fps regardless of source video FPS
**Why human:** Requires a video file to confirm cv2.VideoWriter produces correct FPS metadata

---

## Commits Verified

All commits referenced in summaries confirmed present in git history:

5b5c3ee: fix(07-01): config cleanup - deprecated keys, quality presets to init-time
8f42b05: fix(07-01): add --output-fps flag to temporal-filter; remove computed fps
30836d3: perf(07-01): use bottleneck.median for temporal median (3-10x faster)
bb59f9a: feat(07-03): rebuild benchmark module as pathway comparison tool
80eed6b: feat(07-03): replace benchmark/profile CLI commands with unified benchmark

---

## Summary

All 14 must-have truths verified. Phase 7 achieved its goal of fixing bugs discovered during Phase 6 CLI QA:

- **Config cleanup (Plan 01):** REMOVED_KEYS stripping, init-time preset baking, output-fps fix, bottleneck.median optimization - all implemented correctly and wired end-to-end.
- **Profiler wiring (Plan 02):** Thread-local registry pattern bridges timed_stage to PipelineProfiler with zero pipeline stage changes. Backward-compatible.
- **Benchmark rebuild (Plan 03):** New unified aquamvs benchmark command with 4-pathway comparison, fresh-profiler-per-pathway pattern, tabulate console table, markdown report, and full removal of old broken benchmark/profile infrastructure.

No stubs, no orphaned code, no anti-patterns found. Three human verification items remain for
confirmation with real data, but all code paths are substantive and wired.

---

_Verified: 2026-02-17T20:51:40Z_
_Verifier: Claude (gsd-verifier)_
