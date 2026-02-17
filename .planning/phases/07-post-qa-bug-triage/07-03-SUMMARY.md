---
phase: 07-post-qa-bug-triage
plan: 03
subsystem: benchmark
tags: [cli, benchmark, profiler, pathway-comparison, tabulate, open3d]

# Dependency graph
requires:
  - phase: 07-02-PLAN
    provides: Thread-local profiler registry (set_active_profiler) that wires PipelineProfiler to timed_stage
  - phase: 05-performance
    provides: PipelineProfiler, timed_stage, ProfileReport
provides:
  - Unified aquamvs benchmark command using PipelineConfig (not BenchmarkConfig)
  - run_benchmark() that runs 4+ pathways with isolated profiler instances
  - build_pathways() that creates (name, config) variants for extractor/CLAHE combinations
  - Per-stage timing table (tabulate) + markdown report saved to output_dir
  - profile command fully removed; all profiling now via benchmark
affects: [future-benchmark-extensions, cli-users]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Fresh-profiler-per-pathway: instantiate PipelineProfiler() inside loop, not outside — prevents snapshots dict collision"
    - "Args-object dispatch: benchmark_command(args) receives argparse Namespace directly — cleaner than individual keyword args"

key-files:
  created: []
  modified:
    - src/aquamvs/benchmark/__init__.py
    - src/aquamvs/benchmark/runner.py
    - src/aquamvs/benchmark/metrics.py
    - src/aquamvs/benchmark/report.py
    - src/aquamvs/cli.py
    - tests/test_cli.py
  deleted:
    - src/aquamvs/benchmark/config.py
    - src/aquamvs/benchmark/datasets.py
    - src/aquamvs/benchmark/comparison.py
    - src/aquamvs/benchmark/visualization.py
    - src/aquamvs/benchmark/synthetic.py
    - src/aquamvs/benchmark/synthetic_benchmark.py
    - src/aquamvs/profiling/synthetic_profile.py

key-decisions:
  - "aquamvs benchmark accepts a PipelineConfig YAML (not BenchmarkConfig) — removes the synthetic/dataset model entirely"
  - "Fresh PipelineProfiler() per pathway prevents snapshots dict collision across pathways"
  - "cloud_density computed from bounding-box XY area as scan-area proxy — not true surface area, but useful relative comparison"
  - "profile CLI command removed — aquamvs benchmark subsumes it with multi-pathway comparison"

patterns-established:
  - "Fresh-profiler-per-pathway: each pathway loop iteration creates PipelineProfiler(), calls set_active_profiler(), runs process_frame(), then set_active_profiler(None)"
  - "Args-object CLI dispatch: benchmark_command(args) receives argparse Namespace directly for easier extensibility"

# Metrics
duration: 13min
completed: 2026-02-17
---

# Phase 07 Plan 03: Benchmark Command Rebuild Summary

**Unified `aquamvs benchmark config.yaml` replaces both old broken benchmark (BenchmarkConfig/synthetic) and old profile commands — runs all 4 pipeline pathways with fresh per-pathway profiler instances and prints timing + cloud density comparison table**

## Performance

- **Duration:** 13 min
- **Started:** 2026-02-17T20:34:01Z
- **Completed:** 2026-02-17T20:47:24Z
- **Tasks:** 2
- **Files modified:** 7 (plus 7 deleted)

## Accomplishments

- Deleted 7 stale files: old BenchmarkConfig/dataset/synthetic/visualization benchmark code and synthetic_profile.py
- New `benchmark/runner.py`: `run_benchmark()` loads PipelineConfig, calls `build_pathways()` to generate 4+ variants, runs each with a fresh isolated `PipelineProfiler` instance wired via `set_active_profiler`, collects timing + point cloud metrics
- New `benchmark/metrics.py`: `compute_relative_metrics()` loads `fused_points.ply` via Open3D, counts points, computes cloud density from bounding-box XY area
- New `benchmark/report.py`: `format_console_table()` with tabulate grid layout, `format_markdown_report()` with system info, `save_markdown_report()` timestamped to output_dir
- New `benchmark/__init__.py`: clean public API with 7 exports
- New CLI: `aquamvs benchmark config.yaml [--frame N] [--extractors list] [--with-clahe]`; `profile` subcommand fully removed
- Updated `test_cli.py` benchmark tests to match new args-object calling convention

## Task Commits

1. **Task 1: Delete old benchmark/profiler code, build new benchmark module** - `bb59f9a` (feat)
2. **Task 2: Replace CLI benchmark/profile commands with new unified benchmark** - `80eed6b` (feat)

## Files Created/Modified

- `src/aquamvs/benchmark/__init__.py` - Clean public API: run_benchmark, BenchmarkResult, PathwayResult, build_pathways, format_console_table, format_markdown_report, save_markdown_report
- `src/aquamvs/benchmark/runner.py` - Core benchmark runner with fresh-profiler-per-pathway pattern
- `src/aquamvs/benchmark/metrics.py` - compute_relative_metrics() from fused_points.ply
- `src/aquamvs/benchmark/report.py` - Console table (tabulate) and markdown report formatting
- `src/aquamvs/cli.py` - New benchmark_command(args), profile subcommand removed
- `tests/test_cli.py` - Updated benchmark tests to check args.config, args.frame, args.extractors, args.with_clahe

## Decisions Made

- aquamvs benchmark accepts a PipelineConfig YAML (same as aquamvs run) — drops the synthetic/BenchmarkConfig/dataset model entirely
- Fresh PipelineProfiler() per pathway: profiler.snapshots is a plain dict so reusing one profiler across pathways would overwrite same-named stage keys
- cloud_density uses bounding-box XY area as scan-area proxy — not true surface area, but gives a useful relative comparison without requiring surface reconstruction
- profile CLI command removed — `aquamvs benchmark` subsumes it with multi-pathway comparison; single-pathway profiling available via Python API

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] ruff B905 — zip() without explicit strict= parameter**
- **Found during:** Task 1 (pre-commit hook during commit attempt)
- **Issue:** Two `zip()` calls in the tabulate fallback path of `report.py` lacked `strict=` parameter, triggering ruff B905
- **Fix:** Added `strict=False` to both zip calls in the fallback formatter
- **Files modified:** `src/aquamvs/benchmark/report.py`
- **Verification:** Pre-commit ruff hook passed on second commit attempt
- **Committed in:** `bb59f9a` (included in Task 1 commit after fix)

---

**Total deviations:** 1 auto-fixed (Rule 1 — Bug: lint error)
**Impact on plan:** Trivial lint fix, no behavior change.

## Issues Encountered

- Pre-commit hook rejected first commit attempt due to ruff B905 on zip() calls in fallback formatter. Fixed inline, re-staged, and committed cleanly.

## Next Phase Readiness

- `aquamvs benchmark config.yaml` is functional end-to-end: runs 4 base pathways, collects timing from profiler, saves markdown report
- Plan 03 is the final plan in Phase 07 — phase is now complete
- The QA-identified issues (broken benchmark, empty profile table) are both resolved

## Self-Check: PASSED

- `src/aquamvs/benchmark/runner.py` — FOUND
- `src/aquamvs/benchmark/metrics.py` — FOUND
- `src/aquamvs/benchmark/report.py` — FOUND
- `src/aquamvs/benchmark/__init__.py` — FOUND
- `src/aquamvs/cli.py` — FOUND
- Commit `bb59f9a` — FOUND
- Commit `80eed6b` — FOUND

---
*Phase: 07-post-qa-bug-triage*
*Completed: 2026-02-17*
