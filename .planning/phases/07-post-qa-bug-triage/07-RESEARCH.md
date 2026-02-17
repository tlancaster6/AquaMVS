# Phase 07: Post-QA Bug Triage - Research

**Researched:** 2026-02-17
**Domain:** Python CLI bug triage, pipeline profiling, benchmark architecture, preprocessing I/O
**Confidence:** HIGH (all findings from direct codebase inspection)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Benchmark & Profiler Rebuild
- Combine benchmark and profile into one command: `aquamvs benchmark` does everything. Drop the separate `profile` command entirely.
- Pathway comparison: Runs all 4 execution paths (LG+SP sparse, LG+SP full, RoMa sparse, RoMa full) on a single frame
- Extractor sweep: Optional `--extractors superpoint,aliked,disk` to test multiple extractors within LightGlue paths
- CLAHE toggle: Optional `--with-clahe` to test with/without CLAHE preprocessing
- Per-stage timing: Each pathway reports wall time per stage (undistortion, matching, depth, fusion, surface)
- Relative accuracy metrics: Point count, cloud density (points/m²), outlier removal %. No absolute accuracy (no ground truth for real data).
- Report output: Console pretty-printed table + markdown file saved to output dir
- Default frame: `--frame` defaults to 0
- Single frame only: No multi-frame averaging — keep it simple
- Real data only: Remove broken synthetic scene code. Build benchmark architecture so synthetic data could plug in later, but don't implement synthetic support now.
- Fix profiler wiring: Stage timing hooks exist but aren't connected to the report table — wire them up as part of the combined benchmark

#### Preprocessing Improvements
- Output FPS: Add `--output-fps` flag (default 30) to fix videos writing at 0.0 fps
- NAL error: Quick investigation (~15 min) to check if it correlates with corrupted frames. Document findings either way (likely benign ffmpeg warning).
- Performance optimization: Profile to identify bottleneck (video decoding vs median calculation), then apply targeted fix (bottleneck.median for computation, or hw-accelerated decode)

#### Fix vs. Remove Decisions
- STL export: Fix it — compute normals before STL write (simple one-liner)
- Quality presets: Move to init-time. `aquamvs init --preset fast` generates config with preset values baked in. No runtime silent override.
- Deprecated config keys: Strip dead code. Remove all references to save_depth_maps, save_point_cloud, save_mesh. Users with old configs get a clear error.
- natsort workaround: Clean up — remove hacky utility inlining in .benchmarks/, use normal natsort import (now in AquaCal deps)

#### Sparse Summary Output
- Investigate first: Have Claude check whether sparse mode is supposed to produce summary output. Fix only if it's a bug, not if it's by design.

#### ImageDirectorySet
- Minimal fix only: Keep the read_frame method added during QA. Skip the shared base class refactor — it works as-is.

### Claude's Discretion
- Exact benchmark table layout and column ordering
- How to structure the markdown report file
- Whether to remove synthetic code in the benchmark plan or a cleanup plan
- Profiler wiring implementation details (how stage timings flow to report)

### Deferred Ideas (OUT OF SCOPE)
- Synthetic scene benchmark support — rebuild from scratch if/when wanted, don't maintain broken Phase 5 code
- ImageDirectorySet / VideoSet shared base class refactor — AquaCal concern, not AquaMVS Phase 7
- Multi-frame benchmark averaging — could add `--frames 0-4` mode later if needed
</user_constraints>

---

## Summary

Phase 07 is a focused bug triage. All 9 issues are now fully mapped to their root causes via direct codebase inspection. The work decomposes into four buckets: (1) benchmark/profiler rebuild — the largest piece, requiring a full CLI redesign from two broken commands into one working `aquamvs benchmark` command; (2) preprocessing I/O fixes — two small bugs (FPS and NAL) plus a profile-then-fix performance task; (3) config cleanup — removing dead keys and moving quality presets to init-time; (4) one investigation already closed (sparse summary output is by design, not a bug).

The most important discovery is that the STL export bug is already fixed in the current codebase (`export_mesh` in `surface.py` already calls `compute_vertex_normals()` before write). This means the STL fix plan can be verified-only, or skipped if the fix was committed before Phase 7 planning. Similarly, the natsort workaround in `.benchmarks/` exists in standalone scripts that are being replaced or deleted as part of the benchmark rebuild — cleanup may happen implicitly.

**Primary recommendation:** Structure Phase 07 as 5-6 plans: (1) sparse summary output investigation (fast, gates the rest), (2) preprocessing fixes, (3) config cleanup (dead keys + quality presets), (4) benchmark/profiler rebuild (the main work), (5) verify/close STL and natsort.

---

## Current State Findings (HIGH confidence)

### Finding 1: STL Export Bug Already Fixed

The error log in `issues-found.md` shows `Write STL failed: compute normals first`. However, the **current** `export_mesh` function in `/src/aquamvs/surface.py` (lines 392-402) already contains the fix:

```python
if output_format == ".stl":
    if not mesh.has_vertex_normals():
        logger.info("Computing vertex normals for STL export")
        mesh.compute_vertex_normals()
```

Verified via `python -c`: `compute_vertex_normals()` + `write_triangle_mesh(*.stl)` succeeds. The fix was introduced between the QA session and now. **Action:** Verify the fix holds end-to-end; no new code needed.

### Finding 2: Profiler Empty Report — Root Cause Confirmed

The profiler empty table is caused by a **disconnect between two separate timing mechanisms**:

- **`timed_stage(name, logger)`** (`profiling/profiler.py` lines 22-42): A lightweight context manager used by all pipeline stages. It only logs elapsed time to the logger — it does NOT write to any profiler instance.
- **`PipelineProfiler.stage(name)`** (`profiling/profiler.py` lines 91-133): The profiler's own context manager that captures wall time, CUDA time, CPU and GPU memory into `self.snapshots`.

`profile_pipeline()` (lines 144-182) creates a `PipelineProfiler` and calls `process_frame()`. But `process_frame()` calls pipeline stage functions that each use `timed_stage()`, not `profiler.stage()`. The profiler's `snapshots` dict is never populated, so `get_report()` returns a report with no stages, and the table is empty.

All 6 pipeline stage files use `timed_stage`:
- `pipeline/stages/undistortion.py`: `with timed_stage("undistortion", logger)`
- `pipeline/stages/sparse_matching.py`: `with timed_stage("sparse_matching", logger)`
- `pipeline/stages/dense_matching.py`: `with timed_stage("dense_matching", logger)`
- `pipeline/stages/depth_estimation.py`: `with timed_stage("depth_estimation", logger)`
- `pipeline/stages/fusion.py`: `with timed_stage("fusion", logger)`
- `pipeline/stages/surface.py`: `with timed_stage("surface_reconstruction", logger)`

**Fix options for profiler wiring (Claude's discretion):**

Option A — Parse log lines: Make `timed_stage` optionally write to a thread-local or global profiler registry that `profile_pipeline` can read after the run. Invasive.

Option B — Replace `timed_stage` with `profiler.stage` in the combined benchmark: When the benchmark runs in timing mode, inject a profiler into the context and have stages call `ctx.profiler.stage()` instead of `timed_stage()`. Requires passing profiler through `PipelineContext`. Cleanest long-term.

Option C — Parse logger output: Run the pipeline, capture the logger output, parse the `"{stage}: {ms} ms"` lines that `timed_stage` already emits, and feed those timings into a report. Minimal invasive change, works immediately.

Option D — Register active profiler globally: Have `profile_pipeline` set a module-level variable that `timed_stage` reads; if set, write to the profiler. Decoupled but uses global state.

**Recommended approach (Option B):** The combined benchmark command owns the profiler. Add a `profiler: PipelineProfiler | None` field to `PipelineContext`. When not None, `timed_stage` is replaced with `profiler.stage()` at call sites. This is clean and testable. Alternatively, since the combined benchmark owns the run, it can just parse the log lines emitted by `timed_stage` — simpler for a single-frame timing use case.

### Finding 3: Preprocessing Output FPS Bug

In `preprocess.py` line 79:
```python
output_fps = fps / framestep
```

When `fps` comes from `cv2.CAP_PROP_FPS` on certain H.264 files, it can return `0.0` (codec doesn't report FPS reliably). The fix is to add an `--output-fps` CLI flag with default 30, pass it through to `process_video_temporal_median`, and use it directly instead of computing from source FPS. The existing parameter name `output_fps` is already used internally — just need to expose it as a CLI arg and stop computing it from source FPS.

### Finding 4: NAL Unit Error

From the error log:
```
[h264 @ 000001eb9c426740] Invalid NAL unit size (395781 > 65136).
[h264 @ 000001eb9c426740] Error splitting the input into NAL units.
```

This is an ffmpeg/libav message emitted by `cv2.VideoCapture` (which uses ffmpeg under the hood). It appears at the **end** of video decoding and the pipeline continues successfully. This is a known benign issue with some H.264 streams where the last NAL unit has an invalid size marker. It does not indicate corrupted frames — the output is verified good. Mitigation: suppress ffmpeg stderr by wrapping in a subprocess, or document it. Direct suppression of OpenCV's ffmpeg log is possible via `cv2.setLogLevel(0)` or environment variable `OPENCV_VIDEOIO_PRIORITY_FFMPEG=0`, but these affect all CV logging. **Recommended:** Document as known benign warning in a code comment near the `VideoCapture` open call or in user docs.

### Finding 5: Quality Presets — Current Behavior

`config.py` lines 408-413 show `auto_apply_preset` as a Pydantic `@model_validator(mode="after")`. This means when a YAML config has `quality_preset: fast`, loading it will silently override whatever values the user set in the YAML for affected fields. This is the confusing behavior reported.

**Implementing init-time presets:**
- Add `--preset {fast,balanced,quality}` flag to `aquamvs init`
- In `init_config()`, if preset is specified, call `config.apply_preset(preset)` before `to_yaml()`
- The generated YAML will have the preset values baked in as explicit field values, and `quality_preset: null`
- Remove (or make no-op) the `auto_apply_preset` validator so loading a YAML never silently overrides

### Finding 6: Deprecated Config Keys Warning

The warning `Unknown config keys in RuntimeConfig (ignored): ['save_depth_maps', 'save_point_cloud', 'save_mesh']` is triggered by `RuntimeConfig`'s `warn_extra_fields` model validator when old YAML configs are loaded.

These fields were removed when the 2-pass pipeline was introduced. The fix is:
1. Search for any remaining references to these field names in `src/aquamvs/`
2. If they appear in the `_migrate_legacy_config()` method, add entries there to strip them from old configs with an INFO log (not a WARNING)
3. Update the `RuntimeConfig` model validator to either map these to their new equivalents or silently ignore them with a migration message

**Current state:** `config.py`'s `_migrate_legacy_config()` (lines 489-588) does NOT handle `save_depth_maps`, `save_point_cloud`, `save_mesh`. They fall through as unknown extra keys.

### Finding 7: Sparse Mode — Summary Output by Design

Investigation result: sparse mode is **not a bug**. The `run_visualization_pass()` function calls `_collect_height_maps()` which looks for `depth_maps/` directories. In sparse mode, no depth maps are produced (the pipeline returns after `run_sparse_surface_stage()`). Therefore `height_maps` is empty and the summary gallery is skipped. This is correct behavior — there are no depth maps to put in a timeseries gallery. The issue can be closed as "by design."

### Finding 8: natsort Workaround

`.benchmarks/run_profile.py` and `.benchmarks/run_benchmark.py` are **standalone scripts** that avoid importing from the aquamvs package. The natsort issue report (6.1/1.1) says to "clean up hacky utility inlining in .benchmarks/". After inspection, neither script uses natsort — they don't import it or inline a replacement. The natsort workaround was likely in code that has already been revised. The `.benchmarks/` scripts themselves are using the old `torch.profiler` approach (now known to OOM) and will be replaced or deleted as part of the benchmark rebuild. **Action:** No separate cleanup plan needed; the scripts will be replaced or deleted by the benchmark rebuild plan.

### Finding 9: Benchmark Command — Current Broken State

The current `aquamvs benchmark` command in `cli.py` (lines 438-525):
- Takes a **BenchmarkConfig YAML** (not a PipelineConfig)
- Calls `run_benchmarks(config)` from `aquamvs.benchmark.runner`
- The runner (`benchmark/runner.py`) runs `_run_pipeline_config()` which calls `load_dataset()` which expects synthetic or ChArUco datasets

The new design (per CONTEXT.md) completely replaces this with:
- Takes a **PipelineConfig YAML** (same config as `aquamvs run`)
- `--frame 0` (default), `--extractors`, `--with-clahe` flags
- Runs all 4 pathways on the specified frame
- Reports per-stage wall time from profiler + relative accuracy metrics
- No datasets, no ground truth, no synthetic code

The entire `aquamvs/benchmark/` package (config.py, datasets.py, metrics.py, runner.py, comparison.py, visualization.py, report.py, synthetic.py, synthetic_benchmark.py) is built around the old BenchmarkConfig-driven model with synthetic/ChArUco datasets. All of this needs to be gutted and replaced.

---

## Architecture Patterns

### New `aquamvs benchmark` Command Design

```
aquamvs benchmark config.yaml [--frame 0] [--extractors sp,aliked,disk] [--with-clahe]
```

**Input:** PipelineConfig YAML (same as `aquamvs run`)
**Operation:** Runs 4 fixed pathways + optional extractor sweep on a single frame
**Output:** Console table + `{output_dir}/benchmark_YYYYMMDD_HHMMSS.md`

**4 Base Pathways:**
1. LightGlue + SuperPoint + sparse
2. LightGlue + SuperPoint + full
3. RoMa + sparse
4. RoMa + full

**With `--extractors sp,aliked,disk`:** Adds ALIKED and DISK variants for LightGlue paths (6 more rows)
**With `--with-clahe`:** Adds CLAHE-on variants for each LightGlue path

**Per-pathway metrics:**
- Wall time per stage: undistortion, matching, depth, fusion, surface (ms)
- Total wall time (ms)
- Point count (raw from fusion)
- Cloud density (points/m², requires computing scan area from calibration geometry)
- Outlier removal % (logged by fusion stage)

**Console table layout (rows × columns):**
```
| Pathway                  | Undist (ms) | Match (ms) | Depth (ms) | Fusion (ms) | Surface (ms) | Total (ms) | Points   | Density (pts/m²) | Outlier % |
| LG+SP sparse             | ...         | ...        | —          | —           | ...          | ...        | ...      | ...              | ...       |
| LG+SP full               | ...         | ...        | ...        | ...         | ...          | ...        | ...      | ...              | ...       |
| RoMa sparse              | ...         | ...        | —          | —           | ...          | ...        | ...      | ...              | ...       |
| RoMa full                | ...         | ...        | —          | ...         | ...          | ...        | ...      | ...              | ...       |
```

### Profiler Wiring — Recommended Approach

The cleanest approach that minimizes invasiveness: **thread-local profiler registry**.

```python
# profiling/profiler.py
import threading
_active_profiler: threading.local = threading.local()

def set_active_profiler(profiler: PipelineProfiler | None) -> None:
    _active_profiler.instance = profiler

def get_active_profiler() -> PipelineProfiler | None:
    return getattr(_active_profiler, 'instance', None)
```

`timed_stage` checks for an active profiler:
```python
@contextmanager
def timed_stage(name: str, logger: logging.Logger):
    profiler = get_active_profiler()
    if profiler is not None:
        with profiler.stage(name):
            yield
    else:
        # original behavior: log only
        start = time.perf_counter()
        yield
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info("%s: %.1f ms", name, elapsed_ms)
```

The benchmark command calls `set_active_profiler(profiler)` before running each pathway and `set_active_profiler(None)` after. This approach:
- Requires no changes to any pipeline stage files
- Is thread-safe (thread-local storage)
- Works for the single-threaded pipeline
- Does not affect normal `aquamvs run` usage

### Benchmark Module Structure (After Rebuild)

The `aquamvs/benchmark/` package should be rebuilt from scratch:

```
src/aquamvs/benchmark/
├── __init__.py          # Exports run_benchmark, BenchmarkResult, format_report
├── runner.py            # Core: runs each pathway, collects metrics
├── metrics.py           # Relative metrics: point count, density, outlier %
└── report.py            # Console table + markdown file formatting
```

Files to delete: `config.py`, `datasets.py`, `comparison.py`, `visualization.py`, `synthetic.py`, `synthetic_benchmark.py` — these are all part of the old BenchmarkConfig/dataset model.

Keep `metrics.py` as a starting point but strip the accuracy metric functions (which require ground truth). Keep only relative metric helpers.

### Preprocessing `--output-fps` Fix

The fix is minimal — add one parameter through the call stack:

1. `cli.py` `temporal-filter` subparser: add `--output-fps` arg (type=int, default=30)
2. `temporal_filter_command()`: pass `output_fps=args.output_fps` to `process_batch()`
3. `process_batch()`: add `output_fps: int = 30` parameter, pass to `process_video_temporal_median()`
4. `process_video_temporal_median()`: add `output_fps: int = 30` parameter, use it directly instead of computing `fps / framestep`

### Config Cleanup Pattern

For deprecated keys (`save_depth_maps`, `save_point_cloud`, `save_mesh`):

In `_migrate_legacy_config()`, add handling before the existing migrations:
```python
REMOVED_KEYS = {"save_depth_maps", "save_point_cloud", "save_mesh"}
for key in list(data.keys()):
    if key in REMOVED_KEYS:
        logger.warning(
            "Config key '%s' has been removed. "
            "Depth maps, point clouds, and meshes are always saved. "
            "Remove this key from your config.", key
        )
        del data[key]
```

For quality presets — move to init-time:

1. Add `--preset {fast,balanced,quality}` to `init_parser` in `cli.py`
2. In `init_config()`: if preset provided, call `config.apply_preset(preset)` before `to_yaml()`
3. In `config.py`: keep `apply_preset()` method, but remove (or gut) `auto_apply_preset` model validator so loading a YAML doesn't re-apply presets. The preset is already baked in.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ASCII table output | Custom string formatting | `tabulate` (already in deps) | Already used in benchmark runner |
| Markdown report | Custom template engine | f-strings with tabulate output | Simple enough for single-file |
| Performance timing | torch.profiler (OOM risk) | `time.perf_counter` + `torch.cuda.Event` | Already implemented in PipelineProfiler |
| Thread-local state | Global mutable dict | `threading.local()` | Standard Python pattern |
| Video FPS detection | Parse codec metadata | Don't — use explicit `--output-fps` arg | CV2 FPS reporting unreliable |

---

## Common Pitfalls

### Pitfall 1: Benchmark Rebuilds Assume Datasets
**What goes wrong:** The old BenchmarkConfig requires datasets (synthetic or ChArUco). Any code that loads `BenchmarkConfig` will fail if no datasets are configured. The new command takes a `PipelineConfig`, not `BenchmarkConfig`.
**Prevention:** Delete the old `BenchmarkConfig` and `benchmark_command` entirely. New command parses `PipelineConfig.from_yaml()`.

### Pitfall 2: Profiler Snapshots Empty When Using `timed_stage`
**What goes wrong:** Pipeline stages use `timed_stage()` which only logs; the profiler's `stage()` context manager is never called.
**Prevention:** Use the thread-local registry pattern (see Architecture Patterns). Do not pass profiler as function argument — that would require changing all stage signatures.

### Pitfall 3: STL Fix Already In Place
**What goes wrong:** Writing a plan to fix the STL export when it's already fixed.
**Prevention:** Verify current `surface.py` `export_mesh()` behavior before writing the plan. It already checks `has_vertex_normals()` and calls `compute_vertex_normals()`. Plan should just verify.

### Pitfall 4: natsort Cleanup Already Implicit
**What goes wrong:** Writing a separate cleanup plan for natsort when the `.benchmarks/` scripts are being replaced anyway.
**Prevention:** The natsort workaround (if it even exists in `.benchmarks/`) is inside scripts that will be replaced or deleted by the benchmark rebuild. No separate plan needed.

### Pitfall 5: Sparse Summary Is By Design
**What goes wrong:** Writing a fix for "no summary output in sparse mode" when it's intentional.
**Prevention:** Research confirms: sparse mode produces no depth maps, so the height field timeseries gallery has nothing to draw. This is correct behavior.

### Pitfall 6: `auto_apply_preset` Validator Side Effect
**What goes wrong:** Removing `auto_apply_preset` validator breaks configs that currently rely on it.
**Prevention:** The fix is to have the validator be a no-op (or emit a clear error/warning) rather than silently applying preset overrides. The generated YAML from `aquamvs init --preset X` will have explicit values, so the validator is not needed.

### Pitfall 7: Cloud Density Needs Scan Area Estimate
**What goes wrong:** Cloud density (points/m²) requires knowing the scanned area. Without ground truth, this must be estimated from the calibration geometry (e.g., project water surface bounds through camera FOV to get approximate area).
**Prevention:** Either compute approximate scan area from calibration (ring camera geometry, known water_z), or use a simpler proxy metric like "points per stereo pair" or skip density and just report raw point count. Simpler is better for a bug triage phase.

---

## Code Examples

### Thread-Local Profiler Registry Pattern
```python
# profiling/profiler.py — add after existing imports
import threading
_profiler_local = threading.local()

def set_active_profiler(profiler: "PipelineProfiler | None") -> None:
    """Set the active profiler for the current thread."""
    _profiler_local.instance = profiler

def get_active_profiler() -> "PipelineProfiler | None":
    """Get the active profiler for the current thread, or None."""
    return getattr(_profiler_local, 'instance', None)
```

### Updated `timed_stage` That Respects Active Profiler
```python
@contextmanager
def timed_stage(name: str, logger: logging.Logger):
    profiler = get_active_profiler()
    if profiler is not None:
        with profiler.stage(name):
            yield
    else:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info("%s: %.1f ms", name, elapsed_ms)
```

### Benchmark Runner Skeleton
```python
def run_benchmark(config_path: Path, frame: int, extractors: list[str], with_clahe: bool) -> BenchmarkResult:
    config = PipelineConfig.from_yaml(config_path)
    profiler = PipelineProfiler()
    results = []

    for pathway in build_pathways(config, extractors, with_clahe):
        pathway_config = apply_pathway(config, pathway)
        set_active_profiler(profiler)
        try:
            raw_images = read_single_frame(pathway_config, frame)
            ctx = build_pipeline_context(pathway_config)
            process_frame(frame, raw_images, ctx)
        finally:
            set_active_profiler(None)

        report = profiler.get_report()
        metrics = compute_relative_metrics(ctx, pathway_config)
        results.append(PathwayResult(pathway=pathway, timing=report, metrics=metrics))

    return BenchmarkResult(results=results)
```

### Preprocessing FPS Fix
```python
# preprocess.py — process_video_temporal_median signature change
def process_video_temporal_median(
    video_path: Path,
    output_dir: Path,
    window: int = 30,
    framestep: int = 1,
    output_format: str = "png",
    exact_seek: bool = False,
    window_step: int = 1,
    output_fps: int = 30,  # NEW: explicit output FPS for mp4
) -> int:
    ...
    if output_format == "mp4":
        # Use explicit output_fps instead of computing from source fps
        output_path = output_dir / f"{video_path.stem}_median.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
        logger.info(f"Writing video to: {output_path} @ {output_fps} fps")
```

### Deprecated Config Key Removal
```python
# config.py — in _migrate_legacy_config(), add at the top:
REMOVED_KEYS = {"save_depth_maps", "save_point_cloud", "save_mesh"}
for key in list(data.keys()):
    if key in REMOVED_KEYS:
        logger.warning(
            "Config key '%s' has been removed and will be ignored. "
            "All outputs are now always saved. Remove this key from your config.",
            key,
        )
        del data[key]
# Also check inside 'runtime' subdict:
if "runtime" in data and isinstance(data["runtime"], dict):
    for key in list(data["runtime"].keys()):
        if key in REMOVED_KEYS:
            logger.warning("Config key 'runtime.%s' has been removed.", key)
            del data["runtime"][key]
```

---

## Phase Plan Decomposition

Based on the research, the recommended plan structure is:

| Plan | Work | Size |
|------|------|------|
| 01: Sparse summary investigation | Confirm by-design, close issue, add code comment | Small |
| 02: Preprocessing fixes | `--output-fps` flag, NAL investigation, bottleneck profiling | Medium |
| 03: Config cleanup | Remove deprecated keys, move quality presets to init | Medium |
| 04: Benchmark/profiler rebuild | New `aquamvs benchmark` command replacing both old commands | Large |
| 05: Verify STL + cleanup .benchmarks | Verify STL fix, delete/replace .benchmarks scripts | Small |

Plan 01 can be done inline in under 30 minutes and gates nothing. Plans 02 and 03 are independent. Plan 04 is the main work and can be split into sub-plans if needed.

---

## Open Questions

1. **Cloud density metric feasibility**
   - What we know: Cloud density (pts/m²) requires scan area. We have calibration geometry (water_z, ring radius, camera positions).
   - What's unclear: Is computing approximate scan area from calibration too complex for a bug triage phase?
   - Recommendation: Use raw point count only. Cloud density can be added later. Outlier removal % and match count are simpler proxies.

2. **Delete vs. replace `.benchmarks/` scripts**
   - What we know: `.benchmarks/run_profile.py` uses old `torch.profiler` (OOM risk). `.benchmarks/run_benchmark.py` is a standalone synthetic benchmark.
   - What's unclear: Does the user want to keep any of these scripts, or delete them entirely?
   - Recommendation: Delete both. The new `aquamvs benchmark` command replaces the profiling functionality. The synthetic benchmark was explicitly deferred out of scope.

3. **`profile` CLI command removal**
   - What we know: The decision is to drop the `profile` command and fold it into `benchmark`.
   - What's unclear: Should the old `profile_command` function in `cli.py` be deleted or just un-registered from argparse?
   - Recommendation: Delete `profile_command` and `profile_pipeline` entirely. Remove `profiling/synthetic_profile.py`. Keep `PipelineProfiler`, `timed_stage`, and the thread-local registry — these are used by the new benchmark.

---

## Sources

### Primary (HIGH confidence — direct codebase inspection)
- `/src/aquamvs/profiling/profiler.py` — timed_stage vs profiler.stage disconnect confirmed
- `/src/aquamvs/profiling/analyzer.py` — ProfileReport build logic, confirmed snapshots empty = empty report
- `/src/aquamvs/pipeline/stages/*.py` — all 6 stages use timed_stage
- `/src/aquamvs/surface.py` lines 392-402 — STL fix confirmed already in place
- `/src/aquamvs/preprocess.py` lines 79-85 — FPS computed from source, 0.0 fps bug confirmed
- `/src/aquamvs/config.py` lines 489-588 — `_migrate_legacy_config` does not handle removed keys
- `/src/aquamvs/pipeline/runner.py` lines 114-124 — sparse mode early return, no depth maps
- `/src/aquamvs/pipeline/visualization.py` lines 39-53 — summary viz requires height maps from depth dir
- `/src/aquamvs/benchmark/runner.py` — confirms old BenchmarkConfig-driven architecture
- `/src/aquamvs/benchmark/config.py` — confirms BenchmarkConfig structure to be removed
- `/.benchmarks/run_profile.py` — confirms old torch.profiler approach, no natsort inlining
- `/.planning/qa/issues-found.md` — all 9 issues documented with reproduction context
- `open3d.io.write_triangle_mesh` verified via REPL — compute_vertex_normals() sufficient for STL

---

## Metadata

**Confidence breakdown:**
- Current state (bugs): HIGH — confirmed by direct source inspection
- Architecture patterns: HIGH — based on existing codebase patterns
- Profiler wiring fix: HIGH — thread-local pattern is standard Python
- Pitfalls: HIGH — derived from confirmed root causes

**Research date:** 2026-02-17
**Valid until:** 2026-03-17 (stable codebase, no external dependency changes expected)
