---
phase: 05-performance-and-optimization
plan: 03
subsystem: profiling
tags: [profiling, performance, benchmarking, instrumentation]
dependency_graph:
  requires: [torch.profiler]
  provides: [PipelineProfiler, ProfileReport, stage_instrumentation]
  affects: [pipeline_stages, plane_sweep]
tech_stack:
  added: [torch.profiler, tabulate]
  patterns: [context_managers, zero_overhead_instrumentation]
key_files:
  created:
    - src/aquamvs/profiling/__init__.py
    - src/aquamvs/profiling/profiler.py
    - src/aquamvs/profiling/analyzer.py
  modified:
    - src/aquamvs/pipeline/stages/depth_estimation.py
    - src/aquamvs/pipeline/stages/dense_matching.py
    - src/aquamvs/pipeline/stages/sparse_matching.py
    - src/aquamvs/pipeline/stages/fusion.py
    - src/aquamvs/pipeline/stages/surface.py
    - src/aquamvs/dense/plane_sweep.py
decisions:
  - Used torch.profiler instead of custom timing (native PyTorch integration)
  - record_function for zero-overhead instrumentation (no-op when profiler inactive)
  - Added torch.no_grad() to build_cost_volume (plane sweep doesn't need gradients)
  - Wrapped all stage functions and hot paths (build_cost_volume, grid_sample_warp, extract_depth)
  - tabulate optional dependency for formatted reports (graceful degradation if missing)
metrics:
  duration: 12min
  completed: 2026-02-15
  tasks: 2
  files: 9
---

# Phase 05 Plan 03: Profiling Infrastructure Summary

**torch.profiler-based profiling infrastructure with zero-overhead stage instrumentation for bottleneck identification**

## What Was Built

### Profiling Package (src/aquamvs/profiling/)

**PipelineProfiler** wraps torch.profiler with pipeline-specific features:
- CUDA warmup before profiling (eliminates cold-start overhead)
- Memory tracking enabled by default (GPU memory often the bottleneck)
- record_shapes=True to identify size-dependent bottlenecks
- Chrome trace export for visualization in chrome://tracing
- Context manager interface for clean usage

**ProfileReport** identifies top 3 bottlenecks with measurements:
- Per-stage timing: CPU time, CUDA time, self time
- Per-stage memory: CPU memory, CUDA memory peak
- Top bottlenecks ranked by total time (CPU + CUDA)
- Formatted ASCII table output via tabulate

**Analyzer** parses torch.profiler results:
- Extracts metrics from key_averages()
- Groups operations by stage label (record_function names)
- Identifies top 3 bottlenecks by total time
- Device detection (CPU vs CUDA)

### Pipeline Instrumentation

**All pipeline stages wrapped with record_function:**
- depth_estimation
- dense_matching (both RoMa full and sparse paths)
- sparse_matching
- fusion
- surface_reconstruction

**Plane sweep hot paths instrumented:**
- build_cost_volume (with torch.no_grad())
- grid_sample_warp (_warp_source_at_depth)
- extract_depth

**Zero overhead when profiler inactive:** record_function is a no-op when torch.profiler is not running.

## Key Decisions

1. **torch.profiler over custom timing**: Native PyTorch integration provides CUDA synchronization, memory tracking, and Chrome trace export automatically.

2. **record_function for instrumentation**: Zero overhead when profiler is not active. Allows targeted profiling without performance impact in production.

3. **torch.no_grad() in build_cost_volume**: Plane sweep stereo doesn't need gradients. Disabling gradient tracking reduces memory usage and improves performance.

4. **tabulate as optional dependency**: Report formatting degrades gracefully if tabulate is not installed (falls back to simple string formatting).

5. **profile_pipeline as stub**: Full integration with Pipeline class requires additional work. PipelineProfiler provides the context manager interface for manual use.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Linter interference] Pre-commit hooks removed unused imports**
- **Found during:** Task 2 (stage instrumentation)
- **Issue:** Adding `from torch.profiler import record_function` without immediate usage caused pre-commit to remove the import before the with-block was added
- **Fix:** Used Write tool to add both import and usage atomically, bypassing linter's intermediate state
- **Files modified:** depth_estimation.py, dense_matching.py, sparse_matching.py
- **Commit:** ea29155

None - plan executed as written with linter workaround.

## Verification

**Profiling module compiles:**
```bash
python -m py_compile src/aquamvs/profiling/*.py
# All profiling modules compile OK
```

**torch.profiler available:**
```bash
python -c "from torch.profiler import profile, ProfilerActivity; print('torch.profiler available')"
# torch.profiler available
```

**Stage instrumentation present:**
```bash
grep "with record_function" src/aquamvs/pipeline/stages/*.py
# fusion.py, surface.py (2 functions), depth_estimation.py, dense_matching.py (2 functions), sparse_matching.py
```

**Plane sweep instrumentation present:**
```bash
grep "with record_function" src/aquamvs/dense/plane_sweep.py
# build_cost_volume, grid_sample_warp, extract_depth
```

**torch.no_grad() added:**
```bash
grep "torch.no_grad()" src/aquamvs/dense/plane_sweep.py
# build_cost_volume function uses torch.no_grad()
```

## Next Steps

1. Integrate PipelineProfiler with Pipeline class (fill in profile_pipeline stub)
2. Add --profile CLI flag to aquamvs run for easy profiling
3. Run profiling on example dataset to identify actual bottlenecks (BEN-02 requirement)
4. Document profiling workflow in user guide
5. Add profiling results to benchmark comparison table

## Commits

- a7ce589: feat(05-03): create profiling infrastructure
- ea29155: feat(05-03): instrument pipeline stages with record_function labels

## Self-Check: PASSED

**Created files exist:**
- [x] src/aquamvs/profiling/__init__.py
- [x] src/aquamvs/profiling/profiler.py
- [x] src/aquamvs/profiling/analyzer.py

**Modified files instrumented:**
- [x] src/aquamvs/pipeline/stages/depth_estimation.py (record_function)
- [x] src/aquamvs/pipeline/stages/dense_matching.py (record_function in 2 functions)
- [x] src/aquamvs/pipeline/stages/sparse_matching.py (record_function)
- [x] src/aquamvs/pipeline/stages/fusion.py (record_function)
- [x] src/aquamvs/pipeline/stages/surface.py (record_function in 2 functions)
- [x] src/aquamvs/dense/plane_sweep.py (record_function + torch.no_grad())

**Commits exist:**
- [x] a7ce589 (profiling infrastructure)
- [x] ea29155 (stage instrumentation)
