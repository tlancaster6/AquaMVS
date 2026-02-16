---
created: 2026-02-16T18:36:05.916Z
title: Fix benchmark package broken imports and stale test wiring
area: testing
files:
  - src/aquamvs/benchmark/__init__.py
  - src/aquamvs/benchmark/report.py
  - src/aquamvs/benchmark/visualization.py
  - src/aquamvs/benchmark/runner.py
  - src/aquamvs/benchmark/metrics.py
  - tests/benchmarks/test_ci_benchmarks.py
---

## Problem

The `aquamvs.benchmark` package has broken internal imports that prevent the
entire package from loading. This cascades to block `tests/benchmarks/test_ci_benchmarks.py`
from collecting.

**Root cause:** Phase 05 rewrote the benchmark source modules (runner.py,
visualization.py) with new function names and signatures, but left stale
references in `__init__.py` and `report.py` pointing at the old API.

Three categories of breakage:

1. **Source package broken imports** (`__init__.py` and `report.py`):
   - `render_comparison_grids` imported from `visualization.py` — doesn't exist
     (only `generate_visualizations` exists)
   - `render_config_outputs` imported from `visualization.py` — doesn't exist
   - `run_benchmark` (singular) imported from `runner.py` — doesn't exist
     (only `run_benchmarks` plural exists)
   - `report.py:14` imports `render_comparison_grids` from visualization,
     causing any `from aquamvs.benchmark import ...` to fail

2. **Legacy report.py**: Uses old `BenchmarkResults`/`ConfigResult` data
   structures and calls `render_comparison_grids`. May need rewrite or removal
   depending on whether the legacy feature-extraction benchmark report is
   still needed.

3. **test_ci_benchmarks.py aspirational imports**: Created in Phase 05-06
   against the *new* API, but references functions that earlier Phase 05 tasks
   never implemented:
   - `compute_completeness` — doesn't exist; nearest is `compute_accuracy_metrics`
     (takes Open3D mesh, not raw point arrays)
   - `compute_geometric_error` — doesn't exist; nearest is `compute_accuracy_metrics`
   - `extract_depth_map` from `dense.plane_sweep` — doesn't exist as standalone export

   The `tests/test_benchmark/` directory (old API tests) has already been deleted.

## Solution

1. **Fix `__init__.py`**: Remove exports for nonexistent symbols (`run_benchmark`,
   `render_comparison_grids`, `render_config_outputs`). Only export what exists.

2. **Fix `report.py`**: Either rewrite to use `generate_visualizations` from the
   new API, or remove the `render_comparison_grids` call and degrade gracefully
   (skip grid generation). Evaluate whether the legacy report generator is still
   useful or should be removed entirely.

3. **Fix `test_ci_benchmarks.py`**: Two options for each broken test:
   - `test_ci_metric_computation`: Add thin `compute_completeness` and
     `compute_geometric_error` wrappers to `metrics.py` that delegate to
     `compute_accuracy_metrics`, OR rewrite the test to call
     `compute_accuracy_metrics` directly.
   - `test_ci_depth_extraction_correctness`: Either add `extract_depth_map`
     as a public function in `dense.plane_sweep`, or rewrite the test to use
     whatever depth extraction API actually exists.
   - The two `pytest.skip` placeholder tests and the no-op runtime test can
     be left as-is or removed.

4. **Verify**: `pytest tests/benchmarks/ --collect-only` should succeed with
   no import errors. `from aquamvs.benchmark import *` should work.
