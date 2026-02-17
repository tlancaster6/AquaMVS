---
phase: 08-user-guide-and-tutorials-overhaul
plan: 03
subsystem: docs
tags: [jupyter, notebook, sphinx, benchmark, smoke-test, pytest]

# Dependency graph
requires:
  - phase: 08-01
    provides: End-to-end API tutorial notebook structure and Colab/dataset patterns
  - phase: 07-03
    provides: Unified benchmark runner (run_benchmark, BenchmarkResult, PathwayResult API)

provides:
  - Benchmarking tutorial notebook with full workflow (run, load results, visualize)
  - benchmarks.rst redirect page pointing to benchmark notebook
  - docs/index.rst toctree updated with troubleshooting entry
  - Notebook smoke test that validates syntax and imports for all docs notebooks

affects:
  - 08-04 (checkpoint plan that executes notebooks and commits real outputs)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "importlib.util.find_spec for Colab install guard (avoids ruff F401 unused-import)"
    - "ast.parse for notebook cell syntax validation without execution"
    - "strict=False on zip() for ruff B905 compliance"

key-files:
  created:
    - docs/tutorial/benchmark.ipynb
    - tests/test_notebook_smoke.py
  modified:
    - docs/benchmarks.rst
    - docs/index.rst

key-decisions:
  - "importlib.util.find_spec used instead of try/import for Colab install guard (ruff F401 compliance)"
  - "strict=False on zip() calls in notebook visualizations (ruff B905 compliance)"

patterns-established:
  - "Notebook install guard: importlib.util.find_spec check rather than try/except ImportError"
  - "Smoke test pattern: ast.parse for syntax + importlib.import_module for imports, no execution"

# Metrics
duration: 12min
completed: 2026-02-17
---

# Phase 8 Plan 03: Benchmarking Notebook and Smoke Tests Summary

**15-cell benchmarking tutorial notebook comparing all four LightGlue/RoMa pathways with Colab badge, bar-chart visualizations, and stage timing breakdown; plus pytest smoke tests validating syntax and imports for all docs notebooks.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-02-17T23:21:10Z
- **Completed:** 2026-02-17T23:33:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Created `docs/tutorial/benchmark.ipynb` covering the full benchmarking workflow: Colab install guard, auto-download example dataset, CLI invocation via subprocess, Python API (`run_benchmark`), runtime/point-count bar charts, stacked stage timing chart, depth map side-by-side comparison, pathway selection guidance table
- Replaced stale `docs/benchmarks.rst` (which documented removed commands like `aquamvs profile`) with a redirect page pointing to the benchmark notebook
- Added `troubleshooting` to the User Guide toctree in `docs/index.rst`
- Created `tests/test_notebook_smoke.py` with parametrized `test_notebook_syntax` and `test_notebook_imports` tests that discover all `docs/**/*.ipynb` notebooks and validate without executing them

## Task Commits

Each task was committed atomically:

1. **Task 1: Create benchmarking notebook and replace benchmarks.rst** - `78f97a6` (feat)
2. **Task 2: Add notebook smoke test** - `411b751` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `docs/tutorial/benchmark.ipynb` - 15-cell benchmarking tutorial with Colab badge, dataset auto-download, CLI and Python API usage, matplotlib visualizations for runtime/points/stage breakdown/depth-map comparison
- `docs/benchmarks.rst` - Replaced stale benchmark suite docs with redirect to benchmark notebook
- `docs/index.rst` - Added `troubleshooting` entry to User Guide toctree
- `tests/test_notebook_smoke.py` - Smoke tests: syntax validation via `ast.parse`, import validation via `importlib.import_module`, discovers all docs notebooks, no dataset or GPU required

## Decisions Made

- `importlib.util.find_spec("aquamvs") is None` used as install guard in Colab cell instead of `try: import aquamvs except ImportError:` — ruff F401 flags the try-except pattern as unused import since the module is never used beyond the check
- `strict=False` added to all `zip()` calls in visualization cells — ruff B905 requires explicit `strict=` parameter; `False` is correct since STAGE_ORDER and STAGE_COLORS may differ in length from pathway results

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed ruff lint failures blocking commit**
- **Found during:** Task 1 (initial commit attempt)
- **Issue:** Pre-commit ruff hook flagged: (a) F401 `import aquamvs` unused in try-except guard, (b) B905 two `zip()` calls without `strict=`, (c) ruff-format reformatted notebook
- **Fix:** Replaced `try: import aquamvs` with `importlib.util.find_spec("aquamvs") is None`; added `strict=False` to both zip() calls; rewrote notebook JSON with correct formatting
- **Files modified:** `docs/tutorial/benchmark.ipynb`
- **Verification:** Pre-commit passed on second commit attempt
- **Committed in:** `78f97a6` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — lint/correctness)
**Impact on plan:** Necessary fix for code style compliance. Colab install cell now uses idiomatic pattern. No scope creep.

## Issues Encountered

- Pre-commit ruff hooks are more strict on notebook cells than expected — `importlib.util.find_spec` is the correct pattern for availability checks (also matches existing `notebook.ipynb` which uses the same pattern). Note for future notebooks: use `find_spec` from the start.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Benchmark notebook is ready for human execution and output capture (Phase 8 Plan 04)
- Smoke tests are operational and will catch future notebook code rot in CI
- `docs/index.rst` toctree is complete with troubleshooting entry
- `docs/tutorial/index.rst` already references `benchmark` page (was pre-existing from prior plan)

## Self-Check: PASSED

- docs/tutorial/benchmark.ipynb: FOUND
- docs/benchmarks.rst: FOUND
- tests/test_notebook_smoke.py: FOUND
- docs/index.rst: FOUND
- 08-03-SUMMARY.md: FOUND
- Commit 78f97a6: FOUND
- Commit 411b751: FOUND

---
*Phase: 08-user-guide-and-tutorials-overhaul*
*Completed: 2026-02-17*
