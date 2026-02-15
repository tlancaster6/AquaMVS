---
phase: quick-3
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - tests/test_pipeline.py
autonomous: true

must_haves:
  truths:
    - "All 6 failing pipeline tests pass"
    - "All previously passing tests remain passing"
  artifacts:
    - path: "tests/test_pipeline.py"
      provides: "Fixed test assertions and mock patch paths"
  key_links:
    - from: "tests/test_pipeline.py"
      to: "src/aquamvs/pipeline/runner.py"
      via: "mock patch paths match import locations"
      pattern: "patch.*aquamvs\\.pipeline\\.runner"
---

<objective>
Fix 6 failing tests in tests/test_pipeline.py caused by mock patch paths not matching
where functions are imported (a common Python mocking pitfall after refactoring).

Purpose: Restore CI green after the Phase 02 Pydantic config migration.
Output: All 42 pipeline tests pass.
</objective>

<execution_context>
@C:/Users/tucke/.claude/get-shit-done/workflows/execute-plan.md
@C:/Users/tucke/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@tests/test_pipeline.py
@src/aquamvs/pipeline/runner.py
@src/aquamvs/pipeline/helpers.py
@src/aquamvs/pipeline/stages/surface.py
@src/aquamvs/pipeline/builder.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Fix mock patch paths and device Mock in test_pipeline.py</name>
  <files>tests/test_pipeline.py</files>
  <action>
Fix 6 failing tests by correcting mock patch paths to match where functions are
imported (patch where used, not where defined):

**Fix 1: TestSummaryViz.test_summary_viz_in_run_pipeline (line ~912)**
Change patch path from `aquamvs.pipeline.helpers._collect_height_maps` to
`aquamvs.pipeline.runner._collect_height_maps`. The runner does
`from .helpers import _collect_height_maps`, so the name is bound in runner's
namespace.

**Fix 2: TestSummaryViz.test_summary_viz_not_called_when_disabled (line ~949)**
Same fix: change `aquamvs.pipeline.helpers._collect_height_maps` to
`aquamvs.pipeline.runner._collect_height_maps`.

**Fix 3: _mock_pipeline_stages helper (line ~434)**
Change the `_sparse_cloud_to_open3d` patch from
`aquamvs.pipeline.helpers._sparse_cloud_to_open3d` to
`aquamvs.pipeline.stages.surface._sparse_cloud_to_open3d`. The stages/surface.py
module does `from ..helpers import _sparse_cloud_to_open3d`, so the name is bound
in surface's namespace. This fixes:
- TestSparseMode.test_sparse_mode_calls_surface_reconstruction
- TestSparseMode.test_sparse_mode_saves_sparse_ply
- TestSparseMode.test_sparse_mode_scene_viz
- TestSparseMode.test_sparse_mode_rig_viz

**Fix 4: TestMaskIntegration.test_masks_loaded_in_setup (line ~1407)**
Change patch path from `aquamvs.masks.load_all_masks` to
`aquamvs.pipeline.builder.load_all_masks`. The builder does
`from ..masks import load_all_masks`, so the name is bound in builder's namespace.

Do NOT change any production code. Only test mock patch paths are updated.
  </action>
  <verify>
Run `python -m pytest tests/test_pipeline.py --tb=short` and confirm all 42 tests pass.
Then run `python -m pytest tests/ --tb=short -q` to confirm no regressions.
  </verify>
  <done>All 42 pipeline tests pass. Full test suite has no new failures.</done>
</task>

</tasks>

<verification>
- `python -m pytest tests/test_pipeline.py` -- 42 passed, 0 failed
- `python -m pytest tests/ -q --tb=line` -- no new failures vs baseline
</verification>

<success_criteria>
All 6 failing pipeline tests pass with corrected mock patch paths.
No other tests regress.
</success_criteria>

<output>
After completion, no SUMMARY needed for quick tasks -- just commit the fix.
</output>
