---
phase: quick-08
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - src/aquamvs/features/roma.py
  - src/aquamvs/dense/roma_depth.py
  - src/aquamvs/pipeline/stages/dense_matching.py
autonomous: true
requirements: [QT-08]

must_haves:
  truths:
    - "RoMa model is deleted from memory before depth conversion begins"
    - "Warps are converted to pairwise depths incrementally, not accumulated in a dict"
    - "GPU cache is cleared at strategic points between heavy stages"
  artifacts:
    - path: "src/aquamvs/features/roma.py"
      provides: "Incremental warp processing via generator or callback"
    - path: "src/aquamvs/dense/roma_depth.py"
      provides: "torch.no_grad wrapping on depth conversion"
    - path: "src/aquamvs/pipeline/stages/dense_matching.py"
      provides: "Model lifecycle management and memory cleanup"
  key_links:
    - from: "src/aquamvs/pipeline/stages/dense_matching.py"
      to: "src/aquamvs/features/roma.py"
      via: "run_roma_all_pairs returns warps incrementally or dense_matching processes inline"
      pattern: "del matcher|gc\\.collect|empty_cache"
---

<objective>
Reduce peak memory usage of the RoMa pipeline mode by restructuring warp-to-depth
processing to be incremental and adding explicit memory lifecycle management.

Purpose: The RoMa+full path currently (1) accumulates ALL warp results across ~48
camera pairs in a dict before any depth conversion begins, (2) keeps the ~1GB RoMa
model loaded during the entire depth conversion phase, and (3) lacks torch.no_grad
wrapping on depth conversion math. These patterns cause unnecessary peak memory that
forces heavy virtual RAM usage on machines without sufficient physical RAM.

Output: Modified roma.py, roma_depth.py, and dense_matching.py with incremental
processing and explicit cleanup.
</objective>

<execution_context>
@C:/Users/tucke/.claude/get-shit-done/workflows/execute-plan.md
@C:/Users/tucke/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/aquamvs/features/roma.py
@src/aquamvs/dense/roma_depth.py
@src/aquamvs/pipeline/stages/dense_matching.py
@src/aquamvs/pipeline/runner.py
@src/aquamvs/config.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Restructure run_roma_full_path to process warps incrementally and manage model lifecycle</name>
  <files>
    src/aquamvs/pipeline/stages/dense_matching.py
    src/aquamvs/features/roma.py
    src/aquamvs/dense/roma_depth.py
  </files>
  <action>
The goal is to eliminate the two biggest memory waste patterns: (a) the all_warps dict
that holds ~48 warp tensors simultaneously, and (b) the RoMa model staying resident
during depth conversion.

**In `dense_matching.py` — `run_roma_full_path`:**

Restructure the function to inline the matching and depth conversion rather than calling
`run_roma_all_pairs` (which returns a complete dict) followed by `roma_warps_to_depth_maps`
(which consumes the complete dict). Instead:

1. Create the RoMa matcher at the start via `create_roma_matcher(device)`.
2. For each reference camera, for each source camera:
   - Call `_run_roma(img_ref, img_src, matcher)` to get the warp.
   - Immediately call `warp_to_pairwise_depth(...)` to convert to depth.
   - Append the pairwise depth tensor to a per-reference list.
   - Let the warp result go out of scope (no accumulation).
3. After ALL pairs are matched and converted, **delete the matcher** and run
   `gc.collect()` + `torch.cuda.empty_cache()` (if CUDA available).
4. For each reference camera, call `aggregate_pairwise_depths(...)` on its
   pairwise depth list, then upsample and apply masks (same logic currently in
   `roma_warps_to_depth_maps`).

Import `_run_roma` from `...features.roma` (it is module-private but this is a
cross-module private import within the same package, consistent with the existing
pattern for `_triangulate_two_rays_batch`). Add a comment noting this.

Import `warp_to_pairwise_depth`, `aggregate_pairwise_depths`, `_upsample_depth_map`,
`_upsample_confidence_map` from `...dense.roma_depth`. The underscore-prefixed functions
need to be renamed to remove the underscore (make them public) since they are now used
cross-module. Update `dense/__init__.py` to export the two new public names.

Log progress like the current code: "Matching pair N/M: ref -> src".

**In `roma_depth.py`:**

- Rename `_upsample_depth_map` to `upsample_depth_map` and `_upsample_confidence_map`
  to `upsample_confidence_map` (remove underscore prefix since they are now cross-module).
- Update all internal references in the file (including inside `roma_warps_to_depth_maps`).
- Wrap `warp_to_pairwise_depth` body in `torch.no_grad()` context manager. The function
  does ray casting and triangulation which should never track gradients.
- Keep `roma_warps_to_depth_maps` as-is for backward compat (the sparse path via
  `match_all_pairs_roma` and any external callers may still use the batch API).

**In `roma.py`:**

No changes needed. `_run_roma` stays as-is; we just import it from dense_matching.

**In `dense/__init__.py`:**

Add `upsample_depth_map` and `upsample_confidence_map` to exports and `__all__`.

**Memory lifecycle summary:**
- RoMa model: created -> used for all pairs -> deleted -> gc.collect + empty_cache
- Per-pair warp: created -> immediately converted to depth -> goes out of scope
- Pairwise depths: accumulated per-ref (small: ~512x512 float32 each) -> aggregated -> cleared
  </action>
  <verify>
    Run `pytest tests/test_dense/test_roma_depth.py -x -v` to confirm depth conversion
    logic is unchanged. Run `python -c "from aquamvs.dense import upsample_depth_map, upsample_confidence_map; print('OK')"` to verify new exports.
    Run `python -c "from aquamvs.pipeline.stages.dense_matching import run_roma_full_path; print('OK')"` to verify imports resolve.
  </verify>
  <done>
    RoMa+full path processes warps incrementally (no all_warps accumulation), deletes
    the model before depth aggregation, clears GPU cache, and wraps depth conversion
    in torch.no_grad(). All existing tests pass.
  </done>
</task>

<task type="auto">
  <name>Task 2: Add memory cleanup to run_roma_sparse_path and inter-frame boundaries</name>
  <files>
    src/aquamvs/pipeline/stages/dense_matching.py
    src/aquamvs/pipeline/runner.py
  </files>
  <action>
**In `dense_matching.py` — `run_roma_sparse_path`:**

The sparse path calls `match_all_pairs_roma` which internally creates and uses the
matcher. After `match_all_pairs_roma` returns, add explicit cleanup:
```python
import gc
# ... after all_matches = match_all_pairs_roma(...)
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

This ensures the matcher (which `match_all_pairs_roma` creates internally) gets collected
promptly rather than lingering until the next GC cycle.

**In `runner.py` — `process_frame`:**

After each frame's RoMa path completes (after `run_surface_stage` or `run_sparse_surface_stage`),
add cleanup before the next frame. Add at the end of the `if config.matcher_type == "roma":`
block (after both full and sparse sub-blocks), before the "Frame complete" log:

```python
# Free per-frame tensors before next frame
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

This is lightweight (no gc.collect needed here since local variables go out of scope
naturally at function return) but ensures CUDA allocator releases cached blocks between frames.

Add `import gc` at the top of `dense_matching.py` if not already present.
  </action>
  <verify>
    Run `pytest tests/ -x -q --ignore=tests/test_slow` to confirm no regressions.
    Verify `run_roma_sparse_path` and `process_frame` contain cleanup calls via grep.
  </verify>
  <done>
    Sparse RoMa path explicitly triggers garbage collection after matching.
    Inter-frame CUDA cache clearing prevents memory accumulation across frames.
    All tests pass.
  </done>
</task>

</tasks>

<verification>
- `pytest tests/test_dense/ -x -v` passes (depth conversion logic unchanged)
- `pytest tests/test_features/test_roma.py -x -v` passes (matching API unchanged)
- `python -c "from aquamvs.dense import upsample_depth_map, upsample_confidence_map"` succeeds
- No new lint errors: `ruff check src/aquamvs/features/roma.py src/aquamvs/dense/roma_depth.py src/aquamvs/pipeline/stages/dense_matching.py src/aquamvs/pipeline/runner.py`
</verification>

<success_criteria>
- Peak memory during RoMa+full path reduced: warps processed one-at-a-time instead of ~48 accumulated
- RoMa model (~1GB) freed before depth aggregation/fusion/surface stages
- torch.no_grad wraps all depth conversion math
- gc.collect + empty_cache called after model deletion and after sparse matching
- All existing tests pass without modification
</success_criteria>

<output>
After completion, create `.planning/quick/8-optimize-roma-pipeline-memory-usage-redu/8-SUMMARY.md`
</output>
