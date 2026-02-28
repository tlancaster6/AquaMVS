---
phase: quick-08
plan: 1
subsystem: pipeline
tags: [roma, memory, gpu, torch, gc]

requires:
  - phase: 03-pipeline-refactoring
    provides: "Stage-based pipeline architecture with dense_matching stage"
provides:
  - "Incremental warp-to-depth processing (no all_warps accumulation)"
  - "RoMa model lifecycle management (delete before aggregation)"
  - "torch.no_grad wrapping on depth conversion math"
  - "Public upsample_depth_map and upsample_confidence_map APIs"
affects: [dense-matching, roma-depth, pipeline-runner]

tech-stack:
  added: []
  patterns: ["incremental processing to reduce peak memory", "explicit gc.collect + empty_cache at stage boundaries"]

key-files:
  created: []
  modified:
    - src/aquamvs/pipeline/stages/dense_matching.py
    - src/aquamvs/dense/roma_depth.py
    - src/aquamvs/dense/__init__.py
    - src/aquamvs/pipeline/runner.py

key-decisions:
  - "Inline matching+depth conversion in dense_matching.py rather than calling run_roma_all_pairs then roma_warps_to_depth_maps"
  - "Cross-module import of _run_roma (private) is acceptable within same package"

patterns-established:
  - "Incremental processing: match pair -> convert to depth -> discard warp, rather than accumulate all warps"
  - "Explicit model lifecycle: create -> use -> delete -> gc.collect -> empty_cache"

requirements-completed: [QT-08]

duration: 10min
completed: 2026-02-28
---

# Quick Task 8: Optimize RoMa Pipeline Memory Usage Summary

**Incremental warp-to-depth processing with explicit RoMa model lifecycle and GPU cache management**

## Performance

- **Duration:** 10 min
- **Started:** 2026-02-28T18:45:53Z
- **Completed:** 2026-02-28T18:55:44Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Eliminated accumulation of ~48 warp tensors in memory by processing each pair incrementally
- RoMa model (~1GB) is now deleted before depth aggregation phase begins
- Added torch.no_grad() wrapping on warp_to_pairwise_depth to prevent gradient tracking
- Added gc.collect + torch.cuda.empty_cache at strategic points (after model deletion, after sparse matching, between frames)

## Task Commits

Each task was committed atomically:

1. **Task 1: Restructure run_roma_full_path for incremental processing** - `a85a3d2` (feat)
2. **Task 2: Add inter-frame CUDA cache clearing** - `61ba123` (feat)

## Files Created/Modified
- `src/aquamvs/pipeline/stages/dense_matching.py` - Restructured run_roma_full_path to inline matching+depth conversion; added gc cleanup to sparse path
- `src/aquamvs/dense/roma_depth.py` - Wrapped warp_to_pairwise_depth in torch.no_grad(); renamed _upsample_* to public APIs
- `src/aquamvs/dense/__init__.py` - Exported upsample_depth_map and upsample_confidence_map
- `src/aquamvs/pipeline/runner.py` - Added torch.cuda.empty_cache() after each frame completes

## Decisions Made
- Inlined the matching and depth conversion loop in dense_matching.py rather than calling the batch APIs (run_roma_all_pairs + roma_warps_to_depth_maps). This avoids the all_warps dict accumulation pattern.
- Cross-module import of `_run_roma` from features.roma is a private import but consistent with existing pattern for `_triangulate_two_rays_batch`.
- Kept roma_warps_to_depth_maps intact for backward compatibility (sparse path and external callers may still use the batch API).

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Memory optimization is complete and all 608 tests pass
- No behavioral changes to pipeline output; only internal memory management improved

---
*Quick Task: 08-optimize-roma-pipeline-memory-usage*
*Completed: 2026-02-28*
