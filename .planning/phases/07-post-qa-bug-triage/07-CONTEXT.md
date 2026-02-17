# Phase 7: Post-QA Bug Triage - Context

**Gathered:** 2026-02-17
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix bugs and issues discovered during Phase 6 CLI QA execution. 9 issues logged across profiler/benchmark, preprocessing, and minor cleanup. No new features — only repair what's broken and clean up what's messy.

</domain>

<decisions>
## Implementation Decisions

### Benchmark & Profiler Rebuild
- **Combine benchmark and profile into one command**: `aquamvs benchmark` does everything. Drop the separate `profile` command entirely.
- **Pathway comparison**: Runs all 4 execution paths (LG+SP sparse, LG+SP full, RoMa sparse, RoMa full) on a single frame
- **Extractor sweep**: Optional `--extractors superpoint,aliked,disk` to test multiple extractors within LightGlue paths
- **CLAHE toggle**: Optional `--with-clahe` to test with/without CLAHE preprocessing
- **Per-stage timing**: Each pathway reports wall time per stage (undistortion, matching, depth, fusion, surface)
- **Relative accuracy metrics**: Point count, cloud density (points/m²), outlier removal %. No absolute accuracy (no ground truth for real data).
- **Report output**: Console pretty-printed table + markdown file saved to output dir
- **Default frame**: `--frame` defaults to 0
- **Single frame only**: No multi-frame averaging — keep it simple
- **Real data only**: Remove broken synthetic scene code. Build benchmark architecture so synthetic data could plug in later, but don't implement synthetic support now.
- **Fix profiler wiring**: Stage timing hooks exist but aren't connected to the report table — wire them up as part of the combined benchmark

### Preprocessing Improvements
- **Output FPS**: Add `--output-fps` flag (default 30) to fix videos writing at 0.0 fps
- **NAL error**: Quick investigation (~15 min) to check if it correlates with corrupted frames. Document findings either way (likely benign ffmpeg warning).
- **Performance optimization**: Profile to identify bottleneck (video decoding vs median calculation), then apply targeted fix (bottleneck.median for computation, or hw-accelerated decode)

### Fix vs. Remove Decisions
- **STL export**: Fix it — compute normals before STL write (simple one-liner)
- **Quality presets**: Move to init-time. `aquamvs init --preset fast` generates config with preset values baked in. No runtime silent override.
- **Deprecated config keys**: Strip dead code. Remove all references to save_depth_maps, save_point_cloud, save_mesh. Users with old configs get a clear error.
- **natsort workaround**: Clean up — remove hacky utility inlining in .benchmarks/, use normal natsort import (now in AquaCal deps)

### Sparse Summary Output
- **Investigate first**: Have Claude check whether sparse mode is supposed to produce summary output. Fix only if it's a bug, not if it's by design.

### ImageDirectorySet
- **Minimal fix only**: Keep the read_frame method added during QA. Skip the shared base class refactor — it works as-is.

### Claude's Discretion
- Exact benchmark table layout and column ordering
- How to structure the markdown report file
- Whether to remove synthetic code in the benchmark plan or a cleanup plan
- Profiler wiring implementation details (how stage timings flow to report)

</decisions>

<specifics>
## Specific Ideas

- Benchmark console output should look like a comparison table with pathway rows × metric columns (see example in discussion)
- Benchmark should be architected with an eye toward plugging in synthetic data later (separation of data loading from metric computation)
- When removing synthetic code, do a clean removal rather than leaving stubs

</specifics>

<deferred>
## Deferred Ideas

- Synthetic scene benchmark support — rebuild from scratch if/when wanted, don't maintain broken Phase 5 code
- ImageDirectorySet / VideoSet shared base class refactor — AquaCal concern, not AquaMVS Phase 7
- Multi-frame benchmark averaging — could add `--frames 0-4` mode later if needed

</deferred>

---

*Phase: 07-post-qa-bug-triage*
*Context gathered: 2026-02-17*
