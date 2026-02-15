# Phase 6: CLI QA Execution - Context

**Gathered:** 2026-02-15
**Status:** Ready for planning

<domain>
## Phase Boundary

End-to-end QA of all 7 CLI commands and the programmatic Pipeline API using real multi-camera capture data. Verify each command runs successfully on GPU, review output quality, and fix bugs found along the way. This is a hands-on, conversational QA pass — not automated test writing.

</domain>

<decisions>
## Implementation Decisions

### Scope of commands
- All 7 CLI commands in scope: init, run, export-refs, profile, benchmark, preprocess, export-mesh
- Programmatic API also tested: `Pipeline(config).run()` basic smoke test
- Pass criteria: command completes successfully AND output is visually/manually reviewed
- Help text spot-checked only if something looks off, not systematically

### Test data strategy
- Real multi-camera captures with AquaCal calibration data
- 13 cameras, 10-minute videos each
- `aquamvs preprocess` extracts 5 temporal median frames per camera (1 per 2 minutes) → 5 frame sets of 13 images
- Preprocessed image directories are primary input for `aquamvs run`
- Video input tested minimally: one lightglue sparse run to verify video path works
- QA execution order: preprocess → init → export-refs (user creates masks externally) → run (lightglue+full) → run (roma+full) → benchmark --compare → export-mesh → profile

### Issue handling
- Blockers and quick fixes: fix inline immediately, then continue
- Large non-blockers: log to QA markdown file in .planning/ for later
- Commits: one commit per completed CLI command QA
- Tracking: conversational, no formal pass/fail report

### Coverage depth
- Happy path focus — skip error handling / bad input testing
- GPU (CUDA) for all testing
- 2 main execution paths for `aquamvs run`: lightglue+full and roma+full
- All mesh export formats tested: OBJ, STL, GLTF + simplification
- Skip --quiet flag and progress bar UX testing

### Claude's Discretion
- Order of mesh format testing
- Whether to test mesh simplification as separate step or combined with format export
- Profiler test approach (synthetic vs pipeline data)

</decisions>

<specifics>
## Specific Ideas

- User will create ROI masks externally after export-refs, so there's a manual pause in the workflow between export-refs and run
- Benchmark --compare used to compare lightglue vs roma runs (not accuracy benchmark with ground truth)
- Profiler is treated as its own standalone test at the end

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 06-cli-qa-execution*
*Context gathered: 2026-02-15*
