# Phase 3: Pipeline Decomposition and Modularization - Context

**Gathered:** 2026-02-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Refactor the monolithic 995-line `pipeline.py` into a modular `pipeline/` package with separate builder, runner, and stage modules. Each execution path (lightglue+sparse, lightglue+full, roma+sparse, roma+full) becomes a distinct stage module. AquaCal coupling is isolated behind interfaces. Feature extractors, depth estimators, and fusion are accessed via protocol interfaces.

</domain>

<decisions>
## Implementation Decisions

### Backward Compatibility
- **Clean break** — no deprecation shims on old imports
- Only current user is the author; no external consumers to protect
- Primary entry points are CLI (`aquamvs run`) and `run_pipeline()` function — these continue to work
- Update ROADMAP success criterion #5 to remove backward-compat requirement
- Any breakage in tests or internal imports is fixed directly during refactoring

### Public API Surface
- **Pipeline class** as primary programmatic entry point: `pipeline = Pipeline(config); pipeline.run()`
- Top-level re-export: `from aquamvs import Pipeline` works (canonical location: `from aquamvs.pipeline import Pipeline`)
- **Claude's discretion:** Whether to expose intermediate results (depth maps, point clouds) as attributes vs. final output only
- **Claude's discretion:** Whether individual stages (matching, depth estimation, fusion) are independently importable or internal-only

### Extension Points
- Matchers, depth estimation, and fusion are all **potentially swappable** (not urgent, but don't paint into a corner)
- Extension is an **advanced/Python API concern**, not a CLI concern
- **Claude's discretion:** Whether extension happens through Protocol/ABC or config-driven registration — pick what fits the codebase
- Document the process for substituting custom modules (note for Phase 4 documentation)

### AquaCal Isolation
- **FrameSource interface** — abstract frame reading; `VideoSet` and `ImageDirectorySet` are implementations
- **CalibrationProvider interface** — separate from frame reading; provides camera params AND refractive geometry
- CalibrationProvider includes refractive parameters (water_z, n_water, interface_normal)
- **Refraction-naive fallback:** If refractive parameters are missing, print descriptive warning and set n_air=n_water=1.0 (equivalent to non-refractive model downstream)
- Refraction-naive mode is minimally tested for now — add thorough testing as a TODO/backlog item

### Claude's Discretion
- Module boundary decisions (how to split stages into files)
- Protocol vs ABC vs duck typing for interfaces
- Intermediate result exposure on Pipeline class
- Stage independence (importable individually or internal-only)

</decisions>

<specifics>
## Specific Ideas

- Extension point documentation should explain how to substitute custom matchers/depth estimators — capture as a Phase 4 doc task
- Refraction-naive fallback enables future use with non-AquaCal calibration tools (COLMAP, OpenCV stereoCalibrate, etc.) without requiring refractive parameters

</specifics>

<deferred>
## Deferred Ideas

- Support for non-AquaCal calibration formats (COLMAP, Metashape) — future phase, enabled by CalibrationProvider interface
- Thorough testing of refraction-naive mode — backlog item after Phase 3
- Custom module documentation — Phase 4 (Documentation and Examples)

</deferred>

---

*Phase: 03-pipeline-decomposition-and-modularization*
*Context gathered: 2026-02-14*
