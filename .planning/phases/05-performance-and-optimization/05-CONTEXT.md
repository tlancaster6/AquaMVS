# Phase 5: Performance and Optimization - Context

**Gathered:** 2026-02-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Identify and optimize performance bottlenecks through measurement, with benchmarking infrastructure to track regressions. Includes: comprehensive benchmark suite (RoMa vs LightGlue accuracy, CLAHE preprocessing, surface reconstruction methods), runtime and memory profiling, at least one optimization targeting a measured bottleneck, and regression detection. Does NOT include new reconstruction algorithms, new matching backends, or new pipeline features.

</domain>

<decisions>
## Implementation Decisions

### Benchmark comparison scope
- **Dual ground truth strategy**: real ChArUco boards from AquaCal calibration data + synthetic scenes matching experimental geometry
- **ChArUco evaluation**: both plane-fitting (overall shape/scale accuracy) and point-level error at detected corner positions
- **Synthetic scenes**: flat plane at known depth (baseline) + undulating sand-like surface with known analytic form (depth variation recovery). Viewed through flat refractive interface matching 12-camera ring geometry. The refractive surface is always flat (air-water interface); the reconstruction target is the underwater surface (sand, tank floor, etc.)
- **Accuracy metrics**: completeness (surface coverage %) AND geometric error (mean/median distance to ground truth in mm). For synthetic data, report both raw coverage (any valid depth) and accurate coverage (within tolerance). For real data, gracefully skip tolerance-based completeness when dense ground truth unavailable
- **Frame count**: configurable — default to single frame, benchmark config can specify multiple frames for thorough runs with statistical reporting

### Benchmark tests
- **CLAHE benchmark**: CLAHE on vs off, tested across RoMa and LightGlue (with SuperPoint, ALIKED, DISK extractors), sparse mode
- **Execution mode benchmark**: all four modes — LightGlue+sparse, LightGlue+full, RoMa+sparse, RoMa+full. LightGlue uses one configurable matcher (user-selectable, sensible default). Combined or separate sparse/dense benchmarks depending on metric overlap
- **Surface reconstruction benchmark**: Poisson vs heightfield vs BPA
- **Test toggling**: each benchmark test enabled/disabled in the benchmark config YAML, not via CLI flags

### Benchmark CLI and config
- Single `aquamvs benchmark` command replaces existing benchmark command entirely (no backward compatibility concerns)
- Benchmark config YAML lives in the same directory as benchmark output
- Datasets referenced by direct paths in config (no registry)
- Each benchmark run produces a results directory with structured subfolders

### Profiling and optimization targets
- Optimize for both per-frame speed and throughput/scale — profile reveals priorities
- Both runtime AND memory profiling (GPU/CPU peak memory per stage)
- Plane sweep stereo identified as likely primary bottleneck
- Configurable quality presets (fast/balanced/quality) so users choose accuracy-speed tradeoff

### Benchmark suite design
- CI runs synthetic-only benchmarks, under 1 minute (subset of tests selected for speed)
- Real data benchmarks are local-only
- Results stored in separate `.benchmarks/` directory (gitignored), with committed summary
- Fixed percentage thresholds for regression detection (alert if metric degrades by more than X%)
- Built-in lightweight diff: `aquamvs benchmark --compare run1 run2` — summary table with absolute and percent deltas for key metrics

### Results reporting
- Terminal output: ASCII tables by default
- Optional `--visualize` flag generates comparison plots:
  - Error heatmaps (spatial error maps per config)
  - Grouped bar charts (accuracy, completeness, runtime across configs)
  - Depth map side-by-side comparisons
- Plots saved alongside results in structured subfolders
- Auto-generated Sphinx docs page from a published baseline run (intentionally updated, not automatic)

### Claude's Discretion
- Quality preset system design (global preset vs per-stage knobs with preset shortcuts)
- Specific regression threshold percentages
- Synthetic scene generation approach (rendering method, texture choices)
- CI benchmark subset selection (which tests fit in 1-minute budget)
- Compression algorithm for depth sampling in fast presets
- Subfolder structure within results directory
- Choice of profiling tools (cProfile, torch.profiler, etc.)

</decisions>

<specifics>
## Specific Ideas

- Plane sweep stereo is the perceived primary bottleneck — profile to confirm
- Synthetic scenes should match experimental geometry: 12-camera ring, ~0.635m radius, water_z ~0.978m, n_water = 1.333
- Benchmark comparison diff should be lightweight — just a summary table, not a full report
- CI benchmarks must stay under 1 minute total
- The existing `aquamvs benchmark` command should be fully replaced, subsuming useful code

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-performance-and-optimization*
*Context gathered: 2026-02-15*
