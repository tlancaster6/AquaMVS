---
phase: quick-7
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - docs/scripts/generate_benchmark_figures.py
  - docs/benchmarks.rst
  - docs/tutorial/index.rst
  - docs/conf.py
autonomous: true
requirements: [QUICK-7]

must_haves:
  truths:
    - "docs/tutorial/benchmark.ipynb and its checkpoint are deleted"
    - "docs/scripts/generate_benchmark_figures.py runs end-to-end producing 4 figure PNGs"
    - "docs/benchmarks.rst is a visual Sample Output showcase page (not a how-to-run page)"
    - "docs/tutorial/index.rst no longer references the deleted notebook"
    - "sphinx-build completes without errors"
  artifacts:
    - path: "docs/scripts/generate_benchmark_figures.py"
      provides: "Benchmark visualization generator script"
      min_lines: 150
    - path: "docs/benchmarks.rst"
      provides: "Updated benchmarks page with sample output figures"
    - path: "docs/tutorial/index.rst"
      provides: "Tutorial index without benchmark notebook reference"
  key_links:
    - from: "docs/scripts/generate_benchmark_figures.py"
      to: "aquamvs.benchmark.runner.run_benchmark"
      via: "Python import"
      pattern: "from aquamvs\\.benchmark\\.runner import run_benchmark"
    - from: "docs/benchmarks.rst"
      to: "docs/scripts/generate_benchmark_figures.py"
      via: "RST instructions referencing the script"
      pattern: "generate_benchmark_figures"
---

<objective>
Replace the Jupyter notebook benchmark tutorial with a standalone Python script that
generates publication-quality comparison figures for the 3 benchmark pathways
(ROMA+FULL, LIGHTGLUE+FULL, LIGHTGLUE+SPARSE).

Purpose: The notebook is fragile (Colab installs, dataset downloads, myst_nb execution
quirks) and hard to maintain. A standalone script is simpler to run, version-control,
and produces deterministic static assets for docs.

Output: docs/scripts/generate_benchmark_figures.py + updated docs pages.
</objective>

<execution_context>
@C:/Users/tucke/.claude/get-shit-done/workflows/execute-plan.md
@C:/Users/tucke/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@docs/tutorial/benchmark.ipynb
@docs/benchmarks.rst
@docs/tutorial/index.rst
@docs/index.rst
@docs/conf.py
@src/aquamvs/benchmark/runner.py

<interfaces>
From src/aquamvs/benchmark/runner.py:
```python
@dataclass
class PathwayResult:
    pathway_name: str
    timing: ProfileReport  # .stages: dict[str, StageProfile], .total_time_ms: float
    point_count: int = 0
    cloud_density: float = 0.0
    outlier_removal_pct: float = 0.0
    stages_run: list[str] = field(default_factory=list)

@dataclass
class BenchmarkResult:
    results: list[PathwayResult]
    config_path: str = ""
    output_dir: str = ""
    frame: int = 0

def run_benchmark(config_path: Path, frame: int = 0, ...) -> BenchmarkResult
```

From src/aquamvs/profiling/analyzer.py:
```python
@dataclass
class StageProfile:
    name: str
    wall_time_ms: float
    cuda_time_ms: float
    cpu_memory_peak_mb: float
    cuda_memory_peak_mb: float

@dataclass
class ProfileReport:
    stages: dict[str, StageProfile]
    total_time_ms: float
    total_memory_peak_mb: float
```

From src/aquamvs/benchmark/report.py:
```python
def format_markdown_report(result: BenchmarkResult) -> str
def save_markdown_report(result: BenchmarkResult, output_dir: Path) -> Path
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create benchmark figure generator script and delete notebook</name>
  <files>
    docs/scripts/generate_benchmark_figures.py
  </files>
  <action>
Create `docs/scripts/` directory and `docs/scripts/generate_benchmark_figures.py`.

The script must:

1. **CLI interface** using argparse:
   - `config_path` (positional): Path to pipeline config YAML
   - `--frame` (default 0): Frame index to benchmark
   - `--output-dir` (default `docs/_static/benchmark/`): Where to save figures
   - Script is user-invoked only, not CI.

2. **Run benchmark** by calling `run_benchmark(config_path, frame=frame)` which
   returns a `BenchmarkResult`. This already runs all 3 pathways and generates
   per-pathway viz PNGs in `{output_dir}/benchmark/{safe_name}/frame_NNNNNN/viz/`.

3. **Generate 4 comparison figures** (matplotlib, saved as PNG at 150 DPI):

   a. `runtime_and_points.png` — Two side-by-side bar charts:
      - Left: Total runtime per pathway (seconds). Bar labels with "Xs" format.
      - Right: Reconstructed point count per pathway. Bar labels with comma-separated integers.
      - Color palette: `["#e07b54", "#2e86ab", "#1a5276"]` (coral for RoMa, teal shades for LG+SP).
      - figsize=(12, 5).

   b. `stage_timing.png` — Stacked bar chart of per-stage wall times:
      - Stage order: undistortion, sparse_matching, dense_matching, depth_estimation, fusion, surface.
      - Stage colors: `["#4e8098", "#90c2e7", "#6baed6", "#e07b54", "#f4a261", "#a8dadc"]`.
      - Access stage times via `pw.timing.stages[key].wall_time_ms / 1000.0` (0.0 if stage missing).
      - figsize=(12, 6).

   c. `depth_comparison.png` — Side-by-side depth maps for LG+SP full and RoMa full:
      - Load from `{benchmark_output}/benchmark/{safe_name}/frame_{frame:06d}/depth_maps/{cam}.npz`.
      - Use first camera from `PipelineConfig.from_yaml(config_path).camera_input_map`.
      - `np.load(path)["depth"]` with viridis colormap, colorbar labeled "Depth (m)".
      - Show "not found" text if file missing. figsize=(14, 5).

   d. `reconstruction_comparison.png` — Side-by-side 3D renders from all 3 pathways:
      - Load pre-rendered `fused_oblique.png` from `{benchmark_output}/benchmark/{safe_name}/frame_{frame:06d}/viz/fused_oblique.png`.
      - Show pathway name as title, axis off. figsize=(18, 5).
      - Skip pathways where PNG not found.

4. **Print summary** at end: list of saved figure paths + markdown report path.

5. Also call `save_markdown_report(result, output_dir)` to save the text report alongside figures.

6. Use `plt.savefig(path, dpi=150, bbox_inches="tight")` and `plt.close()` for all figures (no plt.show()).

7. Add `if __name__ == "__main__":` entry point.

8. Module docstring: `"""Generate static benchmark comparison figures for AquaMVS documentation."""`

Then DELETE the notebook and its checkpoint:
- `docs/tutorial/benchmark.ipynb`
- `docs/tutorial/.ipynb_checkpoints/benchmark-checkpoint.ipynb` (if it exists)

Also remove `docs/tutorial/.ipynb_checkpoints/` directory if empty after deletion.
  </action>
  <verify>
    python -c "import ast; ast.parse(open('docs/scripts/generate_benchmark_figures.py').read()); print('syntax OK')"
    Confirm docs/tutorial/benchmark.ipynb no longer exists.
  </verify>
  <done>
    Script parses without errors, accepts config_path and --output-dir args, imports
    run_benchmark and save_markdown_report, generates 4 figure types. Notebook deleted.
  </done>
</task>

<task type="auto">
  <name>Task 2: Update docs pages to reference script instead of notebook</name>
  <files>
    docs/benchmarks.rst
    docs/tutorial/index.rst
    docs/conf.py
  </files>
  <action>
**docs/benchmarks.rst** — Rewrite as a "Sample Output" showcase page. The page title
should be "Sample Output" (not "Benchmark Results"). Structure:

1. Brief intro (2-3 sentences): AquaMVS supports 3 reconstruction pathways, here are
   representative results from the example dataset showing what each produces.

2. Section "3D Reconstruction Comparison" — the hero section, first thing users see:
   - `.. image::` for `_static/benchmark/reconstruction_comparison.png`
   - Brief caption: side-by-side point cloud renders from all 3 pathways on the same frame.

3. Section "Depth Map Comparison":
   - `.. image::` for `_static/benchmark/depth_comparison.png`
   - Caption: colormapped depth maps for the two dense pathways (LG+SP full, RoMa full).

4. Section "Performance":
   - `.. image::` for `_static/benchmark/runtime_and_points.png`
   - `.. image::` for `_static/benchmark/stage_timing.png`
   - Brief narrative: sparse is fastest but fewest points, RoMa full is slowest but
     densest, LG+SP full is the balanced middle ground.

5. Section "Choosing a Pathway" — the recommendation table from the notebook:
   (LG+SP sparse = fast preview, LG+SP full = general purpose, RoMa full = max quality).
   Config snippet showing `matcher_type` and `pipeline_mode` fields.

6. No "how to regenerate" instructions on the page itself. The script is a dev-only tool;
   a one-line comment at the top of benchmarks.rst noting figures are generated by
   `docs/scripts/generate_benchmark_figures.py` is sufficient for maintainers.

7. Cross-reference to CLI Guide and API docs at the bottom.

**docs/tutorial/index.rst** — Remove the benchmark notebook references:
- Remove the "Benchmarking" paragraph that links to `:doc:\`Benchmarking Tutorial <benchmark>\``.
- Remove `benchmark` from the toctree.
- Keep the rest (CLI, Python API sections) intact.

**docs/conf.py** — Add `"tutorial/.ipynb_checkpoints"` to `exclude_patterns` if not present
(prevents Sphinx from picking up stale checkpoint files). No other conf.py changes needed
since `docs/scripts/` is not in the Sphinx source tree.
  </action>
  <verify>
    python -m sphinx -b html docs docs/_build/html -W --keep-going 2>&1 | tail -20
    Verify no warnings about missing benchmark notebook or broken references.
    If sphinx-build is not available, verify with: grep -r "tutorial/benchmark" docs/ (should find nothing).
  </verify>
  <done>
    benchmarks.rst is a visual "Sample Output" showcase with image directives (no script instructions).
    tutorial/index.rst has no benchmark notebook references.
    Sphinx build passes without errors related to the deleted notebook.
  </done>
</task>

</tasks>

<verification>
1. `docs/tutorial/benchmark.ipynb` does not exist
2. `docs/scripts/generate_benchmark_figures.py` parses and has correct imports
3. `docs/benchmarks.rst` references the script and shows image directives
4. `docs/tutorial/index.rst` toctree does not include `benchmark`
5. `grep -r "tutorial/benchmark" docs/` returns no matches (except possibly _build cache)
6. Sphinx build completes (if available): `python -m sphinx -b html docs docs/_build/html -W`
</verification>

<success_criteria>
- Notebook and checkpoint deleted from docs/tutorial/
- Standalone script created with 4 figure generators using run_benchmark API
- Docs pages updated: benchmarks.rst has script instructions + image placeholders,
  tutorial index no longer references the notebook
- No broken cross-references in Sphinx build
</success_criteria>

<output>
After completion, create `.planning/quick/7-replace-benchmark-ipynb-with-docs-script/7-SUMMARY.md`
</output>
