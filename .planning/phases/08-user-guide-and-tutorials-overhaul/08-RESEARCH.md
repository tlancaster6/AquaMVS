# Phase 8: User Guide and Tutorials Overhaul - Research

**Researched:** 2026-02-17
**Domain:** Sphinx documentation, Jupyter notebook rendering, ReadTheDocs deployment, CI smoke testing
**Confidence:** HIGH (core stack), MEDIUM (RTD config workaround)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- CLI guide (`docs/cli_guide.md`): polish in place — update for Phase 7 changes (benchmark rebuild, preset baking via `init --preset`, `--output-fps`, removed deprecated keys), fix formatting/tone
- API notebook (`docs/tutorial/notebook.ipynb`): polish in place — update for Phase 7 changes, ensure Colab plug-and-play experience
- Notebooks must auto-download and unpack the example dataset (GitHub Releases URL) in the first code cell — no manual download step
- All file paths in notebooks hardcoded to match the published example dataset structure (not configurable variables)
- Inline rendering on RTD — notebooks display as HTML pages with pre-executed static outputs
- Use nbsphinx or myst-nb (Claude's choice) + pandoc via RTD `build.apt_packages`
- Notebooks committed with cell outputs already executed (RTD does not re-execute)
- Each notebook page includes an "Open in Colab" launch button at the top
- Import-only smoke test in GitHub Actions — verify notebook code cells parse and imports resolve
- No full execution in CI (avoids needing example dataset and GPU in CI environment)
- Benchmarking notebook (new): full workflow — `aquamvs benchmark`, load results, compare LightGlue vs RoMa pathways, visualize metrics with torch.profiler stage timings
- Uses the example dataset (same auto-download as API tutorial), not synthetic data
- Visual comparisons: side-by-side rendered point clouds, depth maps, and/or metric bar charts
- Mixed audience: researchers reproducing results AND engineers integrating AquaMVS
- Professional/academic tone — clean, precise language fitting an academic library
- Troubleshooting centralized only — tutorials link to a central troubleshooting section

### Claude's Discretion

- Choice between nbsphinx and myst-nb for notebook rendering
- Overview page structure (standalone vs folded into index)
- Exact notebook cell organization and markdown structure
- How to implement the Colab launch button (badge image, nbsphinx config, or manual markdown)
- Specific matplotlib/Open3D visualization approach for benchmark comparisons

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

---

## Summary

Phase 8 replaces the thin placeholder tutorials section with 4 polished documentation pages: an overview page (routing users to CLI vs API paths), an updated CLI guide, a polished end-to-end Python API notebook, and a new benchmarking notebook. The existing `docs/cli_guide.md` and `docs/tutorial/notebook.ipynb` need updates for Phase 7 changes — the benchmark command was completely rebuilt, preset baking moved to `init --preset`, `--output-fps` was added to `temporal-filter`, and deprecated config keys were removed.

The core technical challenge is rendering Jupyter notebooks inline on ReadTheDocs with static (pre-executed) outputs. The project's `.readthedocs.yaml` currently uses `build.commands`, which creates a critical constraint: **`build.apt_packages` cannot be used simultaneously with `build.commands`** (RTD limitation, GitHub issue #9599). This eliminates nbsphinx (which requires pandoc) unless we switch to `build.jobs` or pip-install pandoc via `pypandoc-binary`. myst-nb does not require pandoc and is the better fit.

Both the API notebook and the benchmarking notebook must auto-download the example dataset from GitHub Releases in their first cell, include an "Open in Colab" badge at the top, and be committed with outputs pre-executed. The GitHub Actions smoke test verifies that notebook code cells parse syntactically and that imports resolve — without actually running the notebooks.

**Primary recommendation:** Use myst-nb (not nbsphinx) to avoid the pandoc/build.commands conflict. Set `nb_execution_mode = "off"` so RTD renders pre-executed outputs without re-running. Add myst-nb to `pyproject.toml [dev]` extras. Implement the Colab badge as a raw HTML cell at the top of each notebook.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| myst-nb | current (≥1.0) | Parse and render `.ipynb` files in Sphinx | No pandoc dependency; `nb_execution_mode = "off"` for static outputs; RTD recommended; ExecutableBooks org supported |
| myst-parser | already installed | Parse `.md` files in RST docs | Already in project (`docs/conf.py` uses it) |
| Sphinx + Furo | already installed | Documentation framework and theme | Already in project |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| matplotlib | already in deps | Visualizations in notebook cells | Bar charts for benchmark metrics, depth map colormaps |
| open3d | already in deps | Point cloud rendering for visual comparisons | Headless rendering via OffscreenRenderer |
| tabulate | already in deps | Structured table output | Displaying benchmark results in notebook |
| nbformat | installed with Jupyter | Reading/validating .ipynb files | Smoke test: extract code cells and ast.parse them |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| myst-nb | nbsphinx | nbsphinx requires pandoc — conflicts with `build.commands` in `.readthedocs.yaml` (cannot mix `build.apt_packages` with `build.commands`). Would require switching to `build.jobs` or pip-installing `pypandoc-binary`. myst-nb avoids this entirely. |
| myst-nb | jupyter-sphinx | jupyter-sphinx re-executes notebooks at build time — not appropriate for static pre-executed outputs |
| raw HTML Colab badge | nbsphinx_prolog | nbsphinx_prolog is nbsphinx-specific. For myst-nb, place a raw HTML cell at the top of each notebook as the first markdown cell. |

**Installation (additions to pyproject.toml `[dev]`):**
```toml
"myst-nb",
"nbformat",
```

Note: `myst-parser` is already listed. `myst-nb` supersedes it for notebook support but they can coexist; `myst_nb` extension replaces `myst_parser` in the extensions list.

---

## Architecture Patterns

### Recommended Documentation Structure

```
docs/
├── conf.py                      # Add myst_nb extension, nb_execution_mode = "off"
├── index.rst                    # Update toctree (add tutorial/overview, tutorial/api-notebook, tutorial/benchmark-notebook)
├── cli_guide.md                 # Polish in place — update for Phase 7
├── tutorial/
│   ├── index.rst                # Overview page (CLI vs API routing) — fold here OR make standalone
│   ├── notebook.ipynb           # API notebook — polish in place, add Colab badge + auto-download
│   └── benchmark.ipynb          # NEW — benchmarking notebook
└── troubleshooting.rst          # NEW — centralized troubleshooting (tutorials link here)
```

### Pattern 1: myst-nb Configuration

**What:** Add `myst_nb` to Sphinx extensions and set execution mode to off.

**conf.py changes:**
```python
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.mermaid",
    "myst_nb",           # replaces "myst_parser" for notebook support
]

# Use pre-executed outputs — RTD does not re-execute
nb_execution_mode = "off"

# Source suffixes: retain .rst and .md, add .ipynb
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-markdown",
    ".ipynb": "myst-nb",
}
```

**When to use:** Any time notebooks are committed with outputs and should render statically.

### Pattern 2: Colab Launch Button (Raw HTML Cell)

**What:** First cell of each notebook is a markdown cell with a raw HTML `<a href>` badge.

**Example first markdown cell in notebook.ipynb:**
```html
<a href="https://colab.research.google.com/github/tlancaster6/AquaMVS/blob/main/docs/tutorial/notebook.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# End-to-End Reconstruction Tutorial
...
```

The Colab URL format: `https://colab.research.google.com/github/{user}/{repo}/blob/{branch}/{path/to/notebook.ipynb}`

**When to use:** Top of every notebook. myst-nb renders raw HTML cells as HTML in the output — no special Sphinx config needed.

### Pattern 3: Auto-Download Example Dataset in First Code Cell

**What:** First code cell of each notebook downloads and unpacks the example dataset unconditionally if not already present.

**Example:**
```python
import os, urllib.request, zipfile
from pathlib import Path

DATASET_URL = "https://github.com/tlancaster6/AquaMVS/releases/download/v0.1.0-example-data/aquamvs-example-dataset.zip"
DATASET_DIR = Path("aquamvs-example-dataset")

if not DATASET_DIR.exists():
    print("Downloading example dataset...")
    urllib.request.urlretrieve(DATASET_URL, "aquamvs-example-dataset.zip")
    with zipfile.ZipFile("aquamvs-example-dataset.zip") as zf:
        zf.extractall()
    print("Done.")

# Hard-coded paths matching published dataset structure
CONFIG_PATH = DATASET_DIR / "config.yaml"
```

**When to use:** First code cell of every notebook. Colab users get zero-setup experience. Local users running from docs/ directory get auto-fetch on first run.

### Pattern 4: Import-Only Smoke Test

**What:** GitHub Actions job that reads each `.ipynb` file, extracts code cells, and calls `ast.parse()` on each cell's source. Validates syntax and that imports are resolvable without running the notebook.

**Example (tests/test_notebooks.py):**
```python
import ast
import json
from pathlib import Path
import pytest

NOTEBOOKS = list(Path("docs").rglob("*.ipynb"))

@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_cells_parse(nb_path):
    """Verify all code cells in notebook parse as valid Python."""
    with open(nb_path) as f:
        nb = json.load(f)
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"Cell {i} in {nb_path.name} has syntax error: {e}")
```

For a stronger check (verify imports resolve), run each notebook's import statements:
```python
import subprocess, sys
# Extract only 'import X' and 'from X import Y' lines, compile as a module, try exec
```

**When to use:** In the existing `docs.yml` CI workflow, or added to `test.yml` as a new step. No dataset or GPU needed.

### Pattern 5: toctree Integration for Notebooks

**What:** Add notebook files directly to toctree in `docs/tutorial/index.rst`.

```rst
Tutorials
=========

.. toctree::
   :maxdepth: 2

   notebook
   benchmark
```

myst-nb renders `.ipynb` files like any other Sphinx source file when added to the toctree. No special directive needed.

### Anti-Patterns to Avoid

- **Setting `nb_execution_mode = "auto"` or `"force"`:** RTD will attempt to re-execute notebooks at build time. This will fail because the example dataset is not present on RTD and Open3D OffscreenRenderer may not be available. Always use `"off"`.
- **Using nbsphinx with the current `.readthedocs.yaml`:** The `build.commands` key conflicts with `build.apt_packages`. pandoc cannot be installed via apt when using `build.commands`. Switching to `build.jobs` is possible but requires restructuring the RTD config significantly.
- **Using configurable path variables:** Decision is hard-coded paths. Do not make `DATASET_DIR` a variable the user can override — keep paths simple and matching the published zip structure.
- **Troubleshooting blocks inside tutorials:** Per the decision, all troubleshooting goes to a centralized section. Tutorials only link to it.
- **Running notebooks in CI:** The smoke test is import/parse-only. Full execution requires dataset + GPU and is explicitly out of scope.
- **Keeping `myst_parser` in extensions alongside `myst_nb`:** `myst_nb` includes all myst_parser functionality. Listing both causes a registration warning. Replace `myst_parser` with `myst_nb` in `conf.py`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Notebook → HTML rendering | Custom Sphinx extension | myst-nb | Handles cell inputs/outputs, code highlighting, widget support, error display |
| Colab launch button | Complex nbsphinx_prolog template | Raw HTML cell at top of notebook | Simpler, works with any renderer, directly visible in notebook |
| Smoke testing notebook syntax | Custom JSON parser | `json.load()` + `ast.parse()` | Standard library, zero extra dependencies |
| Dataset download in notebook | Custom download utility | `urllib.request.urlretrieve` + `zipfile` | Zero extra deps, works in Colab and locally |
| Benchmark visualization | Open3D interactive rendering | matplotlib bar charts + colormapped depth arrays | Works headless in notebook, renders to static output, no display required |

**Key insight:** myst-nb and raw HTML cells solve 90% of the notebook-in-docs problem. Avoid adding dependencies for problems the stdlib handles.

---

## Common Pitfalls

### Pitfall 1: build.commands Conflicts with build.apt_packages

**What goes wrong:** The project's `.readthedocs.yaml` uses `build.commands` for full control over the PyTorch CPU install and prereqs install. RTD explicitly does not allow `build.apt_packages` when `build.commands` is set (issue #9599).

**Why it happens:** nbsphinx requires pandoc (a system binary). pandoc is not installable via pip. The natural RTD solution (`build.apt_packages: [pandoc]`) is blocked by the existing `build.commands` config.

**How to avoid:** Use myst-nb instead of nbsphinx. myst-nb does NOT require pandoc — it uses docutils and MyST markdown directly. No system package install needed.

**Warning signs:** If you see nbsphinx in requirements and `build.commands` in `.readthedocs.yaml`, the build will fail to install pandoc.

**Confidence:** HIGH — verified against official RTD config docs and GitHub issue #9599.

### Pitfall 2: Notebook Re-execution on RTD Blows Up

**What goes wrong:** If `nb_execution_mode` is not explicitly set to `"off"`, myst-nb defaults to `"auto"` — which re-executes any notebook that has missing outputs. If the notebook's first code cell downloads a dataset (which takes minutes) or imports Open3D (which uses OffscreenRenderer on headless), RTD build times explode or crash.

**Why it happens:** myst-nb's default is to fill in missing outputs by re-executing.

**How to avoid:** Commit notebooks with ALL cell outputs populated. Set `nb_execution_mode = "off"` in `conf.py`. RTD then just renders the stored outputs as HTML.

**Warning signs:** RTD build times > 5 minutes for docs; "kernel died" errors in RTD build logs.

### Pitfall 3: Open3D OffscreenRenderer Availability in Notebooks

**What goes wrong:** The benchmarking notebook needs to show point cloud renderings as static images. `o3d.visualization.draw_geometries()` requires a display (X server) and will crash headless. The notebook is committed with pre-executed outputs, but if someone re-runs it locally without a display it fails.

**Why it happens:** Open3D's interactive visualizer is not headless by default.

**How to avoid:** For notebook visualizations, use Open3D's OffscreenRenderer (`o3d.visualization.rendering.OffscreenRenderer`) with fallback to matplotlib scatter plot of projected XYZ. Gate the rendering with a try/except:
```python
try:
    renderer = o3d.visualization.rendering.OffscreenRenderer(640, 480)
    # ... render and show as image
except Exception:
    # fallback: matplotlib 3D scatter
    import matplotlib.pyplot as plt
```

**Warning signs:** `ImportError: cannot import name 'OffscreenRenderer'`; segfaults on headless Linux.

**Confidence:** HIGH — documented in project MEMORY.md ("Open3D OffscreenRenderer may be unavailable on headless/CI — check at runtime, degrade gracefully").

### Pitfall 4: Colab Environment Missing AquaMVS

**What goes wrong:** Colab notebooks open without the AquaMVS package installed. All imports fail immediately.

**Why it happens:** Colab only has standard scientific Python pre-installed.

**How to avoid:** Second cell (right after the Colab badge markdown) must be a code cell that installs AquaMVS and prerequisites:
```python
# Install AquaMVS and prerequisites (run this cell first in Colab)
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision",
                "--index-url", "https://download.pytorch.org/whl/cpu", "-q"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install",
                "git+https://github.com/cvg/LightGlue.git@edb2b83",
                "git+https://github.com/tlancaster6/RoMaV2.git",
                "aquamvs", "-q"], check=True)
```

**Warning signs:** `ModuleNotFoundError: No module named 'aquamvs'` in Colab environment.

### Pitfall 5: Notebook File Paths Break Between Local and Colab

**What goes wrong:** Notebook hardcodes a path like `"./aquamvs-example-dataset"` which works locally but may differ in Colab's working directory (`/content/`).

**Why it happens:** Colab's working directory is `/content/` by default.

**How to avoid:** The dataset download cell should `os.chdir()` or use absolute paths derived from `Path.cwd()`. Simplest: let the download always run into the current working directory. The dataset zip unpacks to `aquamvs-example-dataset/` relative to wherever the cell runs. Use `Path.cwd() / "aquamvs-example-dataset"` — this resolves to the correct location in both environments.

**Confidence:** MEDIUM — based on known Colab behavior, not verified against live Colab run.

### Pitfall 6: Stale CLI Guide Causes User Confusion

**What goes wrong:** `docs/cli_guide.md` still documents the old `aquamvs profile` command (removed in Phase 7), the old benchmark config YAML format (removed), and quality presets as a runtime YAML key rather than an `init --preset` flag.

**Why it happens:** The CLI guide was written before Phase 7 changes and hasn't been updated.

**How to avoid:** Systematically audit the guide against the current `cli.py`. Key Phase 7 changes to cover:
- `aquamvs profile` command REMOVED — replaced by `--frame` flag on `aquamvs benchmark`
- `aquamvs benchmark` now compares 4 pathways (LG+SP sparse, LG+SP full, RoMa sparse, RoMa full) — no benchmark YAML config
- `--extractors` and `--with-clahe` flags on benchmark
- `aquamvs init --preset fast|balanced|quality` bakes values into config (removes quality_preset runtime key)
- `aquamvs temporal-filter --output-fps N` is new
- Deprecated config keys `save_depth_maps`, `save_point_cloud`, `save_mesh` are REMOVED

**Warning signs:** CLI guide mentions `aquamvs profile`, benchmark YAML configs, or `quality_preset:` as a YAML key.

---

## Code Examples

Verified patterns from official sources and codebase inspection:

### myst-nb conf.py Configuration

```python
# Source: myst-nb docs + codebase conf.py inspection
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.mermaid",
    "myst_nb",           # replaces myst_parser; includes myst_parser functionality
]

nb_execution_mode = "off"  # render pre-executed outputs, no kernel needed on RTD

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-markdown",
    ".ipynb": "myst-nb",
}
```

### Colab Badge as First Markdown Cell (notebook JSON snippet)

```json
{
  "cell_type": "markdown",
  "source": [
    "<a href=\"https://colab.research.google.com/github/tlancaster6/AquaMVS/blob/main/docs/tutorial/notebook.ipynb\">\n",
    "  <img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/>\n",
    "</a>\n",
    "\n",
    "# End-to-End Reconstruction Tutorial\n"
  ]
}
```

The Colab URL pattern is: `https://colab.research.google.com/github/{user}/{repo}/blob/{branch}/{path}`

For the API notebook: `.../blob/main/docs/tutorial/notebook.ipynb`
For the benchmark notebook: `.../blob/main/docs/tutorial/benchmark.ipynb`

### Benchmark API Notebook — Key Code Pattern

The benchmark notebook calls the Python API directly rather than shelling out to the CLI:

```python
from aquamvs.benchmark import run_benchmark, BenchmarkResult
from aquamvs.config import PipelineConfig

config_path = DATASET_DIR / "config.yaml"

# Run benchmark — returns BenchmarkResult with per-pathway ProfileReport
result = run_benchmark(
    config_path=config_path,
    frame=0,
    extractors=None,    # default: all 4 pathways
    with_clahe=False,
)

# Access results
for pw in result.results:
    print(f"{pw.pathway_name}: {pw.timing.total_time_ms/1000:.1f}s, {pw.point_count} points")
```

For profiler stage timing (per pathway):
```python
for pw in result.results:
    for stage_name, stage in pw.timing.stages.items():
        print(f"  {stage_name}: {stage.wall_time_ms:.0f} ms")
```

### Visualizing Benchmark Results with matplotlib

```python
import matplotlib.pyplot as plt
import numpy as np

pathways = [pw.pathway_name for pw in result.results]
total_times = [pw.timing.total_time_ms / 1000.0 for pw in result.results]
point_counts = [pw.point_count for pw in result.results]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(pathways, total_times, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
axes[0].set_title("Total Runtime by Pathway")
axes[0].set_ylabel("Time (s)")
axes[0].tick_params(axis="x", rotation=30)

axes[1].bar(pathways, point_counts, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
axes[1].set_title("Fused Point Count by Pathway")
axes[1].set_ylabel("Points")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.show()
```

### Stage Timing Breakdown (Stacked Bar Chart)

```python
stages = ["undistortion", "sparse_matching", "depth_estimation", "fusion", "surface"]
stage_labels = ["Undist", "Match", "Depth", "Fusion", "Surface"]
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

fig, ax = plt.subplots(figsize=(10, 5))
bottom = np.zeros(len(result.results))

for stage, label, color in zip(stages, stage_labels, colors):
    vals = []
    for pw in result.results:
        s = pw.timing.stages.get(stage)
        vals.append(s.wall_time_ms / 1000.0 if s else 0.0)
    ax.bar(pathways, vals, bottom=bottom, label=label, color=color)
    bottom += np.array(vals)

ax.set_title("Stage Timing Breakdown by Pathway")
ax.set_ylabel("Time (s)")
ax.legend()
plt.tight_layout()
plt.show()
```

### Smoke Test (GitHub Actions)

```python
# tests/test_notebook_smoke.py
import ast
import json
from pathlib import Path
import pytest

NOTEBOOK_PATHS = sorted(Path("docs").rglob("*.ipynb"))

@pytest.mark.parametrize("nb_path", NOTEBOOK_PATHS, ids=lambda p: p.stem)
def test_notebook_syntax(nb_path):
    """All notebook code cells must parse as valid Python."""
    with open(nb_path) as f:
        nb = json.load(f)
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if not source.strip():
            continue
        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"{nb_path.name} cell {i}: {e}")
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Download notebook link (`:download:` directive) | Inline HTML rendering with myst-nb | This phase | Notebooks render as full HTML pages with outputs — no download needed |
| Manual dataset download step | Auto-download in first code cell | This phase | Zero-setup Colab experience |
| `aquamvs profile` + `aquamvs benchmark` (two separate commands) | Unified `aquamvs benchmark` with profiling built in | Phase 7 | CLI guide and benchmark notebook must reflect this |
| `quality_preset:` as runtime YAML key | `aquamvs init --preset fast` bakes values at init time | Phase 7 | CLI guide must document new workflow |
| `save_depth_maps`, `save_point_cloud`, `save_mesh` config keys | Removed — stripped with warning on load | Phase 7 | CLI guide must remove these from config examples |
| `--output-fps` absent from `temporal-filter` | `--output-fps N` flag added | Phase 7 | CLI guide must document this flag |

**Deprecated/outdated:**
- `aquamvs profile` command: Removed in Phase 7. The `docs/benchmarks.rst` page documents it — this page needs update or removal.
- `benchmarks.rst` (standalone page): Documents old benchmark YAML config format and `aquamvs profile` command — both gone. This page needs significant revision or removal/replacement in Phase 8.
- Quality preset as runtime YAML key (`quality_preset: "fast"`): Runtime override now only warns; users should use `init --preset fast` instead.
- Notebook download link: Current `docs/tutorial/index.rst` has `:download:` directive — replaced by inline rendering.

---

## Open Questions

1. **What to do with `docs/benchmarks.rst`**
   - What we know: Entire page documents the old Phase 5 benchmark system (YAML-based configs, `aquamvs profile`, synthetic scenes) — all removed in Phase 7.
   - What's unclear: Should it be deleted entirely, replaced by a link to the new benchmark notebook, or kept but rewritten?
   - Recommendation: Replace the standalone `benchmarks.rst` with content pointing to the benchmark notebook. Or delete it and fold benchmarking into the tutorial section (the notebook covers what the page used to cover). The page as-is would confuse users with non-existent commands.

2. **Overview page: standalone vs. folded into `tutorial/index.rst`**
   - What we know: Current `tutorial/index.rst` is thin (53 lines) and could absorb overview content. A standalone page would appear in the sidebar.
   - What's unclear: Whether separate sidebar entries improve or clutter navigation.
   - Recommendation: Fold into `tutorial/index.rst` — replace the current thin content with a proper overview (CLI vs API routing diagram or table, when to use each, quick start snippets). This avoids adding a sidebar entry for a short page.

3. **Troubleshooting page: new file vs. fold into CLI guide**
   - What we know: Decision says "centralized troubleshooting" with tutorials linking to it. Current `cli_guide.md` has a Troubleshooting section at the bottom.
   - What's unclear: Whether to extract this to a standalone `troubleshooting.rst` or expand the CLI guide's existing section.
   - Recommendation: Create `docs/troubleshooting.rst` as a standalone page referenced from all tutorials. Extract the existing `## Troubleshooting` section from `cli_guide.md` there, then expand with notebook-specific issues (Colab install, dataset download).

4. **Example dataset directory structure**
   - What we know: The zip is at `https://github.com/tlancaster6/AquaMVS/releases/download/v0.1.0-example-data/aquamvs-example-dataset.zip`. Current `notebook.ipynb` expects `example_data/config.yaml`.
   - What's unclear: The actual directory name inside the zip when extracted (could be `aquamvs-example-dataset/` or `example_data/` or flat).
   - Recommendation: Unzip the dataset locally before notebook editing to verify the exact internal structure, then hardcode paths accordingly. The notebook auto-download cell should use `zipfile.ZipFile.namelist()` to find the root directory dynamically if it's uncertain.

---

## Sources

### Primary (HIGH confidence)
- Codebase inspection: `src/aquamvs/cli.py`, `src/aquamvs/benchmark/runner.py`, `src/aquamvs/benchmark/report.py`, `src/aquamvs/profiling/analyzer.py` — verified Phase 7 API surface
- `.planning/phases/07-post-qa-bug-triage/07-VERIFICATION.md` — verified 14 Phase 7 truths
- `docs/conf.py` — current Sphinx configuration (Furo theme, myst_parser, extensions)
- `.readthedocs.yaml` — uses `build.commands` (critical constraint)
- RTD config reference: https://docs.readthedocs.com/platform/latest/config-file/v2.html — `build.apt_packages` incompatible with `build.commands`

### Secondary (MEDIUM confidence)
- https://docs.readthedocs.com/platform/latest/guides/jupyter.html — RTD recommends nbsphinx or myst-nb
- https://myst-nb.readthedocs.io/en/latest/computation/execute.html — `nb_execution_mode = "off"` for pre-executed notebooks
- GitHub issue #9599 (readthedocs/readthedocs.org) — `build.commands` cannot coexist with `build.apt_packages`
- nbsphinx.readthedocs.io — nbsphinx v0.9.8, pandoc is required prerequisite
- https://colab.research.google.com/ badge URL pattern — standard `colab.research.google.com/github/{user}/{repo}/blob/{branch}/{path}`

### Tertiary (LOW confidence)
- Colab working directory `/content/` behavior — standard Colab knowledge, not freshly verified
- Open3D OffscreenRenderer headless behavior — based on project MEMORY.md, not freshly tested

---

## Metadata

**Confidence breakdown:**
- Standard stack (myst-nb choice): HIGH — pandoc conflict is documented in RTD official docs and GitHub issue
- RTD `nb_execution_mode = "off"`: HIGH — from myst-nb official docs
- Architecture patterns (notebook structure, Colab badge): HIGH — verified against codebase + official sources
- Phase 7 changes to document: HIGH — verified against Phase 7 VERIFICATION.md + codebase
- Open questions (dataset zip structure): LOW — requires local verification

**Research date:** 2026-02-17
**Valid until:** 2026-03-17 (stable libraries; RTD config behavior unlikely to change)
