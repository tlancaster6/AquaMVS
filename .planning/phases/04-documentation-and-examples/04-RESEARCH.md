# Phase 4: Documentation and Examples - Research

**Researched:** 2026-02-14
**Domain:** Technical documentation, Sphinx/ReadTheDocs, scientific Python tutorials, dataset distribution
**Confidence:** HIGH

## Summary

Phase 4 creates comprehensive user-facing documentation for AquaMVS through README enhancement, Jupyter tutorial, example dataset, and hosted API reference. Research confirms the user's decision to use Sphinx with Furo theme on ReadTheDocs is aligned with current Python scientific software best practices (2026). Key findings:

1. **Sphinx + Furo + ReadTheDocs** is the modern standard for Python scientific libraries (replacing sphinx_rtd_theme)
2. **Dual-format tutorials** (Jupyter notebook for API, CLI markdown guide) effectively serve different user learning styles
3. **Zenodo with GitHub integration** is the gold standard for citable scientific datasets (DOI generation, FAIR compliance)
4. **nbconvert** reliably converts notebooks to markdown with preservation of outputs and visualizations
5. **sphinxcontrib-mermaid** provides diagrams-as-code for theory documentation (geometry, pipeline flow)

The current docs skeleton exists (conf.py with sphinx_rtd_theme, minimal index.rst) but needs migration to Furo, complete content, and .readthedocs.yaml configuration.

**Primary recommendation:** Migrate existing Sphinx configuration to Furo theme, create .readthedocs.yaml v2 config, structure docs with quickstart/tutorial/theory/API hierarchy, use Zenodo-GitHub integration for example dataset with DOI.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**README and Quickstart:**
- Visual-first README: lead with a hero image/GIF showing reconstruction results, then explain what AquaMVS does
- Hero image generated from the example dataset as part of this phase (not pre-existing)
- Minimal quickstart in README: `pip install aquamvs` + short Python snippet, link to full docs for details
- Standard badge set at top: PyPI version, Python versions, CI status, license

**Tutorial Depth and Format:**
- One end-to-end tutorial covering the full reconstruction workflow
- Dual format: Jupyter notebook (Python API focus) + rendered markdown version (CLI focus)
- Jupyter notebook uses `Pipeline` class and programmatic API
- Markdown guide uses `aquamvs run` CLI workflow
- Show intermediate visualizations: sparse match overlays, depth maps, consistency maps, fused point cloud, and final mesh

**Example Dataset:**
- Real data subset from the 13-camera rig, 1 frame per camera (single frame-set demonstrates all key functionality)
- Distributed via separate download (Zenodo or GitHub Releases), not bundled in repo
- Includes pre-made calibration JSON for self-contained usage
- Mentions AquaCal as the calibration source with link, for users who want the full workflow

**API Reference and Hosting:**
- Sphinx with Furo theme (clean, modern, dark mode)
- API scope: public API + extension points (Pipeline class, config models, CLI, protocols like FrameSource and CalibrationProvider). Internal stage modules stay undocumented.
- Brief installation guide: covers basics for Windows/Linux, CPU/GPU, links to PyTorch's own install page for GPU setup
- Detailed theory/concepts section: full walkthrough of refractive multi-view stereo math (ray casting, Snell's law refraction, plane sweep, cost volume, fusion) with diagrams. Reference-quality.
- Hosted on ReadTheDocs

### Claude's Discretion

- Exact Sphinx configuration and navigation structure
- README section ordering beyond the hero image lead
- Tutorial cell/section granularity
- How to render the markdown version from the notebook (or write separately)
- Zenodo vs GitHub Releases for dataset hosting
- Diagram style and tools for theory section

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope

</user_constraints>

## Standard Stack

### Core Documentation Tools

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Sphinx | >=7.0 | Documentation generator | Universal Python documentation standard, powers ReadTheDocs |
| Furo | >=2024.01.29 | Sphinx theme | Modern replacement for sphinx_rtd_theme, dark mode, clean design |
| sphinx.ext.autodoc | Built-in | API reference extraction | Auto-generates docs from docstrings |
| sphinx.ext.napoleon | Built-in | Google/NumPy docstring parsing | Converts Google-style docstrings to reST |
| sphinx.ext.intersphinx | Built-in | Cross-project linking | Links to external docs (PyTorch, NumPy, Open3D) |
| sphinxcontrib-mermaid | >=0.9.2 | Diagrams as code | Flowcharts, sequence diagrams for theory section |

**Confidence:** HIGH — All from official Sphinx documentation ([sphinx-doc.org](https://www.sphinx-doc.org/)), Furo's official repo ([pradyunsg/furo](https://github.com/pradyunsg/furo)), and current scientific Python library practices (Open3D, Kornia confirmed using similar stacks).

### Tutorial and Notebook Tools

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Jupyter | >=1.0 | Interactive notebooks | Python API tutorial (programmatic workflow) |
| nbconvert | >=7.0 | Notebook to markdown | Generate CLI markdown guide from notebook |
| matplotlib | Already dep | Inline visualizations | Tutorial output cells (depth maps, matches) |
| tqdm | Already dep | Progress bars | Tutorial notebook execution feedback |

**Confidence:** HIGH — nbconvert is the official Jupyter conversion tool ([nbconvert.readthedocs.io](https://nbconvert.readthedocs.io/)), verified 7.17.0 release January 2026.

### Dataset Hosting

| Platform | Purpose | Why Chosen |
|----------|---------|------------|
| Zenodo | DOI-minted dataset archive | Citable, FAIR-compliant, 50GB limit, CERN-backed permanence |
| GitHub Releases | (Alternative) | Simpler workflow but no DOI, 2GB file limit |

**Recommendation:** **Zenodo with GitHub integration** for primary hosting. Zenodo generates DOIs for citability, supports FAIR principles, and integrates with GitHub releases for automated preservation ([zenodo.org](https://zenodo.org/), [docs](https://developers.zenodo.org/)).

**Confidence:** HIGH — Recommended by Harvard Medical School Data Management ([source](https://datamanagement.hms.harvard.edu/share-publish/data-repositories/zenodo)), University of Iowa Research Data Services, and verified as scientific software best practice.

### Badges and README Enhancement

| Service | Purpose | Format |
|---------|---------|--------|
| shields.io | Dynamic badges | PyPI version, Python support, CI status, license |
| GitHub Actions | CI status badges | Links to workflow runs |

**Example badges:**
```markdown
![PyPI Version](https://img.shields.io/pypi/v/aquamvs)
![Python Support](https://img.shields.io/pypi/pyversions/aquamvs)
![CI Status](https://github.com/tlancaster6/AquaMVS/workflows/test/badge.svg)
![License](https://img.shields.io/pypi/l/aquamvs)
```

**Confidence:** HIGH — shields.io is the standard badge service ([shields.io](https://shields.io/)), verified in README best practices guides.

### Installation

```bash
# Documentation build dependencies (add to [dev] extra)
pip install sphinx furo sphinxcontrib-mermaid

# Tutorial development
pip install jupyter nbconvert

# Already have: matplotlib, tqdm (for tutorial cells)
```

## Architecture Patterns

### Recommended Sphinx Documentation Structure

```
docs/
├── conf.py                      # Sphinx configuration (migrate to Furo)
├── index.rst                    # Landing page (toctree to all sections)
├── quickstart.rst               # Installation + minimal example
├── tutorial/
│   ├── index.rst                # Tutorial landing (links to both formats)
│   ├── notebook.rst             # Embedded notebook or link to Colab/nbviewer
│   └── cli_guide.md             # CLI-focused markdown (converted from notebook)
├── theory/
│   ├── overview.rst             # High-level pipeline explanation
│   ├── refractive_geometry.rst  # Snell's law, ray casting (mermaid diagrams)
│   ├── dense_stereo.rst         # Plane sweep, cost volume, depth extraction
│   └── fusion.rst               # Geometric filtering, point cloud fusion
├── api/
│   ├── index.rst                # API reference landing
│   ├── pipeline.rst             # Pipeline class, run_pipeline, setup_pipeline
│   ├── config.rst               # PipelineConfig and all config dataclasses
│   ├── projection.rst           # ProjectionModel protocol, RefractiveProjectionModel
│   ├── calibration.rst          # CalibrationData, load_calibration_data
│   ├── features.rst             # Feature extraction, matching (if public API)
│   ├── dense.rst                # Plane sweep, depth extraction (if public API)
│   ├── fusion.rst               # fuse_depth_maps, filter_depth_map
│   ├── surface.rst              # Mesh reconstruction functions
│   ├── evaluation.rst           # Metrics, alignment
│   └── cli.rst                  # CLI commands reference
└── _static/                     # Images, diagrams
    ├── hero_reconstruction.gif  # Hero image for README/docs landing
    ├── pipeline_flowchart.png   # Mermaid-generated pipeline diagram
    └── ...
```

**Pattern rationale:**
- **Quickstart separate from tutorial:** Quickstart = 2-minute success, tutorial = 20-minute deep dive
- **Theory before API:** Users understand concepts before diving into code
- **API organized by module:** Mirrors package structure from `src/aquamvs/__init__.py`

**Confidence:** HIGH — Matches Open3D ([open3d.org/docs](https://www.open3d.org/docs/)), Kornia ([kornia.readthedocs.io](https://kornia.readthedocs.io/)), verified as scientific library standard.

### Pattern 1: Dual-Format Tutorial (Notebook + Markdown)

**What:** Single source notebook converted to markdown for CLI users
**When to use:** Different audiences prefer different interfaces (API vs CLI)
**How:**

```bash
# 1. Write Jupyter notebook: docs/tutorial/tutorial.ipynb
#    - Uses Pipeline class, setup_pipeline(), process_frame()
#    - Includes inline visualizations (plt.imshow for depth maps)

# 2. Convert to markdown with outputs embedded
jupyter nbconvert --to markdown docs/tutorial/tutorial.ipynb \
    --output docs/tutorial/cli_guide.md

# 3. Manually edit CLI guide to replace Python API calls with CLI equivalents:
#    - `setup_pipeline(config)` → `aquamvs init ...`
#    - `run_pipeline(config)` → `aquamvs run config.yaml`

# 4. Keep visualizations from notebook as static images
```

**Alternative (if separate is clearer):** Write two independent guides. Notebook shows Python API, CLI guide is hand-written markdown showing shell commands.

**Source:** nbconvert official docs ([nbconvert usage](https://nbconvert.readthedocs.io/en/latest/usage.html))

**Confidence:** HIGH

### Pattern 2: Mermaid Diagrams for Theory Documentation

**What:** Diagrams-as-code for pipeline flowcharts and geometry
**When to use:** Visual explanation of algorithm steps, coordinate systems
**Example:**

```rst
.. mermaid::

   flowchart TD
       A[Raw Images] --> B[Undistortion]
       B --> C[Feature Extraction]
       C --> D[Sparse Matching]
       D --> E[Triangulation]
       E --> F[Depth Range Estimation]
       F --> G[Plane Sweep Stereo]
       G --> H[Depth Map Fusion]
       H --> I[Surface Reconstruction]
       I --> J[Mesh Output]
```

**Configuration in conf.py:**
```python
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.mermaid',
]
```

**Source:** sphinxcontrib-mermaid docs ([pypi.org/project/sphinxcontrib-mermaid](https://pypi.org/project/sphinxcontrib-mermaid/))

**Confidence:** HIGH — Used by Canonical docs ([example](https://canonical-starter-pack.readthedocs-hosted.com/dev/how-to/diagrams-as-code-mermaid/))

### Pattern 3: README Hero Image with Centered HTML

**What:** Centered hero image at top of README showing reconstruction result
**When to use:** First visual impression, showcases capability
**Implementation:**

```markdown
<div align="center">
  <img src="docs/_static/hero_reconstruction.gif" alt="AquaMVS Reconstruction" width="800">
</div>

# AquaMVS

Multi-view stereo reconstruction of underwater surfaces with refractive modeling.

![PyPI Version](https://img.shields.io/pypi/v/aquamvs)
![Python Support](https://img.shields.io/pypi/pyversions/aquamvs)
![CI Status](https://github.com/tlancaster6/AquaMVS/workflows/test/badge.svg)
![License](https://img.shields.io/pypi/l/aquamvs)
```

**Why HTML:** GitHub Markdown doesn't support centering with pure markdown syntax

**Source:** GitHub README best practices ([guide](https://gist.github.com/DavidWells/7d2e0e1bc78f4ac59a123ddf8b74932d))

**Confidence:** HIGH

### Pattern 4: Intersphinx for External Library References

**What:** Auto-linking to PyTorch, NumPy, Open3D docs
**Configuration in conf.py:**

```python
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'open3d': ('http://www.open3d.org/docs/release', None),
}
```

**Usage in docstrings:**
```python
def backproject_depth_map(
    depth: torch.Tensor,  # Auto-links to torch.Tensor docs
    model: ProjectionModel,
) -> o3d.geometry.PointCloud:  # Auto-links to Open3D PointCloud
    """Backproject depth map to 3D point cloud."""
```

**Source:** Sphinx intersphinx extension ([sphinx-doc.org](https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html))

**Confidence:** HIGH

### Anti-Patterns to Avoid

- **Bundling large datasets in repo:** Use external hosting (Zenodo/GitHub Releases), reference download link
- **Auto-generating markdown from notebook without manual review:** CLI guide needs different workflow (shell commands, not Python), manually adapt or write separately
- **Documenting internal stage modules:** Expose only public API (`__all__` exports), internal implementation details not in API reference
- **sphinx_rtd_theme in 2026:** Outdated, poor dark mode support — migrate to Furo
- **Missing .readthedocs.yaml:** ReadTheDocs v2 config is required for reproducible builds

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| API reference generation | Manual function documentation | sphinx.ext.autodoc | Autodoc extracts from docstrings, stays in sync with code |
| Docstring formatting | Custom parsers | sphinx.ext.napoleon | Handles Google/NumPy styles, widely adopted |
| Cross-project references | Manual URL links | sphinx.ext.intersphinx | Auto-resolves links, updates with upstream docs |
| Diagram generation | Static images | sphinxcontrib-mermaid | Version-controllable source, easy updates |
| Notebook conversion | Custom scripts | nbconvert | Official Jupyter tool, handles outputs/images |
| Dataset DOI generation | Manual Zenodo UI | Zenodo-GitHub integration | Automated on release, preserves GitHub metadata |
| README badges | Static images | shields.io | Dynamic, auto-updates from PyPI/CI |

**Key insight:** Documentation tooling is mature in Python ecosystem — leverage established tools rather than custom solutions. Sphinx ecosystem has extensions for nearly every need.

## Common Pitfalls

### Pitfall 1: Furo Dark Mode CSS Variable Conflicts

**What goes wrong:** Custom color overrides break dark mode or create unreadable text
**Why it happens:** Furo uses separate light_css_variables and dark_css_variables, overriding only one breaks the other
**How to avoid:** Always test both light and dark mode when customizing colors. Use Furo's built-in color variables rather than hardcoding.
**Warning signs:** Text disappears in dark mode, links unreadable in light mode

**Configuration pattern:**
```python
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#0066cc",
    },
    "dark_css_variables": {
        "color-brand-primary": "#66b3ff",
    },
}
```

**Source:** Furo customization docs ([pradyunsg.me/furo/customisation/colors](https://pradyunsg.me/furo/customisation/colors/))

**Confidence:** HIGH

### Pitfall 2: Missing .readthedocs.yaml Causes Build Failures

**What goes wrong:** ReadTheDocs builds with wrong Python version, missing dependencies, or outdated Sphinx
**Why it happens:** Without explicit config, ReadTheDocs uses defaults (may be Python 3.8, old Sphinx)
**How to avoid:** Create .readthedocs.yaml v2 specifying OS, Python version, and dependencies
**Warning signs:** Builds work locally but fail on ReadTheDocs, import errors for type hints (3.10+ syntax)

**Required .readthedocs.yaml:**
```yaml
version: 2

build:
  os: ubuntu-24.04
  tools:
    python: "3.12"

sphinx:
  configuration: docs/conf.py

python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - dev
```

**Source:** ReadTheDocs configuration reference ([docs.readthedocs.com/config-file/v2](https://docs.readthedocs.com/platform/stable/config-file/v2.html))

**Confidence:** HIGH

### Pitfall 3: Jupyter Notebook Output Size Bloat

**What goes wrong:** Notebooks with large image outputs bloat repository size, slow git operations
**Why it happens:** Matplotlib figures embedded as base64 PNG in notebook JSON
**How to avoid:**
1. Clear outputs before committing: `jupyter nbconvert --clear-output tutorial.ipynb --inplace`
2. Use git pre-commit hook to strip outputs
3. For tutorial, commit cleared notebook, render outputs on ReadTheDocs build

**Warning signs:** .ipynb files >1MB, git clone slow, notebook diffs unreadable

**Confidence:** MEDIUM — Common issue, multiple mitigation strategies exist

### Pitfall 4: Example Dataset Without Installation Instructions

**What goes wrong:** Users download dataset but don't know where to place it or how to configure paths
**Why it happens:** Assuming users understand calibration JSON structure and config YAML format
**How to avoid:** Include README.md in Zenodo archive with:
- Directory structure after extraction
- Minimal config.yaml template with correct paths
- One-line command to run reconstruction: `aquamvs run config.yaml`

**Warning signs:** GitHub issues asking "where do I put the calibration file?"

**Confidence:** HIGH — User experience observation from scientific software

### Pitfall 5: Theory Section Too Sparse or Too Dense

**What goes wrong:** Theory docs either assume too much knowledge (users confused) or over-explain basics (experts skip)
**Why it happens:** Hard to balance accessibility vs depth
**How to avoid:**
- Start with high-level overview (what problem pipeline solves)
- Link to external resources for background (multi-view stereo basics, Snell's law physics)
- Focus theory section on **AquaMVS-specific contributions** (refractive modeling, how it differs from standard MVS)
- Use diagrams liberally (coordinate system, ray paths through water surface)

**Warning signs:** Users ask "what is multi-view stereo?" (too sparse) or experts say "why so basic?" (too dense)

**Best practice:** **Reference-quality for refractive MVS specifics, links to textbooks for MVS/CV basics**

**Source:** Scientific documentation best practices ([docs.python-guide.org](https://docs.python-guide.org/writing/documentation/))

**Confidence:** MEDIUM — Subjective balance, iterate based on feedback

## Code Examples

Verified patterns from official sources:

### Sphinx conf.py with Furo Theme

```python
# docs/conf.py
project = "AquaMVS"
copyright = "2024, Tucker Lancaster"
author = "Tucker Lancaster"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.mermaid",
]

# Theme
html_theme = "furo"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#0066cc",
        "color-brand-content": "#0066cc",
    },
    "dark_css_variables": {
        "color-brand-primary": "#66b3ff",
        "color-brand-content": "#66b3ff",
    },
}

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "open3d": ("http://www.open3d.org/docs/release", None),
}

# Napoleon (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
autodoc_typehints = "description"
```

**Source:** Furo quickstart ([pradyunsg.me/furo](https://pradyunsg.me/furo/)), Sphinx intersphinx docs

### ReadTheDocs Configuration (.readthedocs.yaml)

```yaml
# .readthedocs.yaml
version: 2

build:
  os: ubuntu-24.04
  tools:
    python: "3.12"

sphinx:
  configuration: docs/conf.py
  fail_on_warning: true

python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - dev

formats:
  - pdf
  - epub
```

**Source:** ReadTheDocs Sphinx deployment guide ([docs.readthedocs.com/intro/sphinx](https://docs.readthedocs.com/platform/stable/intro/sphinx.html))

### Tutorial Jupyter Notebook Structure

```python
# Cell 1: Setup
import aquamvs
from pathlib import Path

config = aquamvs.PipelineConfig.from_yaml("config.yaml")

# Cell 2: Pipeline initialization
ctx = aquamvs.setup_pipeline(config)
print(f"Loaded {len(ctx.projection_models)} cameras")

# Cell 3: Process single frame
from aquacal.io.video import VideoSet

video_set = VideoSet.from_config(config.camera_video_map)
raw_images = video_set.read_frame(0)

aquamvs.process_frame(0, raw_images, ctx)

# Cell 4: Visualize depth map
import matplotlib.pyplot as plt
import numpy as np

depth_map = np.load("output_dir/frame_000000/depth_maps/e3v82e0.npz")["depth_map"]
plt.imshow(depth_map, cmap="turbo")
plt.colorbar(label="Depth (m)")
plt.title("Reference Camera Depth Map")
plt.show()

# Cell 5: Load and display mesh
import open3d as o3d

mesh = o3d.io.read_triangle_mesh("output_dir/frame_000000/mesh/surface.ply")
o3d.visualization.draw_geometries([mesh])
```

**Source:** Jupyter best practices ([coderpad.io/blog/mastering-jupyter-notebooks](https://coderpad.io/blog/data-science/mastering-jupyter-notebooks-best-practices-for-data-science/))

### Zenodo Dataset README Template

```markdown
# AquaMVS Example Dataset

**Version:** 1.0.0
**DOI:** 10.5281/zenodo.XXXXXXX
**License:** CC BY 4.0

## Contents

- `calibration.json` — AquaCal calibration for 13-camera rig
- `videos/` — 13 synchronized video files (1 frame each)
  - `e3v82e0.mp4`, `e3v82e1.mp4`, ..., `e3v82eb.mp4` (ring cameras)
  - `e3v8330.mp4` (center auxiliary camera)

## Quick Start

1. Download and extract this dataset
2. Install AquaMVS: `pip install aquamvs`
3. Generate config:
   ```bash
   aquamvs init \
       --video-dir videos/ \
       --pattern "^([a-z0-9]+)\.mp4$" \
       --calibration calibration.json \
       --output-dir output/ \
       config.yaml
   ```
4. Run reconstruction:
   ```bash
   aquamvs run config.yaml
   ```

## Expected Outputs

- `output/frame_000000/point_cloud/fused.ply` — Fused point cloud
- `output/frame_000000/mesh/surface.ply` — Reconstructed mesh

## Dataset Details

- **Frame count:** 1 (demonstrative single timestep)
- **Camera count:** 13 (12 ring + 1 center)
- **Resolution:** 1920x1080 (ring), 1920x1440 (center)
- **Water surface Z:** ~0.978 m
- **Target depth:** ~1.2 m (underwater surface)

## Citation

If you use this dataset, please cite:

```
@software{aquamvs_example_2024,
  author = {Lancaster, Tucker},
  title = {AquaMVS Example Dataset},
  year = {2024},
  doi = {10.5281/zenodo.XXXXXXX},
}
```

## Calibration Source

Calibration generated using [AquaCal](https://github.com/tlancaster6/AquaCal).
```

**Source:** Zenodo best practices ([datamanagement.hms.harvard.edu/zenodo](https://datamanagement.hms.harvard.edu/share-publish/data-repositories/zenodo))

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| sphinx_rtd_theme | Furo theme | 2021-2024 | Better dark mode, cleaner design, active development |
| Manual badge creation | shields.io dynamic badges | 2015+ | Auto-updates from PyPI/CI, no manual version bumps |
| GitHub releases only | Zenodo with GitHub integration | 2020+ | DOI generation, citability, FAIR compliance |
| NumPy-style docstrings | Google-style docstrings | 2018+ | More readable, better autodoc support |
| Separate install docs per OS | Link to PyTorch install page for GPU | 2022+ | PyTorch install complexity (CUDA versions), defer to upstream |

**Deprecated/outdated:**
- **sphinx_rtd_theme:** Still works but Furo is modern replacement (better dark mode, accessibility)
- **Manual notebook-to-markdown conversion scripts:** nbconvert is official standard
- **Embedding datasets in repo:** GitHub has 100MB file limit, poor git performance — use external hosting

**Confidence:** HIGH — Verified from official migration guides and current library practices

## Open Questions

### 1. **Zenodo vs GitHub Releases for dataset hosting**
   - What we know: Zenodo provides DOI (citability), 50GB limit, FAIR compliance. GitHub Releases simpler but 2GB file limit, no DOI.
   - What's unclear: Example dataset size (1 frame from 13 cameras + calibration). If <2GB, GitHub Releases viable. If 2-50GB, Zenodo required.
   - Recommendation: **Estimate dataset size first.** If <1GB, GitHub Releases acceptable. If >2GB or citability important, use Zenodo-GitHub integration (automated, best of both).

### 2. **Notebook conversion: automatic vs manual CLI guide**
   - What we know: nbconvert converts notebook to markdown with outputs. But CLI workflow differs from Python API (shell commands vs Python calls).
   - What's unclear: How much manual editing required to adapt notebook markdown to CLI guide?
   - Recommendation: **Write CLI guide separately** as markdown. Notebook shows Python API (Pipeline class), CLI guide shows shell commands (aquamvs run). Different audiences, clearer separation. Share visualizations (copy images from notebook outputs).

### 3. **Hero image format: GIF vs static PNG**
   - What we know: GIF shows animation (reconstruction over time), PNG shows single result.
   - What's unclear: User decision says "hero image/GIF" — preference for animation?
   - Recommendation: **Static PNG initially** (simpler to generate from first reconstruction). Add GIF later if showing time-series reconstruction (multiple frames). GIF creation: ffmpeg from sequence of mesh renders.

### 4. **API reference scope for internal modules**
   - What we know: User decision says "internal stage modules stay undocumented" — only public API.
   - What's unclear: Are `features/`, `dense/`, `fusion.py` public API? They're exported in `__init__.py` `__all__`.
   - Recommendation: **Document everything in `src/aquamvs/__init__.py` `__all__`** (public API). Exclude `pipeline/stages/` (internal orchestration). Check `__all__` for definitive public API list.

## Sources

### Primary (HIGH confidence)

**Sphinx and Furo:**
- [Sphinx documentation](https://www.sphinx-doc.org/en/master/) — Official reference
- [Furo theme](https://pradyunsg.me/furo/) — Official docs
- [Furo customization](https://pradyunsg.me/furo/customisation/) — Colors, dark mode
- [sphinx.ext.napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) — Google docstrings
- [sphinx.ext.intersphinx](https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html) — Cross-project links
- [sphinxcontrib-mermaid](https://pypi.org/project/sphinxcontrib-mermaid/) — Diagrams extension

**ReadTheDocs:**
- [ReadTheDocs Sphinx guide](https://docs.readthedocs.com/platform/stable/intro/sphinx.html) — Deployment
- [.readthedocs.yaml v2 reference](https://docs.readthedocs.com/platform/stable/config-file/v2.html) — Configuration

**Jupyter and nbconvert:**
- [nbconvert documentation](https://nbconvert.readthedocs.io/) — Official conversion tool
- [nbconvert usage guide](https://nbconvert.readthedocs.io/en/latest/usage.html) — Command-line reference

**Zenodo:**
- [Zenodo](https://zenodo.org/) — Dataset hosting platform
- [Zenodo developers](https://developers.zenodo.org/) — API and GitHub integration
- [Harvard Medical School Zenodo guide](https://datamanagement.hms.harvard.edu/share-publish/data-repositories/zenodo) — Best practices

**Badges:**
- [shields.io](https://shields.io/) — Badge generation service
- [shields.io PyPI badges](https://shields.io/badges/py-pi-version) — PyPI integration

### Secondary (MEDIUM confidence)

**Scientific Library Examples:**
- [Open3D documentation](https://www.open3d.org/docs/latest/) — Structure reference
- [Kornia documentation](https://kornia.readthedocs.io/) — PyTorch CV library example
- [Kornia tutorials](https://www.kornia.org/tutorials/) — Tutorial organization

**Documentation Best Practices:**
- [Real Python: Documenting Python Code](https://realpython.com/documenting-python-code/) — Complete guide
- [Python Documentation Guide](https://docs.python-guide.org/writing/documentation/) — Hitchhiker's Guide
- [DataCamp: Documenting Python Code](https://www.datacamp.com/tutorial/documenting-python-code) — Tutorial

**README Best Practices:**
- [GitHub image centering guide](https://gist.github.com/DavidWells/7d2e0e1bc78f4ac59a123ddf8b74932d) — HTML in Markdown
- [Awesome README](https://github.com/matiassingers/awesome-readme) — Curated examples

**Jupyter Best Practices:**
- [CoderPad: Mastering Jupyter Notebooks](https://coderpad.io/blog/data-science/mastering-jupyter-notebooks-best-practices-for-data-science/) — Best practices guide

### Tertiary (LOW confidence — flagged for validation)

None — all findings verified with official documentation or established library practices.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — Official Sphinx, Furo, ReadTheDocs, nbconvert documentation verified
- Architecture: HIGH — Cross-referenced with Open3D, Kornia, scientific Python standards
- Pitfalls: HIGH-MEDIUM — Common issues from ReadTheDocs troubleshooting, Furo GitHub issues
- Dataset hosting: HIGH — Zenodo official docs, university data management guides

**Research date:** 2026-02-14
**Valid until:** 60 days (stable ecosystem — Sphinx/ReadTheDocs rarely introduce breaking changes)
