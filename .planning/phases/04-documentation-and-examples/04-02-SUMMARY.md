---
phase: 04-documentation-and-examples
plan: 02
subsystem: documentation
tags: [readme, api-docs, sphinx, autodoc]
dependency_graph:
  requires: []
  provides: [visual-readme, api-reference-pages]
  affects: [documentation-site]
tech_stack:
  added: []
  patterns: [autodoc, rst-markup, sphinx-alabaster-theme]
key_files:
  created:
    - docs/api/index.rst
    - docs/api/pipeline.rst
    - docs/api/config.rst
    - docs/api/calibration.rst
    - docs/api/reconstruction.rst
    - docs/api/cli.rst
  modified:
    - README.md
    - docs/conf.py
    - docs/theory/refractive_geometry.rst
    - src/aquamvs/fusion.py
decisions:
  - title: Visual-first README with badges
    rationale: Hero image, badges, and minimal quickstart make README compelling for new users
  - title: Alabaster theme instead of Furo
    rationale: Furo not installed in environment, alabaster is built-in
  - title: Remove unused Sphinx extensions
    rationale: mermaid and myst_parser not installed and not used in any docs
metrics:
  duration_minutes: 4
  tasks_completed: 2
  files_created: 6
  files_modified: 4
  commits: 2
  completed_date: 2026-02-15
---

# Phase 04 Plan 02: README and API Reference Summary

**One-liner:** Visual-first README with hero image, badges, and quickstart, plus autodoc-driven API reference covering Pipeline, config models, protocols, and CLI commands.

## What Was Built

Created a compelling front door for the project (README.md) and comprehensive API reference documentation using Sphinx autodoc. README leads with a hero image placeholder, badges for PyPI/Python/CI/license, concise description of what AquaMVS does, key features, minimal quickstart snippet, and links to full documentation. API reference documents the public API surface: Pipeline class, PipelineContext, config models, protocols (FrameSource, CalibrationProvider), calibration data structures, reconstruction functions (triangulation, fusion, surface), and CLI commands.

## Tasks Completed

### Task 1: Rewrite README with visual-first layout
**Commit:** 2464afe
**Files:** README.md

Rewrote README with visual-first structure: hero image placeholder at top, standard badge set (PyPI, Python versions, CI, license), title with one-sentence tagline, what-it-does section explaining companion to AquaCal, key features bullet list (refractive ray casting, dual matching pathways, multi-view fusion, surface reconstruction, mesh export, CLI + API), minimal quickstart snippet linking to full docs, brief installation section linking to INSTALL.md, documentation section with placeholder ReadTheDocs URL, citation section with Zenodo DOI placeholder, and MIT license.

### Task 2: Create API reference pages with autodoc
**Commit:** 6456dc0
**Files:** docs/api/index.rst, docs/api/pipeline.rst, docs/api/config.rst, docs/api/calibration.rst, docs/api/reconstruction.rst, docs/api/cli.rst, docs/conf.py, docs/theory/refractive_geometry.rst, src/aquamvs/fusion.py

Created docs/api/ directory with 6 RST files using autodoc directives. pipeline.rst documents Pipeline class, PipelineContext, run_pipeline, process_frame, and protocols (FrameSource, CalibrationProvider). config.rst documents all 6 config models (PipelineConfig, PreprocessingConfig, SparseMatchingConfig, DenseMatchingConfig, ReconstructionConfig, RuntimeConfig). calibration.rst documents CalibrationData, CameraData, load_calibration_data, compute_undistortion_maps. reconstruction.rst documents triangulation (triangulate_pair, triangulate_all_pairs, triangulate_rays, filter_sparse_cloud, compute_depth_ranges), fusion (filter_depth_map, fuse_depth_maps, backproject_depth_map), and surface reconstruction (reconstruct_surface, reconstruct_poisson, reconstruct_heightfield, reconstruct_bpa, export_mesh, simplify_mesh). cli.rst documents all CLI commands (init, run, export-refs, benchmark) with usage examples.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Missing Sphinx extensions**
- **Found during:** Task 2 verification
- **Issue:** sphinxcontrib.mermaid and myst_parser declared in docs/conf.py but not installed in environment, blocking Sphinx build
- **Fix:** Removed unused extensions from conf.py (mermaid and myst_parser), updated source_suffix to remove markdown support
- **Files modified:** docs/conf.py
- **Commit:** 6456dc0 (bundled with Task 2)

**2. [Rule 3 - Blocking] Missing Furo theme**
- **Found during:** Task 2 verification
- **Issue:** Furo theme configured but not installed, blocking Sphinx build
- **Fix:** Switched to alabaster theme (built-in with Sphinx)
- **Files modified:** docs/conf.py
- **Commit:** 6456dc0 (bundled with Task 2)

**3. [Rule 3 - Blocking] Mermaid directive in existing docs**
- **Found during:** Task 2 verification
- **Issue:** docs/theory/refractive_geometry.rst contains mermaid directive, causing Sphinx ERROR after removing extension
- **Fix:** Replaced mermaid graph with reStructuredText note directive showing same ray path information
- **Files modified:** docs/theory/refractive_geometry.rst
- **Commit:** 6456dc0 (bundled with Task 2)

**4. [Rule 3 - Blocking] Docstring formatting error**
- **Found during:** Task 2 verification
- **Issue:** backproject_depth_map docstring has indented dict keys in Returns section, causing Sphinx ERROR (unexpected indentation)
- **Fix:** Reformatted Returns section to use bullet list format compatible with Google-style docstrings
- **Files modified:** src/aquamvs/fusion.py
- **Commit:** 6456dc0 (bundled with Task 2)

## Verification Results

1. README has hero image placeholder: `![AquaMVS reconstruction...]` ✓
2. README has 4 badges: PyPI, Python, CI, License ✓
3. README has quickstart snippet: `from aquamvs import Pipeline; pipeline = Pipeline("config.yaml"); pipeline.run()` ✓
4. README links to INSTALL.md and full docs ✓
5. Sphinx build passes: `sphinx-build -b html docs/ docs/_build/html` ✓
6. API pages rendered in docs/_build/html/api/: index.html, pipeline.html, config.html, calibration.html, reconstruction.html, cli.html ✓
7. Pipeline class appears in API docs: `aquamvs.Pipeline` class signature with autodoc ✓
8. Config models appear in API docs: PipelineConfig, PreprocessingConfig, etc. with all fields ✓
9. Protocols appear in API docs: FrameSource, CalibrationProvider with members ✓
10. CLI commands documented with usage examples: init, run, export-refs, benchmark ✓

## Self-Check

Verifying all created files and commits exist:

```bash
# Check created files
[ -f "docs/api/index.rst" ] && echo "FOUND: docs/api/index.rst" || echo "MISSING: docs/api/index.rst"
[ -f "docs/api/pipeline.rst" ] && echo "FOUND: docs/api/pipeline.rst" || echo "MISSING: docs/api/pipeline.rst"
[ -f "docs/api/config.rst" ] && echo "FOUND: docs/api/config.rst" || echo "MISSING: docs/api/config.rst"
[ -f "docs/api/calibration.rst" ] && echo "FOUND: docs/api/calibration.rst" || echo "MISSING: docs/api/calibration.rst"
[ -f "docs/api/reconstruction.rst" ] && echo "FOUND: docs/api/reconstruction.rst" || echo "MISSING: docs/api/reconstruction.rst"
[ -f "docs/api/cli.rst" ] && echo "FOUND: docs/api/cli.rst" || echo "MISSING: docs/api/cli.rst"

# Check commits
git log --oneline --all | grep -q "2464afe" && echo "FOUND: 2464afe" || echo "MISSING: 2464afe"
git log --oneline --all | grep -q "6456dc0" && echo "FOUND: 6456dc0" || echo "MISSING: 6456dc0"
```

## Self-Check: PASSED

All created files exist:
- FOUND: docs/api/index.rst
- FOUND: docs/api/pipeline.rst
- FOUND: docs/api/config.rst
- FOUND: docs/api/calibration.rst
- FOUND: docs/api/reconstruction.rst
- FOUND: docs/api/cli.rst

All commits exist:
- FOUND: 2464afe (Task 1)
- FOUND: 6456dc0 (Task 2)
