# Phase 4: Documentation and Examples - Context

**Gathered:** 2026-02-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Comprehensive documentation and working examples so users can learn, install, and use AquaMVS. Covers README, tutorials, example dataset, API reference, and hosted docs site. No new features or code changes beyond what's needed for documentation.

</domain>

<decisions>
## Implementation Decisions

### README and Quickstart
- Visual-first README: lead with a hero image/GIF showing reconstruction results, then explain what AquaMVS does
- Hero image generated from the example dataset as part of this phase (not pre-existing)
- Minimal quickstart in README: `pip install aquamvs` + short Python snippet, link to full docs for details
- Standard badge set at top: PyPI version, Python versions, CI status, license

### Tutorial Depth and Format
- One end-to-end tutorial covering the full reconstruction workflow
- Dual format: Jupyter notebook (Python API focus) + rendered markdown version (CLI focus)
- Jupyter notebook uses `Pipeline` class and programmatic API
- Markdown guide uses `aquamvs run` CLI workflow
- Show intermediate visualizations: sparse match overlays, depth maps, consistency maps, fused point cloud, and final mesh

### Example Dataset
- Real data subset from the 13-camera rig, 1 frame per camera (single frame-set demonstrates all key functionality)
- Distributed via separate download (Zenodo or GitHub Releases), not bundled in repo
- Includes pre-made calibration JSON for self-contained usage
- Mentions AquaCal as the calibration source with link, for users who want the full workflow

### API Reference and Hosting
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

</decisions>

<specifics>
## Specific Ideas

- Hero image should be generated from the example dataset reconstruction output — shows real results, not placeholder art
- Jupyter notebook = Python API path, markdown guide = CLI path — same workflow, two interfaces
- Full 13-camera rig for example data (including center/auxiliary camera) but just 1 frame — minimal download, maximum coverage
- Theory section should be reference-quality: someone could understand the full pipeline math from reading it

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-documentation-and-examples*
*Context gathered: 2026-02-14*
