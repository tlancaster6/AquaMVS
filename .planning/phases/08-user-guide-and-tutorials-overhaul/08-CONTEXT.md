# Phase 8: User Guide and Tutorials Overhaul - Context

**Gathered:** 2026-02-17
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace the current thin tutorials section with 4 polished pages: overview (CLI vs API routing), CLI usage example, end-to-end Python API notebook, and benchmarking API notebook. Jupyter notebooks render with static outputs on RTD and include Colab launch buttons. Import-only smoke tests catch breakage from upstream code changes.

</domain>

<decisions>
## Implementation Decisions

### Existing content reuse
- CLI guide (`docs/cli_guide.md`): polish in place — update for Phase 7 changes (benchmark rebuild, preset baking via `init --preset`, `--output-fps`, removed deprecated keys), fix formatting/tone
- API notebook (`docs/tutorial/notebook.ipynb`): polish in place — update for Phase 7 changes, ensure Colab plug-and-play experience
- Notebooks must auto-download and unpack the example dataset (GitHub Releases URL) in the first code cell — no manual download step
- All file paths in notebooks hardcoded to match the published example dataset structure (not configurable variables)

### Overview page
- Claude's discretion on whether to make a standalone page or fold into tutorial/index.rst — pick what fits existing doc structure best

### Notebook rendering on RTD
- Inline rendering on RTD — notebooks display as HTML pages with pre-executed static outputs
- Use nbsphinx or myst-nb (Claude's choice) + pandoc via RTD `build.apt_packages`
- Notebooks committed with cell outputs already executed (RTD does not re-execute)
- Each notebook page includes an "Open in Colab" launch button at the top

### Notebook CI testing
- Import-only smoke test in GitHub Actions — verify notebook code cells parse and imports resolve
- No full execution in CI (avoids needing example dataset and GPU in CI environment)

### Benchmarking notebook (new)
- Full workflow: run `aquamvs benchmark`, load results, compare LightGlue vs RoMa pathways, visualize metrics, interpret tradeoffs
- Also covers profiling: torch.profiler stage timings showing where time goes per pathway
- Uses the example dataset (same auto-download as API tutorial), not synthetic data
- Includes visual comparisons: side-by-side rendered point clouds, depth maps, and/or metric bar charts

### Audience and tone
- Mixed audience: researchers reproducing results AND engineers integrating AquaMVS — assume basic Python + 3D vision knowledge
- Professional/academic tone — clean, precise language fitting an academic library
- Light domain explanations: 1-2 sentence context for concepts (plane sweep, fusion, etc.) with links to Theory docs for deep dives
- Troubleshooting centralized only — tutorials link to a central troubleshooting section, no per-tutorial troubleshooting blocks

### Claude's Discretion
- Choice between nbsphinx and myst-nb for notebook rendering
- Overview page structure (standalone vs folded into index)
- Exact notebook cell organization and markdown structure
- How to implement the Colab launch button (badge image, nbsphinx config, or manual markdown)
- Specific matplotlib/Open3D visualization approach for benchmark comparisons

</decisions>

<specifics>
## Specific Ideas

- "Build for easy Google Colab plug and play" — notebooks should work with zero local setup when opened in Colab
- Auto-download example dataset in first code cell (wget/requests from GitHub Releases URL, unzip, set paths)
- Benchmarking notebook shows both accuracy comparison AND profiling (stage timing breakdown) in one notebook

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 08-user-guide-and-tutorials-overhaul*
*Context gathered: 2026-02-17*
