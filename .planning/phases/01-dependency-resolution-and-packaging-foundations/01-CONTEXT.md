# Phase 1: Dependency Resolution and Packaging Foundations - Context

**Gathered:** 2026-02-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Make AquaMVS installable from PyPI with clean dependencies, a proper wheel, and CI testing. Users can `pip install aquamvs` and get a working package. Scope: dependency resolution, pyproject.toml cleanup, versioning, CI/CD workflows.

</domain>

<decisions>
## Implementation Decisions

### Dependency strategy
- **LightGlue**: Git dependency pin to specific commit hash (e.g., `lightglue @ git+https://github.com/cvg/LightGlue@<hash>`)
- **RoMa v2**: Git dependency pin to user's fork (`romav2 @ git+https://github.com/tlancaster6/RoMaV2@<hash>`) — switch to upstream when PR merges
- **AquaCal**: Normal PyPI dependency (`aquacal>=X.Y`) — user is publishing AquaCal to PyPI in parallel, assume it will be available
- **PyTorch + torchvision**: NOT declared as dependencies — user-managed prerequisites. Document install instructions in README
- **Git requirement**: Acceptable tradeoff that git-based deps require git at install time. No need for fallback documentation
- All other deps (kornia, Open3D, etc.) declared normally in pyproject.toml

### Install experience
- `pip install aquamvs` installs everything except PyTorch/torchvision
- Import-time check for torch — raise clear error pointing to install docs if missing
- CLI entry points included via console_scripts (`aquamvs` command registered by pip)
- `[dev]` extras group for development dependencies (pytest, black, etc.)

### Claude's Discretion
- Specific commit hashes for LightGlue and RoMa pins
- Versioning scheme (SemVer vs CalVer)
- CI platform matrix and Python version targets
- PyPI publish trigger mechanism (tag-based vs manual)
- Wheel build configuration details
- Minimum version pins for kornia, Open3D, and other standard deps

</decisions>

<specifics>
## Specific Ideas

- User expects the install flow to be: create venv/conda env → install PyTorch per pytorch.org instructions → `pip install aquamvs` → ready to go
- README installation section should guide users through this sequence
- AquaCal PyPI publication is happening in parallel — not a blocker for planning, but the package may not be live yet during implementation

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-dependency-resolution-and-packaging-foundations*
*Context gathered: 2026-02-14*
