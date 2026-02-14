---
phase: 01-dependency-resolution-and-packaging-foundations
plan: 01
subsystem: packaging
tags: [pyproject, dependencies, lightglue, romav2, aquacal, semantic-versioning, pypi]

# Dependency graph
requires:
  - phase: 01-RESEARCH
    provides: Dependency resolution strategy for git-based packages
provides:
  - Locked dependency versions in pyproject.toml with minimum bounds
  - Import-time PyTorch check with clear installation instructions
  - CHANGELOG.md with v0.1.0 entry and semantic versioning infrastructure
  - Installation documentation for git-based prerequisites (LightGlue, RoMa)
  - PyPI-compatible dependency specification with prerequisite workflow
affects: [01-02-ci-cd, packaging, distribution]

# Tech tracking
tech-stack:
  added: [python-semantic-release]
  patterns: [prerequisite-based-install, import-time-dependency-check]

key-files:
  created: [CHANGELOG.md, INSTALL.md, requirements-prereqs.txt]
  modified: [pyproject.toml, src/aquamvs/__init__.py, README.md]

key-decisions:
  - "PyTorch as user-managed prerequisite with import-time check (not declared in pyproject.toml)"
  - "LightGlue pinned to commit edb2b83 (v0.2 release) via git URL"
  - "RoMa v2 pinned to user fork at tlancaster6/RoMaV2 (dataclasses metadata fix)"
  - "AquaCal as standard PyPI dependency (aquacal>=0.1.0)"
  - "prereq-docs strategy: Document LightGlue and RoMa as manual install prerequisites"
  - "Minimum version bounds for all dependencies (kornia>=0.7.0, open3d>=0.18.0, etc.)"

patterns-established:
  - "Prerequisite installation workflow: PyTorch → git prerequisites → AquaMVS"
  - "Git-based dependencies in requirements-prereqs.txt, bare names in pyproject.toml"
  - "Import-time checks with clear error messages pointing to installation docs"

# Metrics
duration: 15min
completed: 2026-02-14
---

# Phase 01 Plan 01: Dependency Resolution and Packaging Foundations Summary

**PyPI-compatible dependency specification with locked versions, semantic versioning infrastructure, and prerequisite-based installation workflow for git dependencies (LightGlue, RoMa v2)**

## Performance

- **Duration:** 15 min
- **Started:** 2026-02-14T19:45:00Z
- **Completed:** 2026-02-14T20:00:00Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Locked all dependency versions with minimum bounds for reproducible installs
- Resolved PyPI git-URL conflict with prerequisite documentation strategy
- Established semantic versioning infrastructure with CHANGELOG.md
- Added import-time PyTorch check with clear installation instructions
- Created comprehensive installation documentation (INSTALL.md)

## Task Commits

Each task was committed atomically:

1. **Task 1: Update pyproject.toml with locked dependency decisions, create CHANGELOG.md, add torch import check** - `47cb139` (feat)
2. **Task 2: Document LightGlue and RoMa as install prerequisites** - `a469f1e` (docs)

## Files Created/Modified
- `pyproject.toml` - Locked dependency versions, semantic-release config, wheel exclusion, PyPI-compatible dependencies
- `CHANGELOG.md` - Initial v0.1.0 entry with feature list
- `src/aquamvs/__init__.py` - Import-time PyTorch check with installation instructions
- `INSTALL.md` - Three-step installation guide (PyTorch → git prereqs → AquaMVS)
- `requirements-prereqs.txt` - Git URLs for LightGlue and RoMa v2
- `README.md` - Installation section referencing INSTALL.md

## Decisions Made

**Task 1:**
- PyTorch NOT declared in pyproject.toml - user-managed prerequisite with import-time check
- LightGlue pinned to commit edb2b83 (v0.2 release)
- RoMa v2 pinned to user fork at tlancaster6/RoMaV2@3862b19 (dataclasses metadata fix)
- AquaCal as standard PyPI dependency (aquacal>=0.1.0)
- All other dependencies have minimum version bounds (kornia>=0.7.0, open3d>=0.18.0, etc.)
- Wheel excludes tests, docs, planning files (include-package-data = false)
- Semantic-release configured for automated versioning

**Task 2 (checkpoint decision):**
- Selected **prereq-docs** strategy to resolve PyPI git-URL conflict
- Git URLs moved from pyproject.toml to requirements-prereqs.txt
- LightGlue and RoMa v2 listed as bare package names in dependencies
- Users install prerequisites manually before `pip install aquamvs`
- PyPI upload will succeed (no git+https:// URLs in metadata)

## Deviations from Plan

None - plan executed exactly as written. Task 2 was a planned checkpoint decision (not a deviation).

## Issues Encountered

None - both tasks completed without issues.

## User Setup Required

**Manual prerequisite installation required:**

Users must install prerequisites BEFORE installing AquaMVS:

1. **PyTorch** (from pytorch.org with appropriate CUDA version)
2. **LightGlue** (`pip install git+https://github.com/cvg/LightGlue.git@edb2b83`)
3. **RoMa v2** (`pip install git+https://github.com/tlancaster6/RoMaV2.git@3862b19`)

Or via `pip install -r requirements-prereqs.txt` for steps 2-3.

See [INSTALL.md](../../INSTALL.md) for complete instructions.

## Next Phase Readiness

**Ready for Phase 01 Plan 02 (CI/CD):**
- Dependency specification is PyPI-compatible (no git URLs in pyproject.toml)
- Semantic-release configured and ready for automated version bumps
- CHANGELOG.md exists for release notes generation
- Wheel build configuration complete (excludes dev files)

**Considerations for CI/CD:**
- GitHub Actions will need to install prerequisites before running tests
- Publish workflow can use `python -m build` and `twine upload` directly
- Documentation should mention prerequisite installation step for users

**No blockers.**

## Self-Check: PASSED

All claims verified:
- ✓ CHANGELOG.md exists
- ✓ INSTALL.md exists
- ✓ requirements-prereqs.txt exists
- ✓ Task 1 commit 47cb139 exists
- ✓ Task 2 commit a469f1e exists
- ✓ Torch import check in src/aquamvs/__init__.py
- ✓ python-semantic-release in dev dependencies

---
*Phase: 01-dependency-resolution-and-packaging-foundations*
*Completed: 2026-02-14*
