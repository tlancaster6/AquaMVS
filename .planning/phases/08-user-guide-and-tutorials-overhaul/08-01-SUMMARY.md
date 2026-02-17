---
phase: 08-user-guide-and-tutorials-overhaul
plan: 01
subsystem: docs
tags: [sphinx, myst-nb, troubleshooting, cli-guide]

requires:
  - phase: 07-post-qa-bug-triage
    provides: "Unified benchmark command, init --preset, --output-fps, removed deprecated keys"
provides:
  - "myst-nb configured in Sphinx for notebook rendering on RTD"
  - "Centralized troubleshooting page covering installation, pipeline, notebook, and config issues"
  - "CLI guide updated for all Phase 7 command changes"
affects: [08-02, 08-03, 08-04]

tech-stack:
  added: [myst-nb, nbformat]
  patterns: [nb_execution_mode off for pre-executed outputs]

key-files:
  created:
    - docs/troubleshooting.rst
  modified:
    - docs/conf.py
    - docs/cli_guide.md
    - pyproject.toml

key-decisions:
  - "Replace myst_parser with myst_nb (not both — avoids registration conflict)"
  - "nb_execution_mode = off — RTD renders pre-executed outputs, no re-execution"
  - "source_suffix maps .md to myst-markdown and .ipynb to myst-nb"

patterns-established:
  - "Troubleshooting as centralized page with Problem/Solution definition-list format"
  - "CLI guide references init --preset not runtime quality_preset"

duration: 5min
completed: 2026-02-17
---

# Plan 08-01: myst-nb Infrastructure and CLI Guide Polish Summary

**myst-nb configured for notebook rendering, centralized troubleshooting page, CLI guide updated for Phase 7 benchmark/preset/FPS changes**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-17
- **Completed:** 2026-02-17
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Replaced myst_parser with myst_nb in Sphinx config with nb_execution_mode = "off"
- Created comprehensive troubleshooting.rst covering installation, pipeline, notebook, and config issues
- Updated CLI guide: init --preset, unified benchmark command, --output-fps, removed deprecated keys
- Added myst-nb and nbformat to pyproject.toml dev dependencies

## Task Commits

1. **Task 1: Configure myst-nb infrastructure and troubleshooting page** - `b8aae23` (feat)
2. **Task 2: Polish CLI guide for Phase 7 changes** - `ab38fdd` (docs)

## Files Created/Modified
- `docs/conf.py` - myst-nb extension, nb_execution_mode off, source_suffix updates
- `docs/troubleshooting.rst` - Centralized troubleshooting with 4 sections
- `docs/cli_guide.md` - Updated for Phase 7 CLI surface
- `pyproject.toml` - myst-nb and nbformat in dev deps

## Decisions Made
- Replace myst_parser with myst_nb (not keep both — avoids Sphinx registration warning)
- nb_execution_mode = "off" for RTD to render stored outputs without re-execution

## Deviations from Plan
None - plan executed as written

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- myst-nb infrastructure ready for notebook rendering in Plans 02-04
- Troubleshooting page ready for cross-references from notebooks and tutorial index

---
*Phase: 08-user-guide-and-tutorials-overhaul*
*Completed: 2026-02-17*
