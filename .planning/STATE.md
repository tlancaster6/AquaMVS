# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-14)

**Core value:** Accurate refractive multi-view stereo reconstruction — cameras in air, geometry underwater, Snell's law bridging the two.
**Current focus:** Phase 1 - Dependency Resolution and Packaging Foundations

## Current Position

Phase: 1 of 5 (Dependency Resolution and Packaging Foundations)
Plan: 01-02 complete, 2 of 2 phase plans done
Status: Phase complete
Last activity: 2026-02-14 — Completed 01-02-PLAN.md (CI/CD setup)

Progress: [████░░░░░░] 40% (2 of 5 phase plans complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 2
- Average duration: 10 min
- Total execution time: 0.33 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01    | 2     | 20min | 10min    |

**Recent Trend:**
- Last 5 plans: 01-01 (15min), 01-02 (5min)
- Trend: Accelerating (33% faster than baseline)

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- PyTorch over NumPy for internal math (GPU support, differentiability)
- Dual pathway (LightGlue sparse + RoMa dense) for different accuracy/speed tradeoffs
- Package as both CLI + library to serve pipeline users and custom workflow developers

**Phase 01 Plan 01 (2026-02-14):**
- PyTorch as user-managed prerequisite with import-time check (not in pyproject.toml)
- LightGlue pinned to commit edb2b83 (v0.2 release) via git URL
- RoMa v2 pinned to user fork at tlancaster6/RoMaV2 (dataclasses metadata fix)
- AquaCal as standard PyPI dependency (aquacal>=0.1.0)
- prereq-docs strategy: Document LightGlue and RoMa as manual install prerequisites for PyPI compatibility

**Phase 01 Plan 02 (2026-02-14):**
- Matrix testing across Ubuntu and Windows with Python 3.10, 3.11, 3.12 (6 combinations)
- Trusted Publishing (OIDC) eliminates API token management for PyPI uploads
- Three-stage publish pipeline: build -> TestPyPI -> PyPI with manual approval gate

### Pending Todos

None yet.

### Blockers/Concerns

**Phase 1 Dependencies (RESOLVED in 01-01):**
- ✓ LightGlue: git dependency pin to edb2b83, documented as prerequisite
- ✓ RoMa v2: git dependency pin to user fork, documented as prerequisite
- ✓ AquaCal: standard PyPI dependency (aquacal>=0.1.0)
- ✓ PyPI compatibility: prereq-docs strategy allows PyPI upload

**Phase 3 Refactoring:**
- Backward compatibility scope needs definition — audit which APIs are public vs internal to minimize breakage

## Session Continuity

Last session: 2026-02-14 (plan execution)
Stopped at: Completed 01-02-PLAN.md (CI/CD setup) - Phase 01 complete
Resume file: None
