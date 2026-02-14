# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-14)

**Core value:** Accurate refractive multi-view stereo reconstruction — cameras in air, geometry underwater, Snell's law bridging the two.
**Current focus:** Phase 1 - Dependency Resolution and Packaging Foundations

## Current Position

Phase: 1 of 5 (Dependency Resolution and Packaging Foundations)
Plan: Ready to plan
Status: Ready to plan
Last activity: 2026-02-14 — Roadmap created with 5 phases covering 22 v1 requirements

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: N/A
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: None yet
- Trend: N/A

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- PyTorch over NumPy for internal math (GPU support, differentiability)
- Dual pathway (LightGlue sparse + RoMa dense) for different accuracy/speed tradeoffs
- Package as both CLI + library to serve pipeline users and custom workflow developers

### Pending Todos

None yet.

### Blockers/Concerns

**Phase 1 Dependencies:**
- LightGlue PyPI status unclear — may require git dependency pin or fork/vendor approach
- RoMa v2 installation workaround needs investigation — check if newer numpy works despite metadata
- AquaCal publication decision needed — publish to PyPI first, vendor interfaces, or document manual install

**Phase 3 Refactoring:**
- Backward compatibility scope needs definition — audit which APIs are public vs internal to minimize breakage

## Session Continuity

Last session: 2026-02-14 (roadmap creation)
Stopped at: Roadmap and STATE.md created, ready for Phase 1 planning
Resume file: None
