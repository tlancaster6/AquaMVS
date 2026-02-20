---
created: 2026-02-20T13:11:26.900Z
title: Rename ring_camera to core_camera across codebase
area: general
files:
  - src/aquamvs/ (multiple modules)
  - docs/
  - docs/tutorial/
  - CLAUDE.md
---

## Problem

The codebase uses "ring camera" / `ring_camera` / `is_ring` terminology throughout code, docs, and tutorials to distinguish the standard-lens cameras from the auxiliary/center camera. This naming is specific to the user's rig where the core cameras happen to be arranged in a ring, but other rig configurations may arrange their core cameras differently. The terminology should be generalized to "core camera" / `core_camera` / `is_core` so the codebase is rig-agnostic.

This is a broad rename touching:
- Python source code (variable names, function params, config fields, comments)
- Documentation (CLAUDE.md terminology section, API docs, docstrings)
- Tutorial notebooks and example configs
- Tests and fixtures
- Potentially AquaCal upstream as well (coordinate with that project)

## Solution

1. Audit all occurrences of `ring_camera`, `ring camera`, `is_ring`, `ring_count`, etc.
2. Rename to `core_camera`, `core camera`, `is_core`, `core_count`, etc.
3. Update CLAUDE.md terminology section (ring camera -> core camera)
4. Update tutorial notebooks and example configs
5. Consider backwards-compatible config aliases if needed for existing user configs
6. Coordinate with AquaCal if the terminology crosses the boundary
