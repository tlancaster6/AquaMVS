---
status: resolved
trigger: "ci-open3d-segfault"
created: 2026-02-14T00:00:00Z
updated: 2026-02-14T00:00:00Z
---

## Current Focus

hypothesis: CONFIRMED - OffscreenRenderer segfaults at C-level on headless Linux because no environment check happens before instantiation
test: Implement environment detection in _offscreen_available to check DISPLAY/CI before attempting OffscreenRenderer creation
expecting: Skip OffscreenRenderer on headless, fallback to legacy or none
next_action: Fix _offscreen_available to detect headless environment

## Symptoms

expected: Tests should run on headless CI (GitHub Actions Ubuntu) without crashing. Visualization tests that need Open3D rendering should be skipped gracefully.
actual: Fatal segfault at pytest collection time. The segfault occurs in `_offscreen_available()` at scene.py line 24, called from `_detect_rendering_backend()` at scene.py line 69, triggered by test_scene_viz.py line 18 at module level during import.
errors: Fatal Python error: Segmentation fault at scene.py line 24 in _offscreen_available. Exit code 139.
reproduction: Run `pytest tests/ --cov=aquamvs --cov-report=xml --cov-report=term-missing -m "not slow"` on GitHub Actions Ubuntu runner (headless, no GPU, no display).
started: Known issue, previously attempted fix didn't work or wasn't completed. The MEMORY.md notes: "Open3D OffscreenRenderer may be unavailable on headless/CI — check at runtime, degrade gracefully"

## Eliminated

## Evidence

- timestamp: 2026-02-14T00:01:00Z
  checked: scene.py lines 13-32 (_offscreen_available)
  found: Function tries to instantiate OffscreenRenderer(64, 64) inside try/except with stderr redirected
  implication: This should work, but segfault happens at C-level before Python exception handling can catch it

- timestamp: 2026-02-14T00:02:00Z
  checked: test_scene_viz.py line 18
  found: `RENDERING_AVAILABLE = _detect_rendering_backend() != "none"` called at MODULE LEVEL during import
  implication: This triggers _offscreen_available() during pytest collection, before any test runs

- timestamp: 2026-02-14T00:03:00Z
  checked: scene.py lines 55-78 (_detect_rendering_backend)
  found: Deferred detection pattern - uses global cache to run only once. Line 69 calls _offscreen_available()
  implication: The stderr redirection in _offscreen_available isn't enough - segfault happens before Python try/except can catch it

- timestamp: 2026-02-14T00:04:00Z
  checked: .github/workflows/test.yml
  found: No DISPLAY variable set, no xvfb-run. Environment has OPEN3D_CPU_RENDERING=true at line 20
  implication: OPEN3D_CPU_RENDERING doesn't prevent the segfault - need to detect headless environment BEFORE trying to create OffscreenRenderer

- timestamp: 2026-02-14T00:05:00Z
  checked: Overall flow
  found: test_scene_viz.py imports and calls _detect_rendering_backend() at module level → _detect_rendering_backend() calls _offscreen_available() → _offscreen_available() tries to instantiate OffscreenRenderer → C-level segfault on headless Linux
  implication: Need to prevent OffscreenRenderer instantiation on headless systems by checking for DISPLAY environment variable or CI markers BEFORE attempting

- timestamp: 2026-02-14T00:06:00Z
  checked: Local test run on Windows
  found: pytest tests/test_visualization/test_scene_viz.py - 18 passed in 41.05s
  implication: Fix doesn't break existing functionality on Windows where OffscreenRenderer works

## Resolution

root_cause: On headless Linux (GitHub Actions Ubuntu), Open3D's OffscreenRenderer constructor segfaults at the C/C++ level when there is no display server. The try/except in _offscreen_available() cannot catch this because it's a SIGSEGV, not a Python exception. The code needs to detect headless environments BEFORE attempting to instantiate OffscreenRenderer.
fix: Added environment detection in _offscreen_available() to check for DISPLAY environment variable and CI markers (CI, GITHUB_ACTIONS, TRAVIS, CIRCLECI) on Linux systems. If no DISPLAY is set on Linux, skip the OffscreenRenderer probe and return False, allowing the code to fall back to legacy Visualizer or none.
verification: Local Windows tests pass (18/18). Logic verified: on Linux without DISPLAY, function returns False immediately. On headless CI (GITHUB_ACTIONS=1, no DISPLAY), returns False. Tests that require rendering are properly gated by RENDERING_AVAILABLE skip marker, so they'll be skipped when backend detection returns "none".
files_changed: ["src/aquamvs/visualization/scene.py"]
