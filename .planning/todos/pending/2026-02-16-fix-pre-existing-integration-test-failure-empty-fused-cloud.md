---
created: 2026-02-16T22:29:28.512Z
title: Fix pre-existing integration test failure (empty fused cloud)
area: testing
files:
  - tests/test_integration.py:270-336
  - src/aquamvs/pipeline/stages/fusion.py
  - src/aquamvs/pipeline/stages/surface.py
---

## Problem

`tests/test_integration.py::test_end_to_end_reconstruction` fails because the synthetic scene produces an empty fused point cloud after depth map fusion and outlier removal. The test asserts that `point_cloud/` and `mesh/` directories exist (lines 290-291), but since the fused cloud is empty, `save_point_cloud` is never called and neither directory is created.

Confirmed pre-existing: fails identically on both the old code (pre two-pass refactor, commit 3acf5e9) and the new code. The synthetic scene geometry (3 cameras, flat sand plane at Z=1.0, water at Z=0.978) likely doesn't produce enough consistent depth estimates to survive geometric consistency filtering.

Log output shows:
```
WARNING  aquamvs.pipeline.stages.fusion:fusion.py: Frame 0: fused point cloud is empty, skipping point cloud save
WARNING  aquamvs.pipeline.stages.surface:surface.py: Frame 0: fused point cloud is empty, skipping surface reconstruction
```

## Solution

Investigate why the synthetic scene produces an empty fused cloud. Possible causes:
1. Geometric consistency filter is too aggressive for 3-camera synthetic setup (requires `min_consistent_views=3` but only 3 cameras total)
2. Depth range estimation from sparse cloud may be off, causing plane sweep to miss the actual surface
3. Synthetic images may lack sufficient texture for NCC matching

Fix options:
- Adjust synthetic scene config (lower `min_consistent_views`, more cameras, add texture)
- Or relax the test to check for depth maps but not require non-empty fusion
- Or fix the underlying depth estimation to work with the synthetic scene
