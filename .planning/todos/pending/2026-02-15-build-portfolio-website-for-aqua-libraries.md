---
created: 2026-02-15T01:16:50.000Z
title: Build portfolio website for Aqua libraries
area: general
files: []
---

## Problem

After AquaCal, AquaMVS, and the planned AquaPose library are complete, need a
showcase website to demonstrate the work to potential employers. The projects
involve technically impressive computer vision work (refractive multi-view
stereo, underwater 3D reconstruction) that benefits from visual, interactive
presentation rather than just GitHub READMEs.

## Solution

**Hosting:** Static site on GitHub Pages using Astro or Next.js (static export).
Add a small VPS (Hetzner/DigitalOcean/Railway) only if dynamic backends needed.

**Interactive demos (high-impact, low-effort):**
- Pre-computed 3D viewer: Export meshes/point clouds as `.glb`/`.ply`, embed
  with Three.js or model-viewer web component. Orbit, zoom, toggle textures.
- Before/after comparison sliders: Raw images vs undistorted vs depth maps vs
  reconstructions (img-comparison-slider).
- Colab/Binder notebook links for simplified pipeline walkthroughs.
- Lightweight WebGL demos for isolated concepts (e.g., Snell's law ray
  bending visualization).

**Content per project page:**
- Problem statement + approach
- Embedded 3D viewers
- Image comparison sliders
- Quantitative results (reprojection error tables, accuracy benchmarks)
- Links to GitHub repos + docs

**Not worth the effort:**
- Live GPU inference in browser (WebGPU/WASM porting too costly)
- Server-side "upload and process" (GPU costs, abuse prevention)
