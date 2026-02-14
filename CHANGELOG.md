# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## v0.1.0 — Initial Release

### Added
- Refractive projection model with Snell's law ray casting
- Sparse feature extraction via LightGlue (SuperPoint, ALIKED, DISK)
- Dense matching via RoMa v2
- Plane-sweep stereo with NCC/SSIM cost metrics
- Depth map fusion with confidence weighting
- Surface reconstruction (Poisson, BPA, heightfield)
- Best-view vertex coloring
- CLI with init, run, export-refs, and benchmark commands
- YAML-based pipeline configuration
- Benchmark suite with HTML reports
