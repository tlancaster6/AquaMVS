# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## v1.3.5 — PyPI Release

### Added
- Dense matching via RoMa v2 with tiled inference and memory optimization
- Benchmark suite with configurable extractor comparison and HTML reports
- Tutorial Jupyter notebooks with pre-executed outputs
- ROI mask support for selective reconstruction
- Mesh export in PLY, OBJ, STL, and GLTF formats with simplification

### Changed
- Moved unpublished dependencies (AquaCal, LightGlue, RoMa v2) to prerequisites with runtime `find_spec` checks
- Dynamic version from `importlib.metadata` instead of hardcoded string
- Updated development status classifier to Beta

### Fixed
- Bidirectional pair matching creating duplicate triangulated points
- Sparse cloud outliers blowing up depth range (percentile-based filtering)
- RoMa v2 dataclasses metadata bug (upstream PR submitted)
- Benchmark report stage keys, cloud filenames, and encoding issues

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
