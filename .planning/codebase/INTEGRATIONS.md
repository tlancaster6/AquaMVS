# External Integrations

**Analysis Date:** 2026-02-14

## APIs & External Services

**Sibling Library (AquaCal):**
- AquaCal - Refractive multi-camera calibration library
  - SDK/Client: Local editable dependency (`pip install -e ../AquaCal`)
  - Modules imported:
    - `aquacal.io.serialization.load_calibration()` - Load calibration JSON files
    - `aquacal.io.video.VideoSet` - Synchronized multi-camera video frame reading
  - Location in codebase: Imported only in `src/aquamvs/calibration.py` and `src/aquamvs/cli.py`
  - Contract: Provides calibration data with camera intrinsics, extrinsics, interface parameters

**LightGlue (Feature Matching):**
- GitHub: https://github.com/cvg/LightGlue.git
- SDK/Client: Python package via git URL in dependencies
- Provides: Feature matcher objects (LightGlue, SuperPoint, ALIKED, DISK)
- Location: `src/aquamvs/features/extraction.py`, `src/aquamvs/features/matching.py`
- Contract: Takes feature dictionaries (keypoints, descriptors, scores), returns matched keypoint indices

**RoMa v2 (Dense Correspondence):**
- PyPI: `romav2` package
- SDK/Client: Python module with geometry utilities
- Provides: Dense warp computation and overlap certainty estimation
- Location: `src/aquamvs/dense/roma_depth.py`
- Key utility: `romav2.geometry.to_pixel()` - Convert normalized warp coordinates to pixel coordinates
- Contract: Takes two images as input, produces warp field (normalized coordinates) and overlap certainty

## Data Storage

**Databases:**
- Not applicable - No database integration

**File Storage:**
- Local filesystem only
- Input: Video files (MP4, AVI, MKV, MOV) + calibration JSON
- Output formats:
  - YAML: Pipeline configuration (`PipelineConfig.to_yaml()`)
  - PLY: 3D point clouds and meshes (Open3D `.write_point_cloud()`, `.write_triangle_mesh()`)
  - NPZ: Depth maps and confidence scores (NumPy compressed arrays)
  - PNG: Feature visualizations, depth overlays
  - JSON: Calibration input (read only, via AquaCal)
- Output location: Configured via `output_dir` in `PipelineConfig`

**Caching:**
- None

## Authentication & Identity

**Auth Provider:**
- Not applicable - No authentication required
- All access is file-based with no API authentication

## Monitoring & Observability

**Error Tracking:**
- None - No error tracking service

**Logs:**
- Python `logging` module (standard library)
- Logger instances created in modules: `src/aquamvs/cli.py`, `src/aquamvs/pipeline.py`, `src/aquamvs/benchmark/runner.py`, etc.
- Log format: Standard Python logging format (no custom setup detected)
- Destination: stdout/stderr (no file logging configured by default)

## CI/CD & Deployment

**Hosting:**
- Not applicable - Library/command-line tool, not hosted service

**CI Pipeline:**
- Not detected in codebase
- Testing: pytest framework with markers for slow tests (in `pyproject.toml`)
- No GitHub Actions, GitLab CI, or other CI/CD configuration present

## Environment Configuration

**Configuration sources:**
1. YAML files (primary) - `PipelineConfig.from_yaml(path)`
   - All pipeline parameters defined as dataclass fields in `src/aquamvs/config.py`
   - No environment variables used for configuration

2. Command-line arguments (CLI only)
   - Parsed in `src/aquamvs/cli.py` via argparse
   - Subcommands: `init`, `run`, `export-refs`, `benchmark`

**No environment variables:**
- No .env files
- No secrets management
- All credentials/paths passed via YAML or command-line

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None - Purely local computation, no external notifications

---

*Integration audit: 2026-02-14*
