# Technology Stack

**Analysis Date:** 2026-02-14

## Languages

**Primary:**
- Python 3.10+ - All application code, from calibration loading to pipeline orchestration

## Runtime

**Environment:**
- Python 3.10, 3.11, or 3.12 (as specified in `pyproject.toml`)
- Platform: Windows (Git Bash/MINGW64) or Linux
- PyTorch backend: CPU or CUDA (configurable via `device` config)

**Package Manager:**
- pip - Standard Python package management
- Lockfile: Not present (uses `pyproject.toml` with setuptools build backend)

## Frameworks

**Core Computation:**
- PyTorch - Tensor operations, neural networks, automatic differentiation
  - All mathematical operations use PyTorch, never NumPy for internal math
  - NumPy used only at AquaCal integration boundary (`aquacal.io`)

**Image Processing:**
- OpenCV (cv2) 4.6+ - Image I/O, undistortion, CLAHE preprocessing, basic image ops
- Kornia - GPU-accelerated geometric and image transforms

**Computer Vision - Features:**
- LightGlue (git+https://github.com/cvg/LightGlue.git) - Feature matching
  - Supports multiple detector backends: SuperPoint, ALIKED, DISK
  - Created via factory in `src/aquamvs/features/extraction.py:create_extractor()`

**Computer Vision - Dense Matching:**
- RoMa v2 (romav2 PyPI) - Dense warping and correspondence
  - Produces overlap certainty and warp grids
  - Used in `src/aquamvs/dense/roma_depth.py`
  - Note: Installed with `pip install --no-deps romav2` to work around PyPI bug on Python 3.10+

**3D Geometry:**
- Open3D - Point cloud representation, mesh generation, ICP alignment
  - Poisson/Ball Pivoting Algorithm surface reconstruction
  - KD-tree operations for color transfer (`src/aquamvs/surface.py:_transfer_colors()`)

**Configuration & Serialization:**
- PyYAML - YAML config loading/saving in `src/aquamvs/config.py`

**Visualization:**
- Matplotlib - 2D plots, feature visualizations, benchmark reports
- Matplotlib OpenGL backend - For 3D scene rendering

**Scientific Computing:**
- NumPy - Array operations, mostly at boundaries with AquaCal
- SciPy - Interpolation (griddata for heightfield reconstruction in `src/aquamvs/surface.py`)
- einops 0.8.1+ - Tensor rearrangement utilities
- rich 14.2.0+ - Terminal output formatting
- tqdm 4.67.1+ - Progress bar display

## Key Dependencies

**Critical:**
- torch - Core differentiable computation, device management
- romav2 - Dense correspondence matching (required for full pipeline)
- lightglue - Sparse feature matching (required for sparse reconstruction)
- open3d - 3D geometry, mesh generation (required for surface output)
- opencv-python 4.6+ - Image I/O and preprocessing

**Infrastructure:**
- pyyaml - Configuration file parsing and generation
- scipy - Height-field interpolation (SurfaceConfig method = "heightfield")
- matplotlib - Visualization and benchmark plots

**External Dependency - Local Editable:**
- aquacal (not in pyproject.toml dependencies, installed separately via `pip install -e ../AquaCal`)
  - Provides: `aquacal.io.serialization.load_calibration()` for loading calibration JSON
  - Provides: `aquacal.io.video.VideoSet` for synchronized multi-camera video reading
  - Boundary: Only imported in `src/aquamvs/calibration.py` and `src/aquamvs/cli.py`

## Configuration

**Environment:**
- Configured entirely via YAML files (no .env files)
- Device selection: "cpu" or "cuda" in `device` config (defaults to CPU)
- Feature extraction: "superpoint", "aliked", or "disk" (defaults to superpoint)
- Matching: "lightglue" or "roma" (defaults to lightglue)
- All parameters are defined as dataclasses in `src/aquamvs/config.py`

**Build:**
- pyproject.toml: setuptools build backend
  - Entry point: `aquamvs = "aquamvs.cli:main"` (CLI command)
  - Package discovery: `setuptools.packages.find` in `src/` directory

## Platform Requirements

**Development:**
- Python 3.10+
- pip for package installation
- Git Bash or native shell (Windows: MINGW64, not WSL)
- CUDA toolkit (optional, for GPU acceleration)

**Production:**
- Same as development
- Deployment target: Local multi-core systems or GPU-enabled servers
- No web services, cloud APIs, or distributed frameworks required

---

*Stack analysis: 2026-02-14*
