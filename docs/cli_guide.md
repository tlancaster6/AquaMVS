# CLI Reconstruction Guide

AquaMVS provides command-line tools for the complete reconstruction workflow. This guide walks through reconstruction using the `aquamvs` command-line interface, from data preparation to mesh export.

## Overview

The CLI workflow consists of:

1. **Data preparation**: Organize videos/images and calibration
2. **Configuration generation**: Auto-generate pipeline config from data
3. **(Optional) ROI masking**: Export reference images for drawing region-of-interest masks
4. **Reconstruction**: Run the full pipeline
5. **Mesh export**: Convert meshes to different formats

## Prerequisites

Ensure you have installed AquaMVS and its dependencies:

```bash
# Install PyTorch (CPU or CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install AquaMVS
pip install aquamvs

# Install optional dependencies
pip install git+https://github.com/cvg/LightGlue.git@edb2b83
pip install git+https://github.com/tlancaster6/RoMaV2.git
```

See the [Installation Guide](installation.rst) for detailed instructions.

## Step 1: Prepare Data

Organize your data with the following structure:

```
project_data/
├── videos/               # Video files OR image directories
│   ├── e3v82e0-cam1.mp4
│   ├── e3v82e1-cam2.mp4
│   └── ...
└── calibration.json      # AquaCal calibration file
```

**Video files**: Synchronized multi-camera videos (supported formats: `.mp4`, `.avi`, `.mkv`, `.mov`)

**Image directories**: Alternatively, use synchronized image sequences (one directory per camera with matching frame counts)

**Calibration**: AquaCal calibration JSON containing camera intrinsics, extrinsics, and refractive parameters

### Download Example Dataset

An example dataset is available at: [https://github.com/tlancaster6/AquaMVS/releases/download/v0.1.0-example-data/aquamvs-example-dataset.zip](https://github.com/tlancaster6/AquaMVS/releases/download/v0.1.0-example-data/aquamvs-example-dataset.zip)

## Step 2: Generate Configuration

Use `aquamvs init` to auto-generate a pipeline configuration from your data:

```bash
aquamvs init \
  --video-dir project_data/videos \
  --pattern "^([a-z0-9]+)-" \
  --calibration project_data/calibration.json \
  --output-dir ./output
```

**Parameters:**

- `--video-dir`: Directory containing video files (or image directories)
- `--pattern`: Regex pattern to extract camera name from filename. The first capture group is used as the camera name. Example: `"^([a-z0-9]+)-"` extracts `e3v82e0` from `e3v82e0-cam1.mp4`
- `--calibration`: Path to AquaCal calibration JSON
- `--output-dir`: Output directory for reconstruction results
- `--config`: (Optional) Output path for generated config YAML (default: `config.yaml`)

**Output:**

The command prints a summary of matched cameras and saves `config.yaml`:

```
======================================================================
Configuration Initialization Summary
======================================================================

[OK] Matched 12 camera(s):
  e3v82e0        -> e3v82e0-cam1.mp4
  e3v82e1        -> e3v82e1-cam2.mp4
  ...

[OK] Configuration saved to: config.yaml
======================================================================
```

The generated config contains default parameters for all pipeline stages. You can edit `config.yaml` to customize reconstruction settings.

## Step 3: (Optional) Create ROI Masks

To limit reconstruction to a specific region of interest (e.g., exclude pool edges, instruments), you can create mask images:

```bash
aquamvs export-refs config.yaml --frame 0
```

**What this does:**

1. Reads frame 0 from all cameras
2. Applies undistortion (removes lens distortion)
3. Saves undistorted reference images to `output/reference_images/{camera}.png`

**Next steps:**

1. Open the exported images in an image editor (GIMP, Photoshop, Paint.NET)
2. Draw a mask where white = region to reconstruct, black = region to ignore
3. Save masks as PNG files in a `masks/` directory with the same camera names
4. Update your config:

```yaml
mask_dir: ./masks
```

Re-run the pipeline — only masked regions will be reconstructed.

## Step 4: Run Reconstruction

Execute the full reconstruction pipeline:

```bash
aquamvs run config.yaml
```

**What happens:**

1. **Undistortion**: Apply camera calibration to remove lens distortion
2. **Feature matching**: Extract and match features across camera pairs (LightGlue or RoMa)
3. **Triangulation**: Compute 3D points from feature correspondences (sparse mode) or depth ranges (full mode)
4. **Plane sweep stereo**: Dense depth estimation via photometric cost volume (full mode only)
5. **Depth fusion**: Merge multi-view depth maps into a single point cloud
6. **Surface reconstruction**: Generate a triangle mesh from the point cloud

**Progress:** A progress bar shows frame processing status. Logs indicate stage completion.

**Optional flags:**

- `-v` / `--verbose`: Enable verbose (DEBUG) logging
- `--device cuda`: Override device (use GPU if available)
- `-q` / `--quiet`: Suppress progress bars (useful for non-interactive use, CI)

**Example with GPU:**

```bash
aquamvs run config.yaml --device cuda
```

## Step 5: Examine Results

The output directory contains frame-wise results:

```
output/
├── frame_000000/
│   ├── e3v82e0_depth.npz           # Depth map for each camera
│   ├── e3v82e0_consistency.npz     # Consistency map
│   ├── ...
│   ├── fused.ply                   # Fused point cloud
│   └── surface.ply                 # Reconstructed mesh
├── frame_000010/
│   └── ...
└── summary/
    └── timeseries_gallery.png      # (If multiple frames)
```

**Depth maps (`{camera}_depth.npz`)**: Per-camera depth estimates (ray depth in meters)

**Consistency maps (`{camera}_consistency.npz`)**: Number of agreeing source cameras per pixel

**Fused point cloud (`fused.ply`)**: Merged 3D points from all cameras with colors

**Surface mesh (`surface.ply`)**: Triangle mesh (Poisson reconstruction by default)

### View Results

**Point cloud:**

```python
import open3d as o3d
pcd = o3d.io.read_point_cloud("output/frame_000000/fused.ply")
o3d.visualization.draw_geometries([pcd])
```

**Mesh:**

```python
mesh = o3d.io.read_triangle_mesh("output/frame_000000/surface.ply")
o3d.visualization.draw_geometries([mesh])
```

## Step 6: Export Mesh

Convert the reconstructed mesh to other formats:

### Single File Export

```bash
# Export to OBJ (widely supported, preserves colors)
aquamvs export-mesh output/frame_000000/surface.ply --format obj

# Export to STL with simplification (for 3D printing)
aquamvs export-mesh output/frame_000000/surface.ply --format stl --simplify 10000

# Export to GLB (compact, web-ready)
aquamvs export-mesh output/frame_000000/surface.ply --format glb
```

### Batch Export

Convert all meshes in a directory:

```bash
aquamvs export-mesh --input-dir output/ --format obj --output-dir meshes/
```

**Parameters:**

- `--format`: Output format (`obj`, `stl`, `gltf`, `glb`)
- `--simplify`: (Optional) Target face count for mesh simplification
- `--input-dir`: Batch mode — convert all `.ply` files in directory
- `--output-dir`: (Batch mode) Output directory (defaults to `--input-dir`)

## Configuration Tips

Edit `config.yaml` to customize reconstruction:

### Switch Matcher Type

```yaml
sparse_matching:
  matcher_type: lightglue  # Fast, sparse features (default)
  # matcher_type: roma      # Slower, dense correspondences, higher accuracy
```

### Adjust Depth Range

Focus reconstruction on a specific depth range (in meters, ray depth):

```yaml
reconstruction:
  depth_min: 0.5
  depth_max: 2.0
```

**How to determine range:** Run once with defaults, inspect depth map statistics, then narrow range for second pass.

### GPU vs. CPU

```yaml
runtime:
  device: cuda  # Use GPU (requires CUDA-capable GPU)
  # device: cpu  # CPU-only (slower, but works everywhere)
```

### Quality vs. Speed

Increase depth hypotheses for higher quality (slower):

```yaml
reconstruction:
  num_depth_hypotheses: 128  # Default: 64, higher = better quality, longer runtime
```

### Pipeline Mode

```yaml
reconstruction:
  pipeline_mode: full    # Dense stereo (default, best quality)
  # pipeline_mode: sparse  # Sparse reconstruction (faster, lower quality)
```

**Sparse mode:** Uses only feature matches (no plane sweep stereo). Faster but produces sparser point clouds.

**Full mode:** Dense depth estimation for each camera. Slower but produces complete surface reconstructions.

### Multi-Frame Processing

Process a frame range:

```yaml
preprocessing:
  frame_start: 0
  frame_stop: 100   # Process frames 0-99
  frame_step: 10    # Every 10th frame
```

Leave `frame_stop: null` to process all frames.

### Output Control

Enable/disable intermediate outputs:

```yaml
runtime:
  save_depth_maps: true       # Per-camera depth maps
  save_sparse_cloud: true     # Sparse triangulated cloud
  save_fused_cloud: true      # Fused point cloud
  save_mesh: true             # Final mesh
```

Disable unnecessary outputs to save disk space.

## Benchmarking

Compare different feature extractor configurations on a single frame:

```bash
aquamvs benchmark config.yaml --frame 0
```

This runs reconstruction with multiple matcher/detector combinations and generates a comparison report in `output/benchmark/`.

## Preprocessing

Apply temporal median filtering to remove fish/debris from underwater video:

```bash
aquamvs preprocess input_video.mp4 --output-dir filtered/ --window 30
```

**Parameters:**

- `input`: Video file or directory of videos
- `--output-dir`: Output directory
- `--window`: Median window size in frames (default: 30)
- `--framestep`: Output every Nth frame (default: 1)
- `--format`: Output format (`png` or `mp4`, default: `png`)

The median filter removes moving objects (fish, particles) while preserving static structure (water surface).

## See Also

- [Python API Tutorial](tutorial/index.rst): Programmatic workflow using the `Pipeline` class
- [Theory](theory/index.rst): Understand the refractive geometry and algorithms
- [API Reference](api/index.rst): Detailed documentation of all modules and functions

## Troubleshooting

**Issue:** `No cameras matched`

**Solution:** Check your regex pattern. The first capture group must extract the camera name. Test with: `python -c "import re; print(re.match(r'^([a-z0-9]+)-', 'e3v82e0-cam1.mp4').group(1))"`

---

**Issue:** `ModuleNotFoundError: No module named 'torch'`

**Solution:** Install PyTorch first: `pip install torch --index-url https://download.pytorch.org/whl/cu121` (adjust CUDA version)

---

**Issue:** Pipeline crashes with CUDA out of memory

**Solution:** Switch to CPU (`--device cpu`) or reduce resolution/depth hypotheses in config

---

**Issue:** Depth maps are mostly NaN

**Solution:** Check depth range — adjust `reconstruction.depth_min` and `depth_max` to match your scene. Inspect sparse cloud depth statistics with `o3d.io.read_point_cloud("output/frame_000000/sparse.ply")`.
