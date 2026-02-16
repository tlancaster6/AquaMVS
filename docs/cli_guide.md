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
  --input-dir project_data/videos \
  --pattern "^([a-z0-9]+)-" \
  --calibration project_data/calibration.json \
  --output-dir ./output
```

**Parameters:**

- `--input-dir`: Directory containing video files or camera subdirectories with images
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

The output directory contains frame-wise results. Which files are present depends on the matcher type (`roma` vs `lightglue`) and pipeline mode (`full` vs `sparse`).

```
output/
├── config.yaml                          # Copy of the config used for this run
├── frame_000000/
│   ├── sparse/                          # LightGlue and RoMa sparse only
│   │   └── sparse_cloud.pt
│   ├── features/                        # Optional (runtime.save_features)
│   │   ├── {camera}.pt                  #   LightGlue only: per-camera keypoints
│   │   └── {ref}_{src}.pt              #   Per-pair matches
│   ├── depth_maps/                      # Full mode only (always saved)
│   │   └── {camera}.npz
│   ├── consistency_maps/                # Full mode, LightGlue only (runtime.save_consistency_maps)
│   │   ├── {camera}.npz
│   │   └── {camera}.png
│   ├── point_cloud/
│   │   ├── fused.ply                    #   Full mode (always saved)
│   │   └── sparse.ply                   #   Sparse mode (always saved)
│   ├── mesh/                            # Always saved
│   │   └── surface.ply
│   └── viz/                             # runtime.viz_enabled + viz_stages
│       ├── depth_{camera}.png
│       ├── confidence_{camera}.png
│       ├── sparse_{camera}.png          #   LightGlue only (viz_stages: features)
│       ├── matches_{ref}_{src}.png      #   LightGlue only (viz_stages: features)
│       ├── fused_top.png                #   viz_stages: scene
│       ├── fused_oblique.png
│       ├── fused_side.png
│       ├── mesh_top.png
│       ├── mesh_oblique.png
│       ├── mesh_side.png
│       └── rig.png                      #   viz_stages: rig
├── frame_000010/
│   └── ...
└── summary/                             # viz_stages: summary (multi-frame runs)
    └── timeseries_gallery.png
```

### Data outputs

**`sparse/sparse_cloud.pt`** -- Triangulated 3D points from feature correspondences, stored as PyTorch tensors with `points_3d` and `scores` keys. Used internally to derive depth ranges for plane sweep stereo. *LightGlue and RoMa sparse mode only* (RoMa full mode skips triangulation).

**`features/{camera}.pt`** and **`features/{ref}_{src}.pt`** -- Per-camera keypoints/descriptors and per-pair match correspondences. Useful for debugging feature quality. LightGlue saves both per-camera and per-pair files; RoMa sparse saves per-pair only; RoMa full does not save features. *Config: `runtime.save_features` (default: off).*

**`depth_maps/{camera}.npz`** -- Per-camera depth and confidence maps (ray depth in meters, float32). Each `.npz` contains `depth` (H x W) and `confidence` (H x W) arrays. *Full mode only. Always saved. Set `runtime.keep_intermediates: false` to delete after fusion.*

**`consistency_maps/{camera}.npz`** and **`{camera}.png`** -- Number of source cameras that agree on the depth at each pixel, saved as both raw counts (`.npz`) and colormapped visualization (`.png`). Only produced by the geometric consistency filter. *Full mode, LightGlue only* (RoMa full mode skips consistency filtering). *Config: `runtime.save_consistency_maps` (default: off).*

**`point_cloud/fused.ply`** -- Fused point cloud merged from all cameras, with vertex colors. Produced in full mode after depth map fusion and outlier removal. *Always saved.*

**`point_cloud/sparse.ply`** -- Colored sparse point cloud downsampled from triangulated correspondences. Produced in sparse mode. *Always saved.*

**`mesh/surface.ply`** -- Reconstructed triangle mesh with vertex colors (Poisson reconstruction by default). *Always saved.*

### Visualizations

All visualizations require `runtime.viz_enabled: true` (default: off). The `runtime.viz_stages` list controls which groups are generated; leave it empty to enable all stages.

**`viz/depth_{camera}.png`** and **`viz/confidence_{camera}.png`** -- Colormapped depth and confidence maps for each ring camera. *Stage: `depth`. Full mode only.*

**`viz/sparse_{camera}.png`** and **`viz/matches_{ref}_{src}.png`** -- Keypoint overlays on undistorted images and side-by-side match visualizations. *Stage: `features`. LightGlue only.*

**`viz/fused_top.png`**, **`fused_oblique.png`**, **`fused_side.png`**, **`mesh_top.png`**, **`mesh_oblique.png`**, **`mesh_side.png`** -- Point cloud and mesh rendered from three canonical viewpoints (top-down, 45-degree oblique, side). *Stage: `scene`.*

**`viz/rig.png`** -- Camera rig diagram showing camera frustums and the water plane, optionally overlaid with the fused point cloud. *Stage: `rig`.*

**`summary/timeseries_gallery.png`** -- Grid gallery of height maps across all processed frames. Only generated for multi-frame runs. *Stage: `summary`.*

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

Use a quality preset for one-line tuning:

```yaml
quality_preset: fast       # Fewer depth planes, larger batches — quickest results
# quality_preset: balanced # Default tradeoffs
# quality_preset: quality  # Maximum depth planes, smallest batches — best accuracy
```

Or manually increase depth hypotheses for higher quality (slower):

```yaml
reconstruction:
  num_depth_hypotheses: 128  # Default: 64, higher = better quality, longer runtime
  depth_batch_size: 8        # Process depth planes in batches (GPU speedup)
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

Depth maps, point clouds, and meshes are always saved. Control other outputs:

```yaml
runtime:
  save_features: false          # Per-camera features and matches (default: off)
  save_consistency_maps: false  # Consistency maps (default: off)
  keep_intermediates: true      # Keep depth maps after fusion (default: on)
```

## Benchmarking

Compare different feature extractor configurations on a single frame:

```bash
aquamvs benchmark config.yaml --frame 0
```

This runs reconstruction with multiple matcher/detector combinations and generates a comparison report in `output/benchmark/`.

## Temporal Filtering

Apply temporal median filtering to remove fish/debris from underwater video:

```bash
aquamvs temporal-filter input_video.mp4 --output-dir filtered/ --window 30
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
