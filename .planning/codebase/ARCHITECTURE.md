# Architecture

**Analysis Date:** 2026-02-14

## Pattern Overview

**Overall:** Layered pipeline with modular reconstruction stages

**Key Characteristics:**
- Declarative configuration-driven execution (YAML-based PipelineConfig)
- Strict separation between geometric models, feature extraction, dense matching, and fusion
- PyTorch-first with device-agnostic tensor operations
- Protocol-based abstraction for projection models enabling swappable geometric backends
- Multi-path execution: sparse (triangulation-based) and dense (depth map-based) reconstruction pipelines

## Layers

**Configuration Layer:**
- Purpose: Declare pipeline parameters and control flow without code changes
- Location: `src/aquamvs/config.py`
- Contains: Dataclasses for frame sampling, feature extraction, matching, dense stereo, fusion, surface reconstruction, and visualization
- Depends on: YAML serialization
- Used by: Pipeline orchestrator, all stage modules

**Projection & Geometry Layer:**
- Purpose: Abstract geometric transformations and ray-based 3D operations
- Location: `src/aquamvs/projection/` (protocol.py, refractive.py)
- Contains: ProjectionModel protocol, RefractiveProjectionModel implementation handling Snell's law
- Depends on: PyTorch for tensor operations, calibration data from AquaCal
- Used by: Triangulation, dense stereo, fusion modules

**Feature Extraction & Matching Layer:**
- Purpose: Extract sparse correspondences between images for triangulation
- Location: `src/aquamvs/features/` (extraction.py, matching.py, roma.py)
- Contains: LightGlue-based matching, SuperPoint/ALIKED/DISK extractors, RoMa v2 dense matcher, pair selection strategy
- Depends on: kornia for image preprocessing, external model files for neural networks
- Used by: Sparse pipeline path, feature visualization

**Triangulation Layer:**
- Purpose: Convert feature correspondences to 3D points via ray intersection
- Location: `src/aquamvs/triangulation.py`
- Contains: Ray triangulation (least-squares solution), depth range estimation
- Depends on: ProjectionModel.cast_ray() for ray generation
- Used by: Sparse reconstruction path

**Dense Stereo Layer:**
- Purpose: Compute per-pixel depth maps via plane-sweep or dense warping
- Location: `src/aquamvs/dense/` (plane_sweep.py, cost.py, roma_depth.py)
- Contains: Cost volume construction (NCC/SSIM metrics), depth extraction, RoMa warp aggregation
- Depends on: ProjectionModel for depth hypothesis reprojection, image warping via kornia
- Used by: Dense reconstruction path (full and sparse modes)

**Fusion Layer:**
- Purpose: Merge multiple depth maps into unified 3D point clouds with confidence weighting
- Location: `src/aquamvs/fusion.py`
- Contains: Geometric consistency filtering, depth map blending, point cloud backprojection
- Depends on: ProjectionModel for reprojection, Open3D for point cloud operations
- Used by: Final 3D reconstruction

**Surface Reconstruction Layer:**
- Purpose: Convert point clouds to watertight triangle meshes
- Location: `src/aquamvs/surface.py`
- Contains: Poisson reconstruction, ball-pivoting, height-field meshing
- Depends on: Open3D for geometric algorithms
- Used by: Mesh export

**Coloring Layer:**
- Purpose: Assign per-vertex colors by selecting best-view camera projection
- Location: `src/aquamvs/coloring.py`
- Contains: Best-view selection algorithm (angle between ray and surface normal)
- Depends on: Projection models, undistorted images, surface normals
- Used by: Sparse cloud and mesh vertex coloring

**Calibration & Undistortion Layer:**
- Purpose: Load AquaCal metadata and precompute undistortion maps
- Location: `src/aquamvs/calibration.py`
- Contains: CalibrationData, CameraData, undistortion map caching
- Depends on: AquaCal JSON format (local editable dependency)
- Used by: Pipeline setup initialization

**Visualization Layer:**
- Purpose: Generate diagnostic images and 3D renders for result inspection
- Location: `src/aquamvs/visualization/` (depth.py, features.py, scene.py, rig.py, summary.py)
- Contains: Depth/confidence colormaps, feature overlay drawing, 3D mesh/point cloud rendering, rig diagram generation
- Depends on: matplotlib, Open3D for rendering
- Used by: Optional visualization output (gated by VizConfig)

**Evaluation Layer:**
- Purpose: Compute metrics for benchmark comparisons (cross-validation against ground truth)
- Location: `src/aquamvs/evaluation/` (alignment.py, metrics.py)
- Contains: ICP alignment, cloud-to-cloud distance, height map difference, reprojection error
- Depends on: Open3D for point cloud algorithms, scipy for interpolation
- Used by: Benchmark suite, manual evaluation scripts

**Benchmark Layer:**
- Purpose: Sweep over feature extraction/matching configurations and compare results
- Location: `src/aquamvs/benchmark/` (runner.py, metrics.py, report.py, visualization.py)
- Contains: Parameter sweep orchestration, result aggregation, HTML report generation
- Depends on: All core pipeline modules
- Used by: CLI benchmark command

**Pipeline Orchestrator:**
- Purpose: Coordinate all stages for end-to-end reconstruction
- Location: `src/aquamvs/pipeline.py` (PipelineContext, setup_pipeline, process_frame, run_pipeline)
- Contains: Frame-level state management, stage sequencing, output directory structure
- Depends on: AquaCal VideoSet for frame iteration, all stage modules
- Used by: CLI, batch processing workflows

**CLI & I/O:**
- Purpose: Command-line interface and file I/O operations
- Location: `src/aquamvs/cli.py`, module-level functions for save/load
- Contains: Config initialization, reference image export, pipeline execution, benchmarking
- Depends on: argparse, pathlib, all pipeline modules
- Used by: External tools, user scripts

## Data Flow

**Setup Phase (once per video session):**

1. Load calibration JSON (AquaCal format) → CalibrationData
2. Compute undistortion maps for all cameras → UndistortionData
3. Create RefractiveProjectionModel instances (K, R, t, water_z, refractive params)
4. Select camera pairs based on proximity and include_center flag → dict[str, list[str]]
5. Load ROI masks if provided → dict[str, np.ndarray]
6. Return PipelineContext (immutable across all frames)

**Per-Frame Processing (variable path based on matcher_type + pipeline_mode):**

### Sparse Path (LightGlue + Triangulation):
1. Undistort images → apply color normalization (optional)
2. Extract features per image (SuperPoint/ALIKED/DISK) → keypoints + descriptors
3. Apply ROI masks to feature sets (prune masked-out keypoints)
4. Match all pairs via LightGlue → correspondences with match confidence
5. Save matches (optional)
6. Visualize features (optional)
7. Triangulate rays from each pair → sparse cloud (N, 3)
8. Filter sparse cloud (remove points below water surface)
9. Estimate depth ranges from sparse cloud extrema
10. Run plane-sweep dense stereo per ring camera → depth maps
11. Apply mask to depth maps
12. Visualize depth maps (optional)
13. Filter depth maps via geometric consistency across cameras
14. Fuse depth maps (backproject, voxel grid, then extract points)
15. Reconstruct surface (Poisson/BPA/heightfield)
16. Color vertices via best-view selection
17. Save point cloud and mesh
18. Render 3D scene and rig diagram (optional)
19. Return (frame_idx, sparse_cloud, point_cloud, mesh)

### Dense Path - RoMa Full (RoMa Dense Matching):
1. Undistort images → apply color normalization (optional)
2. Run RoMa v2 matching for all pairs → dense warp fields
3. Convert warps to depth maps per ring camera via upsampling + refraction
4. Apply mask to depth maps
5. Save depth maps (optional)
6. Visualize depth maps (optional)
7. **Skip** geometric consistency filtering (already enforced by warp aggregation)
8. Fuse depth maps (backproject, voxel grid, then extract points)
9. Reconstruct surface (Poisson/BPA/heightfield)
10. Color vertices via best-view selection
11. Save point cloud and mesh
12. Render 3D scene and rig diagram (optional)
13. Return (frame_idx, point_cloud, mesh)

### Dense Path - RoMa Sparse (RoMa Correspondences):
1. Undistort images → apply color normalization (optional)
2. Run RoMa v2 matching (correspondence extraction, not dense warps)
3. Convert RoMa correspondences to LightGlue-compatible match format
4. Save matches (optional)
5. Triangulate rays from each pair → sparse cloud
6. Filter sparse cloud
7. Estimate depth ranges
8. Run plane-sweep dense stereo
9. Apply masks, visualize
10. **Apply** geometric consistency filtering (not done by RoMa in sparse mode)
11. Fuse depth maps
12. Reconstruct surface
13. Color vertices
14. Save outputs
15. Visualize (optional)

**Post-Video Summary (if enabled):**

1. Collect fused point clouds from all frame output dirs
2. Grid each into height map
3. Render timeseries gallery (height maps across frames)

**State Management:**

- **Immutable across frames:** PipelineContext (config, calibration, projection models, pairs, masks)
- **Per-frame outputs:** Frame directory structure with depth maps, sparse cloud, point cloud, mesh, visualizations
- **Device placement:** All tensors follow input device (CPU unless config specifies GPU)

## Key Abstractions

**ProjectionModel (Protocol):**
- Purpose: Define interface for geometric projection without implementation coupling
- Examples: `src/aquamvs/projection/refractive.py` (RefractiveProjectionModel)
- Pattern: Runtime checkable protocol with two methods: project() and cast_ray()
- Used by: All geometry-dependent modules (triangulation, dense stereo, fusion) for abstraction

**PipelineContext (Dataclass):**
- Purpose: Encapsulate all frame-invariant state to avoid parameter passing
- Examples: config, calibration, projection_models, pairs, masks
- Pattern: Initialized once by setup_pipeline(), passed to each process_frame() call
- Benefit: Clean separation of one-time setup from per-frame logic

**DepthMap Representation:**
- Purpose: (H, W) float32 tensor with NaN for invalid pixels, separate confidence map (H, W)
- Pattern: Always saved as (depth, confidence) pairs in .npz files
- Semantics: Depth is ray depth (distance along ray from water surface intersection), not Z coordinate

**Match Dict Structure:**
```python
{
    "keypoints_ref": (M, 2),      # Pixel coords in reference
    "keypoints_src": (M, 2),      # Pixel coords in source
    "confidence": (M,),            # Match scores [0, 1]
}
```

**Feature Dict Structure:**
```python
{
    "keypoints": (K, 2),           # Pixel coords
    "descriptors": (K, D),         # Feature vectors
    "scores": (K,),                # Detection scores
}
```

## Entry Points

**CLI Entry Point:**
- Location: `src/aquamvs/cli.py` (main function)
- Triggers: `aquamvs init|run|export-refs|benchmark` command
- Responsibilities: Argument parsing, config management, command dispatch

**Pipeline Entry Point:**
- Location: `src/aquamvs/pipeline.py` (run_pipeline function)
- Triggers: Called by CLI run command with PipelineConfig
- Responsibilities: One-time setup, video frame iteration, per-frame processing, error handling

**Frame Entry Point:**
- Location: `src/aquamvs/pipeline.py` (process_frame function)
- Triggers: Called for each frame from run_pipeline loop
- Responsibilities: All 9 pipeline stages for a single frame

## Error Handling

**Strategy:** Failed projections and invalid depth pixels return validity masks rather than exceptions

**Patterns:**
- ProjectionModel.project() returns (pixels, valid) mask → callers skip invalid entries
- ProjectionModel.cast_ray() returns (origins, directions) for all pixels, validity implicit in coordinates
- Triangulation: If ray system is degenerate, raises ValueError (caller must catch and retry)
- Dense stereo: Invalid depth pixels set to NaN, confidence = 0
- Fusion: NaN depth maps automatically handled (backprojection skips NaN pixels)
- Pipeline: process_frame() wrapped in try/except, logs exception and continues to next frame
- Config validation: PipelineConfig.validate() raises ValueError for invalid settings

## Cross-Cutting Concerns

**Logging:** Python standard logging module, per-module loggers (`logger = logging.getLogger(__name__)`)

**Device Handling:** Low-level math modules (projection, triangulation) follow input tensor device; pipeline modules receive device from config

**Tensor Precision:** Float32 throughout (matching PyTorch defaults and single-precision GPU efficiency)

**Batch Operations:** Vectorized where performance-critical (triangulation, fusion backprojection); per-image loops acceptable for feature extraction/matching

**Dependency Direction:** Core math → Geometry → Features → Dense → Fusion → Surface; no circular dependencies

---

*Architecture analysis: 2026-02-14*
