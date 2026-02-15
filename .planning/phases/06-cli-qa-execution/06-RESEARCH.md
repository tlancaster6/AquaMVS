# Phase 6: CLI QA Execution - Research

**Researched:** 2026-02-15
**Domain:** Manual QA testing of CLI application with real multi-camera datasets
**Confidence:** HIGH

## Summary

Phase 6 focuses on hands-on QA execution of AquaMVS's 7 CLI commands and programmatic Pipeline API using real 13-camera underwater capture data. This is NOT test automation — it's exploratory manual testing to verify commands work end-to-end, produce expected outputs, and identify any remaining bugs before v1.0 release. The phase validates that all prior implementation work actually functions correctly with realistic datasets.

All infrastructure is already built (Phases 1-5 complete). The package includes 7 CLI commands (`init`, `run`, `export-refs`, `profile`, `benchmark`, `preprocess`, `export-mesh`), comprehensive Pydantic config validation, dual matcher pathways (LightGlue+sparse, RoMa+full), quality presets, and GPU support. Testing will use real 13-camera ring rig videos (10 minutes each) with AquaCal calibration data, following a workflow that mirrors production usage: preprocess videos → init config → export reference images → (user creates masks) → run reconstruction → export mesh → profile performance.

**Primary recommendation:** Execute QA commands in dependency order (preprocess first, then init, etc.), commit after each command succeeds, log non-blocking issues to a tracking file for later, and fix blockers immediately in-flow.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Scope of commands:**
- All 7 CLI commands in scope: init, run, export-refs, profile, benchmark, preprocess, export-mesh
- Programmatic API also tested: `Pipeline(config).run()` basic smoke test
- Pass criteria: command completes successfully AND output is visually/manually reviewed
- Help text spot-checked only if something looks off, not systematically

**Test data strategy:**
- Real multi-camera captures with AquaCal calibration data
- 13 cameras, 10-minute videos each
- `aquamvs preprocess` extracts 5 temporal median frames per camera (1 per 2 minutes) → 5 frame sets of 13 images
- Preprocessed image directories are primary input for `aquamvs run`
- Video input tested minimally: one lightglue sparse run to verify video path works
- QA execution order: preprocess → init → export-refs (user creates masks externally) → run (lightglue+full) → run (roma+full) → benchmark --compare → export-mesh → profile
- Commits: one commit per completed CLI command QA

**Issue handling:**
- Blockers and quick fixes: fix inline immediately, then continue
- Large non-blockers: log to QA markdown file in .planning/ for later
- Tracking: conversational, no formal pass/fail report

**Coverage depth:**
- Happy path focus — skip error handling / bad input testing
- GPU (CUDA) for all testing
- 2 main execution paths for `aquamvs run`: lightglue+full and roma+full
- All mesh export formats tested: OBJ, STL, GLTF + simplification
- Skip --quiet flag and progress bar UX testing

### Claude's Discretion

- Order of mesh format testing
- Whether to test mesh simplification as separate step or combined with format export
- Profiler test approach (synthetic vs pipeline data)

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope

</user_constraints>

## Standard Stack

### Core (Already Implemented)

| Component | Version/Type | Purpose | Status |
|-----------|--------------|---------|--------|
| argparse | stdlib | CLI argument parsing | ✓ Implemented in cli.py |
| Pydantic | 2.12.0+ | Config validation | ✓ PipelineConfig validates on load |
| tqdm | 4.66.0+ | Progress bars | ✓ Integrated in pipeline stages |
| PyTorch | 2.0+ | GPU compute | ✓ Device support in config.runtime.device |
| AquaCal | 0.1.0+ | Video I/O, calibration loading | ✓ VideoSet used by runner |
| Open3D | 0.18.0+ | Mesh I/O, surface reconstruction | ✓ Used in surface.py |

### Testing Infrastructure (Already Built)

| Component | Purpose | Location |
|-----------|---------|----------|
| pytest | Unit/integration tests | tests/ (445 passing, 109 skipped) |
| CI/CD | GitHub Actions test matrix | .github/workflows/test.yml |
| Benchmark suite | Synthetic accuracy tests | src/aquamvs/benchmark/ |
| Profiler | Performance measurement | src/aquamvs/profiling/ |

### Manual QA Tools (What's Needed)

| Tool | Purpose | Notes |
|------|---------|-------|
| Windows Explorer / ls | File inspection | Verify output directories created, files present |
| Text editor | Config editing | YAML syntax, path adjustment |
| Image viewer | Visual inspection | Reference images, consistency maps, depth colormaps |
| MeshLab / 3D viewer | Mesh validation | OBJ/STL/GLTF rendering, triangle counts |
| GPU monitor (nvidia-smi) | Resource tracking | VRAM usage, CUDA utilization |
| Git | Commit after each command | One commit per CLI command QA pass |

**Installation:**
Already installed — package built in Phases 1-5.

## Architecture Patterns

### Recommended QA Workflow Structure

```
.planning/
└── qa/
    ├── qa-session-log.md        # Conversational log of QA session
    ├── issues-found.md           # Non-blocking issues for later
    └── outputs/                  # Sample outputs for documentation
        ├── depth_map_frame_000000.png
        ├── sparse_cloud_frame_000000.ply
        └── mesh_frame_000000.obj
```

### Pattern 1: Command Execution Flow

**What:** Test CLI commands in dependency order, verifying outputs before proceeding to next command.

**When to use:** All QA phases where commands depend on prior outputs (e.g., `export-refs` requires config from `init`).

**Example workflow:**
```bash
# 1. Preprocess videos to extract temporal median frames
aquamvs preprocess /path/to/videos --output-dir ./preprocessed --window 30 --framestep 120 --format png

# Verify: Check ./preprocessed/ has 5 frame directories (frame_000000, frame_000120, ...), each with 13 PNG images
ls ./preprocessed/frame_000000/*.png  # Should show 13 images

# 2. Initialize config from preprocessed images
aquamvs init --video-dir ./preprocessed/frame_000000 --pattern "^([a-z0-9]+)\.png$" \
             --calibration /path/to/calibration.json --output-dir ./output

# Verify: Check config.yaml exists, open in editor to review paths and camera mapping

# 3. Export reference images for mask creation
aquamvs export-refs config.yaml --frame 0

# Verify: Check ./output/reference_images/ has 13 undistorted PNGs, visually inspect for distortion artifacts

# 4. (Manual step) User creates ROI masks in external image editor, saves to ./masks/

# 5. Edit config.yaml to add mask_dir: ./masks

# 6. Run reconstruction (LightGlue + full mode)
aquamvs run config.yaml --device cuda

# Verify: Check ./output/frame_000000/ for depth maps, sparse clouds, fused clouds, meshes
# Visually inspect depth_map_*.png colormaps, open mesh_*.ply in MeshLab

# 7. Run reconstruction (RoMa + full mode)
# Edit config.yaml: matcher_type: roma
aquamvs run config_roma.yaml --device cuda

# 8. Compare runs
aquamvs benchmark --compare ./output_lightglue ./output_roma

# 9. Export mesh to multiple formats
aquamvs export-mesh ./output/frame_000000/mesh_*.ply --format obj
aquamvs export-mesh ./output/frame_000000/mesh_*.ply --format stl
aquamvs export-mesh ./output/frame_000000/mesh_*.ply --format gltf --simplify 50000

# 10. Profile performance
aquamvs profile config.yaml --frame 0 --output-dir ./profiling
```

### Pattern 2: Visual Output Validation

**What:** Manual inspection of generated outputs to verify quality (not just existence).

**When to use:** After every stage that produces visual artifacts (depth maps, point clouds, meshes).

**Validation checklist:**
- **Depth maps:** Smooth gradients, no NaN holes outside ROI, depth range plausible (~1-2m for underwater surface)
- **Consistency maps:** High confidence (>0.5) in majority of pixels, low confidence at occlusion boundaries
- **Sparse clouds:** 3D points cluster near expected surface height (Z ~ water_z + depth), no extreme outliers
- **Fused clouds:** Denser than sparse, outlier removal effective (no floating artifacts)
- **Meshes:** Surface continuity (no disconnected patches), appropriate triangle count, no inverted normals

### Pattern 3: Issue Triage During Execution

**What:** Classify bugs as blockers (fix now) vs non-blockers (log for later) and continue QA flow.

**When to use:** When encountering unexpected behavior during manual testing.

**Decision tree:**
```
Does the command complete without crashing?
├─ NO → BLOCKER: Fix immediately, command must complete
└─ YES → Does output exist and pass basic sanity checks?
    ├─ NO → BLOCKER: Fix immediately, output must be produced
    └─ YES → Is there a quality/UX issue?
        ├─ YES → NON-BLOCKER: Log to issues-found.md, continue QA
        └─ NO → PASS: Commit and proceed to next command
```

**Example blocker:** `aquamvs run` crashes with "CUDA out of memory" → Fix by adjusting batch size or image resolution
**Example non-blocker:** Progress bar text formatting looks awkward → Log to issues-found.md, continue testing

### Anti-Patterns to Avoid

- **Premature optimization:** Don't tune performance during QA (that's what profiler is for)
- **Automated test writing:** This is manual exploratory QA, not test automation (pytest suite already exists)
- **Perfect output chasing:** Accept "good enough" quality if workflow succeeds (v1.0 goal is functional, not publication-ready)
- **Exhaustive error testing:** Don't test bad inputs, missing files, malformed configs (happy path only per user constraints)

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CLI help text generation | Custom --help formatter | argparse built-in help | argparse auto-generates help from parser config, handles formatting and wrapping |
| Output file validation | Manual file existence checks | pathlib.Path.exists() + basic assertions | Built-in methods are sufficient for smoke testing file creation |
| Visual 3D inspection | Custom mesh renderer | MeshLab / Open3D viewer | Existing tools handle all mesh formats, interactive viewing |
| Image comparison | Pixel-by-pixel diff | Human visual inspection | Manual QA goal is UX validation, not pixel-perfect regression testing |
| Performance profiling | Custom timing code | aquamvs profile CLI (already built) | PipelineProfiler with torch.profiler integration already implemented in Phase 5 |

**Key insight:** This phase is about USING existing tools (both AquaMVS commands and external viewers), not building new testing infrastructure. All scaffolding already exists from Phases 1-5.

## Common Pitfalls

### Pitfall 1: Package Import Failures Due to Missing Dependencies

**What goes wrong:** CLI commands fail with `ModuleNotFoundError` despite pip install.

**Why it happens:** AquaMVS has complex dependency chain (PyTorch, kornia, LightGlue, RoMa v2, AquaCal). AquaCal may have transitive dependencies (e.g., `natsort`) not declared in pyproject.toml.

**How to avoid:**
1. Create fresh virtual environment for QA
2. Install with all extras: `pip install -e ".[dev]"`
3. Install AquaCal in editable mode if using local dev version: `pip install -e ../AquaCal`
4. Test import before CLI: `python -c "import aquamvs; print('OK')"`

**Warning signs:**
- Import errors mentioning AquaCal submodules
- Missing CUDA libraries if running GPU tests without CUDA installed
- kornia/Open3D version conflicts

**Current evidence:** During research, `python -m aquamvs.cli` failed with `ModuleNotFoundError: No module named 'natsort'` (AquaCal dependency). Must resolve before QA execution.

### Pitfall 2: CUDA Out of Memory on GPU

**What goes wrong:** `aquamvs run` crashes mid-pipeline with CUDA OOM error.

**Why it happens:** Multi-camera plane sweep with 128+ depth hypotheses can exceed VRAM (especially with 1920x1080 images). Default `depth_batch_size=4` may be too aggressive for GPU with <8GB VRAM.

**How to avoid:**
1. Check GPU memory before starting: `nvidia-smi`
2. Use BALANCED quality preset (not QUALITY) for initial testing
3. If OOM occurs, adjust config: `depth_batch_size: 1` or `num_depths: 64`
4. Alternatively, reduce image resolution in preprocessing

**Warning signs:**
- GPU memory usage climbing during depth_estimation stage
- Pipeline stalls at "Computing depth maps" progress bar
- Sudden crash with torch.cuda.OutOfMemoryError

**Mitigation:** Quality presets (FAST/BALANCED/QUALITY) added in Phase 5 specifically to handle this. Use `QualityPreset.FAST` for 4GB GPUs.

### Pitfall 3: Empty or Invalid Output Directories

**What goes wrong:** Pipeline completes but output directory is missing files or has zero-byte files.

**Why it happens:**
- Config paths have Windows/Unix path separator mismatch
- Output directory permissions issue
- Early exit due to no valid cameras (all images None)
- Mask directory path typo causing ROI filtering to reject all pixels

**How to avoid:**
1. Use absolute paths in config.yaml (not relative)
2. Check output_dir exists and is writable before starting
3. Verify camera_video_map keys match calibration camera names exactly
4. If using masks, verify mask filenames match camera names: `{camera_name}.png`

**Warning signs:**
- "No valid cameras found" warning in logs
- Empty frame_000000/ directories
- Depth maps exist but are all NaN/black

**Detection:** After each command, immediately run `ls -lR output_dir/` to verify non-empty files created.

### Pitfall 4: Config Schema Mismatches from Old Examples

**What goes wrong:** Example configs from documentation fail Pydantic validation.

**Why it happens:** Phase 2/3 refactored config from flat structure to nested Pydantic models (PreprocessingConfig, SparseMatchingConfig, ReconstructionConfig, RuntimeConfig). Old examples may reference deprecated fields.

**How to avoid:**
1. Use `aquamvs init` to generate fresh config (guaranteed valid schema)
2. If manually editing, refer to src/aquamvs/config.py for current structure
3. Pydantic errors are clear: read validation message for field path

**Warning signs:**
- `ValidationError` when loading config
- Error messages mentioning `config.device.device` (old path) instead of `config.runtime.device` (new path)

**Example old vs new:**
```yaml
# OLD (Phase 1, will fail)
device:
  device: cuda
visualization:
  enabled: true

# NEW (Phase 2+, current)
runtime:
  device: cuda
  viz_enabled: true
```

**Resolution:** Phase 2 added Pydantic validation that catches these at config load time with clear error messages. Let validation errors guide corrections.

### Pitfall 5: ROI Masks Not Applied (Silent Failure)

**What goes wrong:** Reconstruction processes entire image instead of just ROI, resulting in large meshes with background artifacts.

**Why it happens:**
- `mask_dir` not specified in config
- Mask filenames don't match camera names
- Mask images are RGB instead of grayscale
- Mask path uses wrong separator (Windows backslash vs Unix forward slash)

**How to avoid:**
1. After `export-refs`, verify reference images look correct before creating masks
2. Save masks with exact camera names: `e3v82e0.png`, not `camera_01_mask.png`
3. Save masks as binary (0/255) grayscale PNG, not RGB
4. Use absolute path in config: `mask_dir: /full/path/to/masks`
5. Validate masks loaded: check logs for "Loaded mask for camera X"

**Warning signs:**
- Sparse cloud has points far from water surface (background objects)
- Mesh includes pool walls, equipment, or scene boundaries
- Depth maps show full image range instead of just water surface ROI

**Validation:** Visual inspection of sparse_cloud_*.ply in MeshLab — should only contain surface points, not background.

## Code Examples

These examples show how to inspect QA outputs and diagnose issues during manual testing.

### Verifying Preprocessing Output

```bash
# After: aquamvs preprocess /data/videos --output-dir ./preprocessed --window 30 --framestep 120

# Check temporal median frames exist (5 frames × 13 cameras = 65 images)
find ./preprocessed -name "*.png" | wc -l
# Expected: 65 (or 5 × N_cameras)

# Inspect first frame for a specific camera
ls ./preprocessed/frame_000000/
# Expected: e3v82e0.png, e3v8213.png, ..., e3v8237.png (13 files)

# Open image to verify temporal median worked (fish removed)
# Should show static underwater surface, no moving objects
```

### Inspecting Pipeline Outputs

```bash
# After: aquamvs run config.yaml

# Check frame output directory structure
tree ./output/frame_000000/
# Expected hierarchy:
# frame_000000/
# ├── depth_maps/
# │   ├── e3v82e0_depth.npz          # NumPy depth map
# │   ├── e3v82e0_depth.png          # Colormap visualization
# │   └── ...
# ├── sparse_cloud_frame_000000.ply  # Triangulated points
# ├── fused_cloud_frame_000000.ply   # Merged depth maps
# └── mesh_frame_000000.ply          # Poisson surface

# Verify depth map files are non-empty
ls -lh ./output/frame_000000/depth_maps/*.npz
# Each should be ~few hundred KB to few MB (not 0 bytes)

# Count points in sparse cloud
python -c "import open3d as o3d; pcd = o3d.io.read_point_cloud('./output/frame_000000/sparse_cloud_frame_000000.ply'); print(f'{len(pcd.points)} points')"
# Expected: thousands to tens of thousands (depends on scene complexity)
```

### Validating Mesh Quality

```python
# Mesh inspection script
import open3d as o3d
import numpy as np

mesh = o3d.io.read_triangle_mesh("./output/frame_000000/mesh_frame_000000.ply")
mesh.compute_vertex_normals()

# Basic statistics
print(f"Vertices: {len(mesh.vertices)}")
print(f"Triangles: {len(mesh.triangles)}")
print(f"Has normals: {mesh.has_vertex_normals()}")
print(f"Has colors: {mesh.has_vertex_colors()}")

# Check for degenerate triangles (area near zero)
vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)
tri_verts = vertices[triangles]
areas = 0.5 * np.linalg.norm(
    np.cross(tri_verts[:, 1] - tri_verts[:, 0], tri_verts[:, 2] - tri_verts[:, 0]),
    axis=1
)
print(f"Min triangle area: {areas.min():.6f}")
print(f"Degenerate triangles (<1e-6): {(areas < 1e-6).sum()}")

# Visualize (opens interactive viewer)
o3d.visualization.draw_geometries([mesh])
```

### Comparing Benchmark Runs

```bash
# After: aquamvs run config_lightglue.yaml && aquamvs run config_roma.yaml

# Compare two reconstruction runs
aquamvs benchmark --compare ./output_lightglue/benchmarks ./output_roma/benchmarks

# Expected output: Comparison table showing:
# - Mean/median geometric error
# - Completeness percentage
# - Runtime (if available)
# - Mesh triangle counts

# Example output format:
# Metric               | LightGlue | RoMa   | Delta
# ---------------------|-----------|--------|-------
# Mean Error (mm)      | 1.54      | 0.79   | -48.7%
# Completeness (%)     | 0.60      | 1.90   | +217%
# Runtime (s)          | 12.3      | 45.6   | +271%
```

### Profiler Output Inspection

```bash
# After: aquamvs profile config.yaml --frame 0 --output-dir ./profiling

# View profiling report
cat ./profiling/profile_report.txt
# Expected: Per-stage breakdown with timing and memory

# Example output:
# Stage                 | CPU Time (ms) | Memory (MB) | % Total
# ----------------------|---------------|-------------|--------
# undistortion          | 163.5         | 189.8       | 2.6%
# sparse_matching       | 11.5          | 2.0         | 0.2%
# depth_estimation      | 4158.6        | 545.8       | 67.1%  ← PRIMARY BOTTLENECK
# extract_depth         | 257.3         | 7.9         | 4.2%
# fusion                | 129.7         | 31.6        | 2.1%
# surface_reconstruction| 2058.2        | 2.2         | 33.2%

# Chrome trace (if available)
# Open ./profiling/profile_trace.json in chrome://tracing
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual test scripts | pytest-based unit/integration tests | Phase 1-3 | Automated regression detection (445 tests) |
| Flat config structure | Pydantic nested models | Phase 2 (Feb 2026) | Config validation at load time, clear error messages |
| Monolithic pipeline.py (995 lines) | Modular pipeline/ package | Phase 3 (Feb 2026) | Stages independently testable, clearer failure points |
| No profiling tools | PipelineProfiler + torch.profiler | Phase 5 (Feb 2026) | Bottleneck identification with measurements |
| Manual benchmark comparisons | Automated metrics + comparison | Phase 5 (Feb 2026) | Objective quality comparison across runs |

**Deprecated/outdated:**
- `config.device.device` → Use `config.runtime.device` (Phase 2 refactor)
- `config.visualization.enabled` → Use `config.runtime.viz_enabled` (Phase 2 refactor)
- Direct `from aquamvs.pipeline import run_pipeline` → Use `from aquamvs import Pipeline; Pipeline(config).run()` (Phase 3, though old import still works)
- Manual depth map NPZ inspection → Use saved PNG colormaps for visual inspection (Phase 2 feature)

## Open Questions

1. **Real Dataset Availability**
   - What we know: User has 13-camera, 10-minute videos with AquaCal calibration
   - What's unclear: Are videos already on disk in test environment? Or need to be transferred?
   - Recommendation: Confirm video location before starting QA session. If missing, identify smaller test dataset (1-minute clips) to avoid preprocessing delays.

2. **GPU Hardware Specifications**
   - What we know: CUDA testing required per user constraints
   - What's unclear: Available VRAM? (Determines quality preset choice)
   - Recommendation: Check `nvidia-smi` at session start. Use FAST preset if VRAM < 6GB, BALANCED if 6-10GB, QUALITY if 10GB+.

3. **Mask Creation Workflow**
   - What we know: User creates masks externally after `export-refs`
   - What's unclear: User's preferred image editor? Time estimate for mask creation?
   - Recommendation: Have user prepare masks in advance OR skip ROI masking for first QA pass (full-image reconstruction to unblock pipeline testing).

4. **Profiler Data Source (Claude's Discretion)**
   - What we know: Profiler can run on synthetic data (Phase 5 approach) or real pipeline data
   - What's unclear: User preference — quick synthetic profile vs longer real data profile?
   - Recommendation: Run profiler on real pipeline data (1 frame) to get production-realistic measurements. Synthetic profiling already done in Phase 5, real data adds validation.

5. **Benchmark Comparison Scope**
   - What we know: `--compare` flag compares two run directories (LightGlue vs RoMa)
   - What's unclear: Does comparison require ground truth data? Or just runtime/point count comparison?
   - Recommendation: Comparison is metadata-based (runtime, point counts, mesh stats), not accuracy-based (no ground truth needed). Proceed as planned.

## Sources

### Primary (HIGH confidence)

- AquaMVS codebase inspection:
  - `src/aquamvs/cli.py` - All 7 CLI commands implemented with argparse
  - `src/aquamvs/config.py` - Pydantic models with validation
  - `src/aquamvs/pipeline/` - Modular stage execution
  - `.planning/phases/05-performance-and-optimization/05-VERIFICATION.md` - Phase 5 completion proof

- CONTEXT.md (user decisions) - QA scope, execution order, issue handling strategy

### Secondary (MEDIUM confidence)

- [4 Techniques for Testing Python Command-Line (CLI) Apps – Real Python](https://realpython.com/python-cli-testing/)
- [How To Test CLI Applications With Pytest, Argparse And Typer | Pytest with Eric](https://pytest-with-eric.com/pytest-advanced/pytest-argparse-typer/)
- [Testing argparse Applications | PythonTest](https://pythontest.com/testing-argparse-apps/)

### Tertiary (LOW confidence)

- [Testing Python Data Science pipelines - DSSG Hitchhickers guide](https://dssg.github.io/hitchhikers-guide/curriculum/programming_best_practices/test-test-test/ds_testing/)
- [Building A Test Automation Pipeline for Data Science](https://technology.doximity.com/articles/building-a-test-automation-pipeline-for-data-science)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All tools already implemented and verified in Phases 1-5
- Architecture patterns: HIGH - QA workflow directly derived from CLI command structure and user-specified execution order
- Pitfalls: HIGH - Based on observed import errors, known GPU memory constraints from benchmark/profiling phase, and config refactoring history

**Research date:** 2026-02-15
**Valid until:** 2026-03-15 (30 days — stable QA workflow, unlikely to change)

**Notes:**
- This is a QA execution phase, not infrastructure building. All code already exists.
- Research focused on common CLI testing pitfalls and manual QA best practices rather than new library discovery.
- Confidence is HIGH because the system to be tested is fully documented in the codebase itself (Phases 1-5 complete, verified).
- Open questions are logistical (dataset location, GPU specs) not technical (methodology is clear).
