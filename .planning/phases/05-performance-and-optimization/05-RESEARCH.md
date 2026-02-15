# Phase 5: Performance and Optimization - Research

**Researched:** 2026-02-15
**Domain:** Performance profiling, benchmarking infrastructure, and optimization
**Confidence:** HIGH

## Summary

This phase implements comprehensive benchmarking infrastructure to evaluate reconstruction accuracy across different configurations (RoMa vs LightGlue, CLAHE on/off, surface reconstruction methods), performs runtime and memory profiling to identify bottlenecks, and implements optimizations targeting measured performance issues. The research reveals that plane sweep stereo (cost volume construction via grid_sample) is the likely primary bottleneck, with optimization opportunities in batching, memory layout, and depth sampling strategies.

The standard stack centers on PyTorch's built-in torch.profiler for profiling, pytest-based custom benchmarks for accuracy evaluation (not asv/pytest-benchmark due to custom metrics), tabulate for terminal output, and matplotlib for visualization. Synthetic ground truth generation uses procedural mesh creation with Open3D's rendering or simple analytic surfaces, avoiding heavyweight synthetic dataset toolkits.

**Primary recommendation:** Use torch.profiler with profile_memory=True for both CPU and GPU profiling, implement custom benchmark runner with YAML config for test toggling and dataset references, generate synthetic scenes as simple meshes (flat plane + analytic undulating surface) rendered with known camera geometry, and focus optimization effort on plane sweep cost volume construction (batching depths, optimizing grid_sample calls, reducing memory allocations).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Benchmark comparison scope:**
- Dual ground truth strategy: real ChArUco boards from AquaCal calibration data + synthetic scenes matching experimental geometry
- ChArUco evaluation: both plane-fitting (overall shape/scale accuracy) and point-level error at detected corner positions
- Synthetic scenes: flat plane at known depth (baseline) + undulating sand-like surface with known analytic form (depth variation recovery). Viewed through flat refractive interface matching 12-camera ring geometry. The refractive surface is always flat (air-water interface); the reconstruction target is the underwater surface (sand, tank floor, etc.)
- Accuracy metrics: completeness (surface coverage %) AND geometric error (mean/median distance to ground truth in mm). For synthetic data, report both raw coverage (any valid depth) and accurate coverage (within tolerance). For real data, gracefully skip tolerance-based completeness when dense ground truth unavailable
- Frame count: configurable — default to single frame, benchmark config can specify multiple frames for thorough runs with statistical reporting

**Benchmark tests:**
- CLAHE benchmark: CLAHE on vs off, tested across RoMa and LightGlue (with SuperPoint, ALIKED, DISK extractors), sparse mode
- Execution mode benchmark: all four modes — LightGlue+sparse, LightGlue+full, RoMa+sparse, RoMa+full. LightGlue uses one configurable matcher (user-selectable, sensible default). Combined or separate sparse/dense benchmarks depending on metric overlap
- Surface reconstruction benchmark: Poisson vs heightfield vs BPA
- Test toggling: each benchmark test enabled/disabled in the benchmark config YAML, not via CLI flags

**Benchmark CLI and config:**
- Single `aquamvs benchmark` command replaces existing benchmark command entirely (no backward compatibility concerns)
- Benchmark config YAML lives in the same directory as benchmark output
- Datasets referenced by direct paths in config (no registry)
- Each benchmark run produces a results directory with structured subfolders

**Profiling and optimization targets:**
- Optimize for both per-frame speed and throughput/scale — profile reveals priorities
- Both runtime AND memory profiling (GPU/CPU peak memory per stage)
- Plane sweep stereo identified as likely primary bottleneck
- Configurable quality presets (fast/balanced/quality) so users choose accuracy-speed tradeoff

**Benchmark suite design:**
- CI runs synthetic-only benchmarks, under 1 minute (subset of tests selected for speed)
- Real data benchmarks are local-only
- Results stored in separate `.benchmarks/` directory (gitignored), with committed summary
- Fixed percentage thresholds for regression detection (alert if metric degrades by more than X%)
- Built-in lightweight diff: `aquamvs benchmark --compare run1 run2` — summary table with absolute and percent deltas for key metrics

**Results reporting:**
- Terminal output: ASCII tables by default
- Optional `--visualize` flag generates comparison plots:
  - Error heatmaps (spatial error maps per config)
  - Grouped bar charts (accuracy, completeness, runtime across configs)
  - Depth map side-by-side comparisons
- Plots saved alongside results in structured subfolders
- Auto-generated Sphinx docs page from a published baseline run (intentionally updated, not automatic)

### Claude's Discretion

- Quality preset system design (global preset vs per-stage knobs with preset shortcuts)
- Specific regression threshold percentages
- Synthetic scene generation approach (rendering method, texture choices)
- CI benchmark subset selection (which tests fit in 1-minute budget)
- Compression algorithm for depth sampling in fast presets
- Subfolder structure within results directory
- Choice of profiling tools (cProfile, torch.profiler, etc.)

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope

</user_constraints>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch.profiler | PyTorch 2.0+ | Runtime and memory profiling | Built into PyTorch, supports CPU/CUDA, low overhead, records stack traces |
| tabulate | latest | ASCII table formatting | Industry standard for terminal tables, 10+ output formats |
| matplotlib | 3.7.0+ (already dep) | Visualization (heatmaps, bar charts) | Already dependency, comprehensive plotting, widely known |
| pydantic | 2.12.0+ (already dep) | Benchmark config validation | Already used for pipeline config, type-safe YAML loading |
| Open3D | 0.18.0+ (already dep) | Synthetic scene rendering | Already dependency, can render meshes to depth maps |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy.interpolate | 1.10.0+ (already dep) | Analytic surface generation | Creating undulating surfaces with known ground truth |
| pytest | latest (dev dep) | Test framework for benchmarks | CI integration, familiar test structure |
| rich | latest | Enhanced terminal output (optional) | If basic tabulate insufficient for progress/formatting |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| torch.profiler | cProfile | cProfile adds 50%+ overhead, no GPU memory tracking |
| torch.profiler | py-spy | py-spy: sampling profiler, lower overhead but less detailed for torch ops |
| torch.profiler | Scalene | Scalene: great for memory, but torch.profiler better integrated with PyTorch |
| Custom benchmarks | asv (Airspeed Velocity) | asv: designed for tracking over git history with web UI, but we need custom accuracy metrics not just timing |
| Custom benchmarks | pytest-benchmark | pytest-benchmark: wall-clock only, we need accuracy metrics + custom comparisons |
| tabulate | Rich tables | Rich: prettier output but heavier dependency, tabulate sufficient |

**Installation:**
```bash
# Core profiling/benchmarking (tabulate only new dependency)
pip install tabulate

# Optional: rich for enhanced terminal output
pip install rich
```

## Architecture Patterns

### Recommended Project Structure
```
src/aquamvs/
├── benchmark/
│   ├── __init__.py
│   ├── runner.py           # Main benchmark orchestrator
│   ├── datasets.py         # Ground truth dataset loaders (ChArUco, synthetic)
│   ├── metrics.py          # Accuracy metrics (completeness, geometric error)
│   ├── synthetic.py        # Synthetic scene generation
│   ├── comparison.py       # Benchmark diff/comparison logic
│   └── visualization.py    # Plot generation (heatmaps, bar charts)
├── profiling/
│   ├── __init__.py
│   ├── profiler.py         # torch.profiler wrapper with memory tracking
│   └── analyzer.py         # Profile result parsing and reporting
└── config.py              # Add BenchmarkConfig, QualityPreset

.benchmarks/
├── {run_id}/
│   ├── config.yaml         # Benchmark config for this run
│   ├── results/
│   │   ├── summary.json    # Overall metrics
│   │   ├── per_config/     # Results per tested config
│   │   └── plots/          # Generated visualizations (if --visualize)
│   └── logs/
└── baselines/
    └── {baseline_name}/    # Published baseline runs for docs

tests/
├── test_benchmark/
│   ├── test_synthetic.py   # Synthetic scene generation tests
│   ├── test_metrics.py     # Metric calculation tests
│   └── test_runner.py      # Benchmark runner tests
└── benchmarks/
    └── test_ci_benchmarks.py  # Fast CI benchmarks (synthetic only, <1 min)
```

### Pattern 1: Profiling with torch.profiler
**What:** Wrap pipeline stages with torch.profiler context manager, record both CPU/GPU time and memory allocations
**When to use:** Identifying bottlenecks in PyTorch-heavy code (plane sweep, dense matching)
**Example:**
```python
# Source: https://docs.pytorch.org/docs/stable/profiler.html
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
    with_stack=True
) as prof:
    with record_function("plane_sweep_stereo"):
        cost_volume = build_cost_volume(...)
    with record_function("depth_extraction"):
        depth_map, confidence = extract_depth(...)

# Print sorted by CUDA time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export Chrome trace for visualization
prof.export_chrome_trace("profile_trace.json")
```

### Pattern 2: Custom Benchmark Runner with YAML Config
**What:** YAML-driven benchmark config with test toggles, dataset paths, and comparison settings
**When to use:** Running accuracy benchmarks with custom metrics across multiple configurations
**Example:**
```python
# Benchmark config structure (Pydantic model)
from pydantic import BaseModel

class BenchmarkDataset(BaseModel):
    name: str
    type: Literal["charuco", "synthetic_plane", "synthetic_surface"]
    path: str  # Direct path to data
    ground_truth_tolerance_mm: float | None = None  # For accurate coverage metric

class BenchmarkTests(BaseModel):
    clahe_comparison: bool = True
    execution_mode_comparison: bool = True
    surface_reconstruction_comparison: bool = True

class BenchmarkConfig(BaseModel):
    output_dir: str  # Where results go (.benchmarks/{run_id})
    datasets: list[BenchmarkDataset]
    tests: BenchmarkTests
    regression_thresholds: dict[str, float]  # metric_name -> max % degradation
    frames: int = 1  # Number of frames to benchmark

# Usage: BenchmarkConfig.from_yaml(path)
```

### Pattern 3: Synthetic Scene Generation
**What:** Create meshes with known analytic surfaces, render to depth maps via Open3D or projection
**When to use:** Generating ground truth for benchmarks without heavyweight rendering engines
**Example:**
```python
# Source: Procedural mesh creation pattern
import numpy as np
import open3d as o3d
from scipy.interpolate import RBFInterpolator

def create_undulating_surface(bounds, resolution, amplitude, wavelength):
    """Create analytic undulating surface mesh."""
    x = np.arange(bounds[0], bounds[1], resolution)
    y = np.arange(bounds[2], bounds[3], resolution)
    xx, yy = np.meshgrid(x, y)

    # Analytic surface: sum of sinusoids
    zz = amplitude * (
        np.sin(2 * np.pi * xx / wavelength) +
        np.sin(2 * np.pi * yy / (wavelength * 1.3))
    )

    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

    # Create mesh from grid
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

    return mesh, (xx, yy, zz)  # Return mesh + analytic grid for GT

# Then project mesh into camera views to get ground truth depth
```

### Pattern 4: Metric Calculation with Spatial Coverage
**What:** Compute completeness (coverage %) and geometric error (distance to GT) with graceful degradation
**When to use:** Evaluating reconstruction accuracy against ground truth
**Example:**
```python
def compute_accuracy_metrics(
    reconstructed_points: np.ndarray,  # (N, 3)
    ground_truth_mesh: o3d.geometry.TriangleMesh,
    tolerance_mm: float | None = None
) -> dict[str, float]:
    """Compute accuracy metrics."""
    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(reconstructed_points)

    # Compute distances to ground truth mesh
    distances = np.asarray(pcd.compute_point_cloud_distance(ground_truth_mesh))
    distances_mm = distances * 1000  # Convert to mm

    metrics = {
        "mean_error_mm": float(np.mean(distances_mm)),
        "median_error_mm": float(np.median(distances_mm)),
        "std_error_mm": float(np.std(distances_mm)),
        "raw_completeness_pct": 100.0,  # All reconstructed points counted
    }

    # Accurate completeness only if tolerance provided
    if tolerance_mm is not None:
        accurate_mask = distances_mm <= tolerance_mm
        metrics["accurate_completeness_pct"] = float(np.mean(accurate_mask) * 100)

    return metrics
```

### Pattern 5: Quality Presets
**What:** Pre-configured speed/quality tradeoffs via preset names that modify multiple config params
**When to use:** Letting users choose reconstruction quality without tuning individual parameters
**Example:**
```python
# Preset system design (Claude's discretion)
from enum import Enum

class QualityPreset(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"

PRESET_CONFIGS = {
    QualityPreset.FAST: {
        "num_depths": 64,
        "window_size": 7,
        "max_keypoints": 1024,
        "voxel_size": 0.002,
        "poisson_depth": 8,
    },
    QualityPreset.BALANCED: {
        "num_depths": 128,
        "window_size": 11,
        "max_keypoints": 2048,
        "voxel_size": 0.001,
        "poisson_depth": 9,
    },
    QualityPreset.QUALITY: {
        "num_depths": 256,
        "window_size": 15,
        "max_keypoints": 4096,
        "voxel_size": 0.0005,
        "poisson_depth": 10,
    },
}

# Apply preset: config.apply_preset(QualityPreset.FAST)
# Or allow per-param override: preset="fast" but num_depths=128
```

### Anti-Patterns to Avoid
- **Profiling without record_shapes=True:** Shape information crucial for understanding memory bottlenecks in tensor ops
- **Global regression thresholds:** Different metrics have different stability (10% might be right for runtime, 5% for accuracy)
- **Heavyweight synthetic datasets:** Hypersim and similar are overkill for simple geometric validation
- **Forgetting profile_memory=True:** GPU memory is often the bottleneck, not runtime
- **Not filtering noisy profiler output:** torch.profiler captures EVERYTHING, filter to relevant operations only

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ASCII table formatting | Manual spacing/alignment | tabulate | Handles alignment, multiple formats, unicode, edge cases |
| Profiler result parsing | String parsing of print output | prof.key_averages(), export_chrome_trace() | Structured data, sortable, exportable |
| Statistical outlier removal | Custom std-dev filtering | Open3D remove_statistical_outlier (already in surface.py) | Handles edge cases, optimized |
| Depth map to point cloud | Manual unproject loop | Vectorized ray casting + masking | Orders of magnitude faster |
| Mesh rendering to depth | Ray tracing from scratch | Open3D ray casting or projection | Handles occlusion, optimized |
| Ground truth alignment | Custom ICP | Open3D registration_icp (already used for eval) | Robust, tested, multiple variants |

**Key insight:** Performance optimization is about measuring first, then optimizing hot paths. Custom implementations of profiling infrastructure or geometric algorithms introduce bugs and add maintenance burden without performance benefit.

## Common Pitfalls

### Pitfall 1: Profiling Without Warmup
**What goes wrong:** First PyTorch operation includes CUDA initialization overhead, skewing measurements
**Why it happens:** CUDA context lazy-initializes on first GPU operation
**How to avoid:** Run warmup iterations before starting profiler
**Warning signs:** First operation shows 10-100x longer than subsequent calls
**Example:**
```python
# Warm up CUDA
if device == "cuda":
    dummy = torch.zeros(1, device=device)
    del dummy
    torch.cuda.synchronize()

# Now profile
with profile(...) as prof:
    actual_workload()
```

### Pitfall 2: grid_sample Performance Cliff
**What goes wrong:** grid_sample backward pass scales poorly with input size, can be 10x slower than forward
**Why it happens:** PyTorch grid_sample backward requires expensive gradient computation for both input and grid
**How to avoid:** If gradients not needed, use torch.no_grad() or .detach(); consider batching to amortize overhead
**Warning signs:** Profiler shows grid_sample backward >> forward time
**Example:**
```python
# Plane sweep doesn't need gradients
with torch.no_grad():
    warped = F.grid_sample(src_4d, grid, ...)
```

### Pitfall 3: Memory Fragmentation in Cost Volume Construction
**What goes wrong:** Allocating cost_volume[:, :, d_idx] slice-by-slice causes fragmentation and OOM
**Why it happens:** PyTorch allocates new tensors for each slice assignment, doesn't reuse memory
**How to avoid:** Pre-allocate full cost volume, write into it; or batch depth hypotheses
**Warning signs:** Memory usage grows then plateaus far below GPU capacity, OOM errors
**Example:**
```python
# Bad: allocates new tensor each iteration
for d_idx in range(D):
    cost_volume[:, :, d_idx] = compute_cost(...)

# Good: pre-allocate, write in-place
cost_volume = torch.zeros(H, W, D, device=device)
for d_idx in range(D):
    cost_volume[:, :, d_idx] = compute_cost(...)
```

### Pitfall 4: Not Batching Depth Hypotheses
**What goes wrong:** Processing depths one-at-a-time underutilizes GPU, adds kernel launch overhead
**Why it happens:** Natural to loop over depths, but GPU wants large batches
**How to avoid:** Batch multiple depths together in warp/cost computation, trade memory for speed
**Warning signs:** GPU utilization < 50%, profiler shows tiny kernel times with gaps
**Example:**
```python
# Fast preset: batch 4-8 depths at once
# Balanced: batch 2-4 depths
# Quality: depth-by-depth (memory constrained)
batch_size = 4 if preset == "fast" else 1
for batch_start in range(0, D, batch_size):
    depths_batch = depths[batch_start:batch_start+batch_size]
    # Compute cost for all depths in batch simultaneously
```

### Pitfall 5: ChArUco Corner Detection Without Distortion Parameters
**What goes wrong:** ChArUco corner interpolation via homography fails when images are distorted
**Why it happens:** Homography assumes planar projection, doesn't account for lens distortion
**How to avoid:** Pass calibration parameters to detectMarkers, or use undistorted images
**Warning signs:** Corner reprojection error > 1 pixel, checkerboard corners misaligned
**Example:**
```python
# Good: detect on undistorted image
undistorted = undistort_image(raw_image, undistortion_maps[camera_name])
corners, ids, _ = cv2.aruco.detectMarkers(undistorted, dictionary)
```

### Pitfall 6: Forgetting torch.cuda.synchronize() in Timing
**What goes wrong:** CUDA operations are asynchronous, timing without sync includes only kernel launch
**Why it happens:** torch.cuda.synchronize() required to wait for GPU completion
**How to avoid:** Use torch.profiler (handles sync) or manually sync before stopping timer
**Warning signs:** GPU operations show unrealistically fast times (microseconds)
**Example:**
```python
# Bad
start = time.time()
result = gpu_operation()
end = time.time()  # Only measures kernel launch!

# Good (manual timing)
torch.cuda.synchronize()
start = time.time()
result = gpu_operation()
torch.cuda.synchronize()
end = time.time()

# Better: use torch.profiler, handles sync automatically
```

## Code Examples

Verified patterns from official sources and current codebase:

### torch.profiler Memory Tracking
```python
# Source: https://docs.pytorch.org/docs/stable/profiler.html
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
) as prof:
    model(inputs)

# Memory usage by operation
print(prof.key_averages().table(
    sort_by="self_cuda_memory_usage",
    row_limit=10
))
```

### Tabulate Table Output
```python
# Source: https://pypi.org/project/tabulate/
from tabulate import tabulate

results = [
    ["LightGlue+sparse", 95.2, 2.3, 15.2],
    ["LightGlue+full", 97.8, 1.8, 45.6],
    ["RoMa+sparse", 94.1, 2.5, 12.3],
    ["RoMa+full", 98.5, 1.6, 38.9],
]

print(tabulate(
    results,
    headers=["Config", "Coverage %", "Error (mm)", "Time (s)"],
    tablefmt="grid",
    floatfmt=".1f"
))
```

### Matplotlib Error Heatmap
```python
# Source: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
im = ax.imshow(error_map_mm, cmap="hot", vmin=0, vmax=10)
ax.set_title("Geometric Error (mm)")
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Error (mm)", rotation=-90, va="bottom")
plt.savefig("error_heatmap.png", dpi=150)
```

### Batched Depth Plane Sweep (Optimization)
```python
# Optimization example: batch depth hypotheses
def build_cost_volume_batched(
    ref_model, src_models, ref_image, src_images, depths, config, batch_size=4
):
    """Batched cost volume construction for better GPU utilization."""
    H, W = ref_image.shape
    D = depths.shape[0]
    device = ref_image.device

    cost_volume = torch.zeros(H, W, D, device=device)
    pixel_grid = _make_pixel_grid(H, W, device=device)
    origins, directions = ref_model.cast_ray(pixel_grid)

    for batch_start in range(0, D, batch_size):
        batch_end = min(batch_start + batch_size, D)
        depths_batch = depths[batch_start:batch_end]  # (B,)

        # Warp all depths in batch simultaneously
        # Shape: (B, S, H, W) for B depths, S sources
        warped_batch = []
        for depth in depths_batch:
            warped_sources = []
            for src_idx, src_model in enumerate(src_models):
                warped = _warp_source_at_depth(
                    origins, directions, src_model,
                    src_images[src_idx], depth.item(), H, W
                )
                warped_sources.append(warped)
            warped_batch.append(torch.stack(warped_sources, dim=0))

        # Compute costs for batch
        for i, d_idx in enumerate(range(batch_start, batch_end)):
            source_costs = []
            for s_idx in range(len(src_models)):
                cost = compute_cost(ref_image, warped_batch[i][s_idx], ...)
                source_costs.append(cost)
            cost_volume[:, :, d_idx] = aggregate_costs(source_costs)

    return cost_volume
```

### Regression Detection
```python
# Pattern for regression detection with percentage thresholds
def detect_regressions(
    baseline: dict[str, float],
    current: dict[str, float],
    thresholds: dict[str, float]
) -> dict[str, tuple[float, float, bool]]:
    """
    Detect regressions by comparing to baseline.

    Returns:
        {metric: (baseline_val, current_val, is_regression)}
    """
    regressions = {}
    for metric, baseline_val in baseline.items():
        if metric not in current:
            continue

        current_val = current[metric]
        threshold_pct = thresholds.get(metric, 0.10)  # Default 10%

        # For error metrics: regression if current > baseline
        # For coverage/accuracy: regression if current < baseline
        if "error" in metric.lower():
            is_regression = (current_val - baseline_val) / baseline_val > threshold_pct
        else:
            is_regression = (baseline_val - current_val) / baseline_val > threshold_pct

        regressions[metric] = (baseline_val, current_val, is_regression)

    return regressions
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| cProfile for all profiling | torch.profiler with memory tracking | PyTorch 1.8 (2021) | GPU memory profiling, CUDA kernels visible |
| pytest-benchmark for timing | Custom benchmarks with accuracy metrics | N/A | Need domain-specific metrics (coverage, error) |
| Heavyweight synthetic datasets (Hypersim) | Lightweight procedural meshes | 2023-2024 (MVS research) | Faster generation, known analytic GT |
| Fixed depth sampling | Adaptive/cascaded depth | CasMVSNet (2020) | 50%+ memory reduction for high-res |
| 3D CNN cost regularization | Recurrent/iterative refinement | 2021-2023 (MVSNet variants) | Enables high-resolution reconstruction |
| Single regression threshold | Per-metric thresholds | Bencher/CodSpeed (2025) | Fewer false positives from variable benchmarks |

**Deprecated/outdated:**
- **cProfile for PyTorch profiling:** torch.profiler supersedes it for GPU workloads (cProfile can't see CUDA)
- **pytest-benchmark/asv for accuracy benchmarks:** Designed for timing, not custom metrics like geometric error
- **Homography-based ChArUco corner detection without calibration:** OpenCV docs recommend passing calibration params

## Open Questions

1. **Optimal batch size for depth hypotheses in plane sweep**
   - What we know: Batching improves GPU utilization, trades memory for speed
   - What's unclear: Optimal batch size depends on image resolution, GPU memory, number of sources
   - Recommendation: Start with batch_size=1 (quality preset), profile to find memory headroom, then test 2/4/8 for fast preset

2. **Regression threshold percentages**
   - What we know: 10% common for response time, variable for accuracy metrics
   - What's unclear: Right threshold for completeness vs geometric error vs runtime in this domain
   - Recommendation: Start conservative (5% for accuracy, 10% for runtime), tune based on observed variance

3. **CI benchmark time budget distribution**
   - What we know: Total budget <1 minute for synthetic-only
   - What's unclear: Split between CLAHE (2 configs), execution mode (4 configs), surface (3 configs)
   - Recommendation: Profile each test first, allocate budget proportional to value (execution mode most important)

4. **Synthetic scene texture/realism**
   - What we know: Need analytic surface for GT, matching camera geometry
   - What's unclear: Does synthetic texture need to match real underwater scenes? Photorealism needed?
   - Recommendation: Start with simple uniform texture (validates geometry), add noise/texture if feature extraction fails

## Sources

### Primary (HIGH confidence)
- [PyTorch Profiler Documentation](https://docs.pytorch.org/docs/stable/profiler.html) - torch.profiler API, memory tracking
- [torch.profiler tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) - Usage examples
- [torch.nn.functional.grid_sample](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html) - API reference
- [Open3D Surface Reconstruction](https://www.open3d.org/docs/latest/tutorial/geometry/surface_reconstruction.html) - Poisson, BPA methods
- [tabulate PyPI](https://pypi.org/project/tabulate/) - ASCII table library
- [matplotlib heatmap](https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html) - Heatmap examples
- [OpenCV ChArUco Detection](https://docs.opencv.org/3.4/df/d4a/tutorial_charuco_detection.html) - Corner detection

### Secondary (MEDIUM confidence)
- [Bencher Thresholds](https://bencher.dev/docs/explanation/thresholds/) - Regression detection best practices (10% common)
- [Scalene profiler](https://github.com/plasma-umass/scalene) - Alternative profiler with AI suggestions
- [py-spy profiler](https://github.com/benfred/py-spy) - Low-overhead sampling profiler
- [Python profiling tools comparison](https://daily.dev/blog/top-7-python-profiling-tools-for-performance) - 2026 overview
- [Pydantic YAML config](https://medium.com/@jonathan_b/a-simple-guide-to-configure-your-python-project-with-pydantic-and-a-yaml-file-bef76888f366) - Best practices
- [PyTorch einsum optimization](https://optimized-einsum.readthedocs.io/en/stable/) - opt_einsum for faster tensor ops

### Tertiary (LOW confidence - WebSearch only)
- [DSC-MVSNet](https://link.springer.com/article/10.1007/s40747-023-01106-3) - Depthwise separable convolutions for cost volume (49% memory reduction)
- [LE-MVSNet](https://link.springer.com/chapter/10.1007/978-3-031-44198-1_40) - Lightweight MVS (52% memory, 88% runtime reduction)
- [grid_sample performance issues](https://discuss.pytorch.org/t/f-grid-sample-extremely-slow/164220) - Community reports (backward pass slow)
- [Hypersim dataset](https://github.com/apple/ml-hypersim) - Synthetic dataset toolkit (for context, not recommended use)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - torch.profiler, tabulate, matplotlib all standard tools with official docs
- Architecture: HIGH - Patterns verified against PyTorch docs and current codebase structure
- Pitfalls: MEDIUM-HIGH - grid_sample performance issues documented in PyTorch issues, profiling pitfalls from official docs, ChArUco from OpenCV docs
- Optimization approaches: MEDIUM - MVSNet variants from 2023-2025 research, not yet standard practice but published and cited

**Research date:** 2026-02-15
**Valid until:** ~60 days (stable domain - PyTorch profiling API stable since 1.8, benchmarking practices evolving slowly)
