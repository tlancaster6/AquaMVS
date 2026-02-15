"""Synthetic profiling script for pipeline stage performance measurement.

Profiles representative pipeline operations using synthetic tensor data
without requiring real video files or calibration. Exercises computational
kernels directly with realistic tensor sizes.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.profiler import record_function

# Handle both module and script execution
try:
    from .analyzer import format_report
    from .profiler import PipelineProfiler
except ImportError:
    # Running as script, use absolute imports
    from aquamvs.profiling.analyzer import format_report
    from aquamvs.profiling.profiler import PipelineProfiler

logger = logging.getLogger(__name__)


def synthetic_undistortion(images: torch.Tensor, device: str) -> torch.Tensor:
    """Simulate undistortion with grid_sample operations.

    Args:
        images: Batch of images (N, C, H, W).
        device: Device for computation.

    Returns:
        Undistorted images (N, C, H, W).
    """
    N, C, H, W = images.shape

    # Create synthetic distortion grid (barrel distortion pattern)
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    r = torch.sqrt(x**2 + y**2)
    k = 0.2  # distortion coefficient
    scale = 1 + k * r**2
    grid = (
        torch.stack([x / scale, y / scale], dim=-1).unsqueeze(0).expand(N, -1, -1, -1)
    )

    # Apply grid_sample (simulates remap operation)
    undistorted = F.grid_sample(images, grid, align_corners=True, mode="bilinear")
    return undistorted


def synthetic_sparse_matching(num_features: int, device: str) -> dict:
    """Simulate sparse feature extraction and matching.

    Args:
        num_features: Number of features per image.
        device: Device for computation.

    Returns:
        Dictionary with keypoints and descriptors.
    """
    # Simulate keypoint extraction (random positions)
    keypoints = torch.rand(num_features, 2, device=device) * 1920

    # Simulate descriptor computation (256-dim normalized vectors)
    descriptors = torch.randn(num_features, 256, device=device)
    descriptors = F.normalize(descriptors, dim=-1)

    # Simulate matching (matrix multiplication + thresholding)
    scores = descriptors @ descriptors.T  # (N, N)
    matches = torch.nonzero(scores > 0.8, as_tuple=False)

    return {
        "keypoints": keypoints,
        "descriptors": descriptors,
        "matches": matches,
    }


def synthetic_depth_estimation(
    ref_image: torch.Tensor,
    src_images: torch.Tensor,
    num_depths: int,
    device: str,
    batch_size: int = 1,
) -> torch.Tensor:
    """Simulate plane sweep depth estimation.

    Args:
        ref_image: Reference image (1, 1, H, W).
        src_images: Source images (N, 1, H, W).
        num_depths: Number of depth planes.
        device: Device for computation.
        batch_size: Depth batch size for processing.

    Returns:
        Cost volume (D, H, W).
    """
    _, _, H, W = ref_image.shape
    N = src_images.shape[0]

    cost_volume = torch.zeros(num_depths, H, W, device=device)

    # Generate synthetic homographies (simulate warping)
    for d_idx in range(num_depths):
        # Create synthetic warp (translation pattern)
        scale = 1.0 + 0.01 * (d_idx - num_depths // 2)
        theta = torch.tensor(
            [[scale, 0, 0.01 * d_idx], [0, scale, 0.01 * d_idx]],
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0)

        # Warp each source image
        grid = F.affine_grid(theta, (1, 1, H, W), align_corners=False)

        for src_idx in range(N):
            warped = F.grid_sample(
                src_images[src_idx : src_idx + 1],
                grid,
                align_corners=False,
                mode="bilinear",
            )

            # Compute cost (L1 difference)
            cost = torch.abs(ref_image - warped).squeeze()
            cost_volume[d_idx] += cost

    # Normalize by number of sources
    cost_volume /= N

    return cost_volume


def synthetic_fusion(depth_maps: torch.Tensor, device: str) -> torch.Tensor:
    """Simulate depth map fusion with median filtering.

    Args:
        depth_maps: Stack of depth maps (N, H, W).
        device: Device for computation.

    Returns:
        Fused depth map (H, W).
    """
    # Median fusion
    fused = torch.median(depth_maps, dim=0).values

    # Apply 3x3 median filter for smoothing
    # Pad and unfold for median filter
    padded = F.pad(fused.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate")
    patches = padded.unfold(2, 3, 1).unfold(3, 3, 1)  # (1, 1, H, W, 3, 3)
    patches_flat = patches.reshape(
        patches.shape[0], patches.shape[1], patches.shape[2], patches.shape[3], -1
    )
    filtered = torch.median(patches_flat, dim=-1).values.squeeze()

    return filtered


def synthetic_surface_reconstruction(points: torch.Tensor, device: str) -> dict:
    """Simulate Poisson reconstruction operations.

    Args:
        points: Point cloud (N, 3).
        device: Device for computation.

    Returns:
        Dictionary with mesh statistics.
    """
    # Simulate normal estimation (cross product of local neighborhoods)
    # Use simple kNN-like operation with matrix ops
    N = points.shape[0]

    # Compute pairwise distances (N, N)
    dists = torch.cdist(points, points)

    # Find k-nearest neighbors (k=10)
    k = 10
    _, indices = torch.topk(dists, k + 1, largest=False, dim=-1)  # exclude self

    # Compute normals via covariance
    normals = torch.zeros_like(points)
    for i in range(N):
        neighbors = points[indices[i, 1:]]  # exclude self
        centered = neighbors - neighbors.mean(dim=0)
        cov = centered.T @ centered
        # Normal is smallest eigenvector (approximate with last row of SVD V)
        _, _, v = torch.svd(cov)
        normals[i] = v[:, -1]

    return {
        "normals": normals,
        "num_points": N,
    }


def run_synthetic_profile(output_path: Path, device: str = "cpu") -> dict:
    """Run full synthetic profiling pipeline.

    Args:
        output_path: Path to save JSON report.
        device: Device for computation ("cpu" or "cuda").

    Returns:
        Report data dictionary.
    """
    logger.info("Starting synthetic profiling on device: %s", device)

    # Synthetic data dimensions (realistic pipeline sizes)
    num_cameras = 4
    height, width = 1080, 1920
    num_depths = 64
    num_features = 2000

    # Create profiler
    profiler = PipelineProfiler(
        activities=["cpu", "cuda"] if device == "cuda" else ["cpu"],
        profile_memory=True,
        record_shapes=True,
    )

    with profiler:
        # Stage 1: Undistortion
        with record_function("undistortion"):
            images = torch.randn(num_cameras, 3, height, width, device=device)
            undistorted = synthetic_undistortion(images, device)

        # Stage 2: Sparse matching
        with record_function("sparse_matching"):
            _ = synthetic_sparse_matching(num_features, device)

        # Stage 3: Depth estimation (plane sweep)
        with record_function("depth_estimation"):
            ref_gray = F.rgb_to_grayscale(undistorted[0:1])
            src_gray = torch.stack(
                [
                    F.rgb_to_grayscale(undistorted[i : i + 1])
                    for i in range(1, num_cameras)
                ]
            )
            cost_volume = synthetic_depth_estimation(
                ref_gray, src_gray, num_depths, device, batch_size=1
            )

            # Extract depth via winner-takes-all
            with record_function("extract_depth"):
                depth_map = torch.argmin(cost_volume, dim=0).float()

        # Stage 4: Fusion
        with record_function("fusion"):
            # Simulate multiple depth maps
            depth_maps = torch.stack(
                [depth_map + torch.randn_like(depth_map) * 0.5 for _ in range(3)]
            )
            fused_depth = synthetic_fusion(depth_maps, device)

        # Stage 5: Surface reconstruction
        with record_function("surface_reconstruction"):
            # Convert depth to point cloud (simulate)
            valid_mask = fused_depth > 0
            num_valid = valid_mask.sum().item()
            points = torch.randn(num_valid, 3, device=device)  # synthetic 3D points
            _ = synthetic_surface_reconstruction(points, device)

    # Generate report
    report = profiler.get_report()
    formatted = format_report(report)

    print("\n" + formatted)

    # Extract report data for JSON
    report_data = {
        "device": report.device,
        "total_time_ms": report.total_time_ms,
        "total_memory_peak_mb": report.total_memory_peak_mb,
        "stages": {
            name: {
                "cpu_time_ms": stage.cpu_time_ms,
                "cuda_time_ms": stage.cuda_time_ms,
                "total_time_ms": stage.cpu_time_ms + stage.cuda_time_ms,
                "cpu_memory_mb": stage.cpu_memory_mb,
                "cuda_memory_mb": stage.cuda_memory_mb,
                "total_memory_mb": abs(stage.cpu_memory_mb) + abs(stage.cuda_memory_mb),
            }
            for name, stage in report.stages.items()
        },
        "top_bottlenecks": [
            {
                "name": name,
                "time_ms": time_ms,
                "memory_mb": memory_mb,
            }
            for name, time_ms, memory_mb in report.top_bottlenecks
        ],
    }

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report_data, f, indent=2)

    logger.info("Profile report saved to: %s", output_path)

    return report_data


def benchmark_depth_batching(device: str = "cpu") -> dict:
    """Benchmark plane sweep performance with different batch sizes.

    Args:
        device: Device for computation ("cpu" or "cuda").

    Returns:
        Dictionary with batch size -> timing results.
    """
    print("\n" + "=" * 60)
    print("Depth Batching Optimization Comparison")
    print("=" * 60)

    # Synthetic scenario
    num_cameras = 5
    height, width = 480, 640  # Smaller for faster iteration
    num_depths = 64

    # Prepare synthetic data
    ref_image = torch.randn(1, 1, height, width, device=device)
    src_images = torch.randn(num_cameras - 1, 1, height, width, device=device)

    batch_sizes = [1, 8, 16]
    results = {}

    for batch_size in batch_sizes:
        # Warmup
        if device == "cuda":
            torch.cuda.synchronize()
            _ = synthetic_depth_estimation(
                ref_image, src_images, num_depths, device, batch_size
            )
            torch.cuda.synchronize()

        # Timing
        num_runs = 5
        times = []

        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = synthetic_depth_estimation(
                    ref_image, src_images, num_depths, device, batch_size
                )
                torch.cuda.synchronize()
                end = time.perf_counter()
            else:
                start = time.perf_counter()
                _ = synthetic_depth_estimation(
                    ref_image, src_images, num_depths, device, batch_size
                )
                end = time.perf_counter()

            times.append((end - start) * 1000)  # convert to ms

        avg_time = np.mean(times)
        std_time = np.std(times)
        results[batch_size] = {"avg_ms": avg_time, "std_ms": std_time}

        print(f"  batch_size={batch_size:2d}: {avg_time:6.2f} ± {std_time:4.2f} ms")

    # Compute speedup
    baseline = results[1]["avg_ms"]
    print("\nSpeedup vs batch_size=1:")
    for batch_size in [8, 16]:
        speedup = baseline / results[batch_size]["avg_ms"]
        print(f"  batch_size={batch_size:2d}: {speedup:.2f}x")

    return results


def main():
    """CLI entry point for synthetic profiling."""
    parser = argparse.ArgumentParser(description="Profile pipeline with synthetic data")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".benchmarks/profile_report.json"),
        help="Output path for JSON report",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for computation",
    )
    parser.add_argument(
        "--batch-benchmark",
        action="store_true",
        help="Also run depth batching benchmark",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA requested but not available, falling back to CPU")
        args.device = "cpu"

    # Run profiling
    try:
        report_data = run_synthetic_profile(args.output, args.device)

        # Optionally run batching benchmark
        if args.batch_benchmark:
            batch_results = benchmark_depth_batching(args.device)
            report_data["depth_batching"] = batch_results

            # Update JSON with batching results
            with open(args.output, "w") as f:
                json.dump(report_data, f, indent=2)

        return 0

    except Exception as e:
        logger.exception("Profiling failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
