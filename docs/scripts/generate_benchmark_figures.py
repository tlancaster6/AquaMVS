"""Generate static benchmark comparison figures for AquaMVS documentation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from aquamvs.benchmark.runner import BenchmarkResult

# Color palettes
PATHWAY_COLORS = ["#e07b54", "#2e86ab", "#1a5276"]
STAGE_COLORS = ["#4e8098", "#90c2e7", "#6baed6", "#e07b54", "#f4a261", "#a8dadc"]

STAGE_ORDER = [
    ("undistortion", "Undistortion"),
    ("sparse_matching", "Sparse Matching"),
    ("dense_matching", "Dense Matching"),
    ("depth_estimation", "Depth Estimation"),
    ("fusion", "Fusion"),
    ("surface", "Surface"),
]


def _save_figure(fig: plt.Figure, path: Path) -> None:
    """Save a matplotlib figure and close it.

    Args:
        fig: Figure to save.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_runtime_and_points(result: BenchmarkResult, output_dir: Path) -> Path:
    """Generate side-by-side bar charts of runtime and point count.

    Args:
        result: BenchmarkResult from run_benchmark.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved PNG.
    """
    names = [pw.pathway_name for pw in result.results]
    total_times_s = [pw.timing.total_time_ms / 1000.0 for pw in result.results]
    point_counts = [pw.point_count for pw in result.results]

    colors = PATHWAY_COLORS[: len(names)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Total runtime
    axes[0].bar(names, total_times_s, color=colors)
    axes[0].set_title("Total Runtime per Pathway", fontsize=13)
    axes[0].set_ylabel("Time (s)")
    axes[0].set_xlabel("Pathway")
    axes[0].tick_params(axis="x", rotation=15)
    for i, v in enumerate(total_times_s):
        axes[0].text(
            i, v + max(total_times_s) * 0.02, f"{v:.1f}s", ha="center", fontsize=9
        )

    # Point count
    axes[1].bar(names, point_counts, color=colors)
    axes[1].set_title("Reconstructed Point Count per Pathway", fontsize=13)
    axes[1].set_ylabel("Points")
    axes[1].set_xlabel("Pathway")
    axes[1].tick_params(axis="x", rotation=15)
    for i, v in enumerate(point_counts):
        axes[1].text(i, v + max(point_counts) * 0.01, f"{v:,}", ha="center", fontsize=9)

    fig.tight_layout()
    out_path = output_dir / "runtime_and_points.png"
    _save_figure(fig, out_path)
    return out_path


def generate_stage_timing(result: BenchmarkResult, output_dir: Path) -> Path:
    """Generate stacked bar chart of per-stage wall times.

    Args:
        result: BenchmarkResult from run_benchmark.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved PNG.
    """
    pathway_names = [pw.pathway_name for pw in result.results]
    x = np.arange(len(pathway_names))

    fig, ax = plt.subplots(figsize=(12, 6))
    bottoms = np.zeros(len(pathway_names))

    for (stage_key, stage_label), color in zip(STAGE_ORDER, STAGE_COLORS, strict=False):
        stage_times = []
        for pw in result.results:
            stage = pw.timing.stages.get(stage_key)
            stage_times.append(
                stage.wall_time_ms / 1000.0 if stage is not None else 0.0
            )
        ax.bar(x, stage_times, bottom=bottoms, color=color, label=stage_label)
        bottoms += np.array(stage_times)

    ax.set_title("Stage Timing Breakdown per Pathway", fontsize=13)
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Pathway")
    ax.set_xticks(x)
    ax.set_xticklabels(pathway_names, rotation=15)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    out_path = output_dir / "stage_timing.png"
    _save_figure(fig, out_path)
    return out_path


def generate_depth_comparison(
    result: BenchmarkResult,
    config_path: Path,
    output_dir: Path,
) -> Path:
    """Generate side-by-side depth maps for LG+SP full and RoMa full.

    Args:
        result: BenchmarkResult from run_benchmark.
        config_path: Path to pipeline config YAML.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved PNG.
    """
    from aquamvs import PipelineConfig

    base_config = PipelineConfig.from_yaml(config_path)
    base_output = Path(base_config.output_dir)
    cam = list(base_config.camera_input_map.keys())[0]
    frame = result.frame

    pathway_dirs = {
        "LG+SP full": base_output / "benchmark" / "LG_SP_full",
        "RoMa full": base_output / "benchmark" / "RoMa_full",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (pathway_name, out_dir) in zip(axes, pathway_dirs.items(), strict=False):
        depth_path = out_dir / f"frame_{frame:06d}" / "depth_maps" / f"{cam}.npz"
        if depth_path.exists():
            depth = np.load(depth_path)["depth"]
            im = ax.imshow(depth, cmap="viridis")
            plt.colorbar(im, ax=ax, label="Depth (m)", shrink=0.8)
            ax.set_title(f"{pathway_name} -- {cam}")
            ax.axis("off")
        else:
            ax.set_title(pathway_name)
            ax.text(
                0.5,
                0.5,
                "Depth map not found.\nRun benchmark first.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
            )
            ax.axis("off")

    fig.suptitle("Depth Map Comparison (full-mode pathways)", fontsize=13)
    fig.tight_layout()

    out_path = output_dir / "depth_comparison.png"
    _save_figure(fig, out_path)
    return out_path


def generate_reconstruction_comparison(
    result: BenchmarkResult,
    config_path: Path,
    output_dir: Path,
) -> Path:
    """Generate side-by-side 3D renders from all 3 pathways.

    Loads pre-rendered fused_oblique.png from each pathway's viz directory.

    Args:
        result: BenchmarkResult from run_benchmark.
        config_path: Path to pipeline config YAML.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved PNG.
    """
    from aquamvs import PipelineConfig

    base_config = PipelineConfig.from_yaml(config_path)
    base_output = Path(base_config.output_dir)
    frame = result.frame

    pathway_dirs = {
        "RoMa full": base_output / "benchmark" / "RoMa_full",
        "LG+SP full": base_output / "benchmark" / "LG_SP_full",
        "LG+SP sparse": base_output / "benchmark" / "LG_SP_sparse",
    }

    images: dict[str, np.ndarray] = {}
    for name, out_dir in pathway_dirs.items():
        img_path = out_dir / f"frame_{frame:06d}" / "viz" / "fused_oblique.png"
        if img_path.exists():
            images[name] = plt.imread(str(img_path))

    if images:
        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]
        for ax, (name, img) in zip(axes, images.items(), strict=False):
            ax.imshow(img)
            ax.set_title(name, fontsize=12)
            ax.axis("off")
        fig.suptitle("3D Reconstruction Comparison", fontsize=14)
    else:
        fig, ax = plt.subplots(figsize=(18, 5))
        ax.text(
            0.5,
            0.5,
            "No rendered PNGs found.\nSet viz_enabled: true in config and re-run benchmark.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.axis("off")

    fig.tight_layout()

    out_path = output_dir / "reconstruction_comparison.png"
    _save_figure(fig, out_path)
    return out_path


def main() -> None:
    """Run benchmark and generate all comparison figures."""
    import logging

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Generate benchmark comparison figures for AquaMVS documentation."
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to pipeline config YAML",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to benchmark (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/_static/benchmark/"),
        help="Where to save figures (default: docs/_static/benchmark/)",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmark
    from aquamvs.benchmark.report import save_markdown_report
    from aquamvs.benchmark.runner import run_benchmark

    print(f"Running benchmark on frame {args.frame}...")
    result = run_benchmark(config_path=args.config_path, frame=args.frame)
    print(f"Benchmarked {len(result.results)} pathways.\n")

    # Generate figures
    saved: list[Path] = []

    print("Generating runtime_and_points.png...")
    saved.append(generate_runtime_and_points(result, output_dir))

    print("Generating stage_timing.png...")
    saved.append(generate_stage_timing(result, output_dir))

    print("Generating depth_comparison.png...")
    saved.append(generate_depth_comparison(result, args.config_path, output_dir))

    print("Generating reconstruction_comparison.png...")
    saved.append(
        generate_reconstruction_comparison(result, args.config_path, output_dir)
    )

    # Save markdown report
    report_path = save_markdown_report(result, output_dir)

    # Print summary
    print("\nSaved figures:")
    for p in saved:
        print(f"  {p}")
    print(f"\nMarkdown report: {report_path}")


if __name__ == "__main__":
    main()
