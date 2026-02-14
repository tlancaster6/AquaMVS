"""Evaluation plots and time-series gallery."""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render_error_histogram(
    distances: np.ndarray,
    output_path: str | Path,
    title: str = "Cloud-to-Cloud Distance",
    xlabel: str = "Distance (mm)",
    n_bins: int = 50,
    dpi: int = 150,
) -> None:
    """Plot a histogram of distance errors with summary statistics.

    Args:
        distances: 1D array of distance values (meters). Will be displayed in mm.
        output_path: Path to save the PNG image.
        title: Plot title.
        xlabel: X-axis label.
        n_bins: Number of histogram bins.
        dpi: Output resolution.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Convert to mm for display
    dist_mm = distances * 1000.0

    ax.hist(dist_mm, bins=n_bins, color="steelblue", edgecolor="white", alpha=0.8)

    # Summary lines
    mean_val = float(np.mean(dist_mm))
    median_val = float(np.median(dist_mm))
    ax.axvline(
        mean_val,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_val:.2f} mm",
    )
    ax.axvline(
        median_val,
        color="orange",
        linestyle="-.",
        linewidth=1.5,
        label=f"Median: {median_val:.2f} mm",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def render_distance_map(
    diff_map: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    output_path: str | Path,
    title: str = "Height-Map Difference",
    dpi: int = 150,
) -> None:
    """Plot a 2D spatial map of height differences.

    Uses a diverging colormap (blue-white-red) centered at zero.
    NaN cells are displayed as gray background.

    Args:
        diff_map: 2D array (Ny, Nx) of Z differences (meters). NaN where no data.
        grid_x: X coordinates of grid columns, shape (Nx,).
        grid_y: Y coordinates of grid rows, shape (Ny,).
        output_path: Path to save the PNG image.
        title: Plot title.
        dpi: Output resolution.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Convert to mm for display
    diff_mm = diff_map * 1000.0

    # Symmetric color range
    valid = diff_mm[np.isfinite(diff_mm)]
    if len(valid) > 0:
        vmax = float(np.max(np.abs(valid)))
    else:
        vmax = 1.0
    vmin = -vmax

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="0.8")

    extent = [grid_x[0], grid_x[-1], grid_y[-1], grid_y[0]]
    im = ax.imshow(
        diff_mm,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        origin="upper",
        aspect="equal",
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Z Difference (mm)")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def render_evaluation_summary(
    cloud_distances: np.ndarray | None,
    height_diff_result: dict | None,
    output_dir: str | Path,
    dpi: int = 150,
) -> None:
    """Produce combined evaluation plots.

    Saves to output_dir:
        eval_histograms.png -- distance error histogram
        eval_distance_map.png -- spatial height-map difference

    Skips individual plots if corresponding input is None.

    Args:
        cloud_distances: 1D array of cloud-to-cloud distances (meters).
            From cloud_to_cloud_distance(), use the per-point distances
            (compute with Open3D's compute_point_cloud_distance).
        height_diff_result: Dict from height_map_difference() containing
            "diff_map", "grid_x", "grid_y", and scalar statistics.
        output_dir: Directory to save summary images.
        dpi: Output resolution.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cloud_distances is not None:
        render_error_histogram(
            cloud_distances,
            output_dir / "eval_histograms.png",
            dpi=dpi,
        )

    if height_diff_result is not None:
        render_distance_map(
            height_diff_result["diff_map"],
            height_diff_result["grid_x"],
            height_diff_result["grid_y"],
            output_dir / "eval_distance_map.png",
            dpi=dpi,
        )


def render_timeseries_gallery(
    height_maps: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    output_path: str | Path,
    n_cols: int = 4,
    dpi: int = 150,
) -> None:
    """Create a grid gallery of top-down height-map views across frames.

    Shows surface evolution over time as a grid of colormapped height maps.

    Args:
        height_maps: List of (frame_idx, height_map, grid_x, grid_y) tuples.
            height_map is a 2D array (Ny, Nx) of Z values, NaN for no data.
            grid_x, grid_y define the spatial extent.
        output_path: Path to save the PNG gallery.
        n_cols: Number of columns in the grid.
        dpi: Output resolution.
    """
    n = len(height_maps)
    if n == 0:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))

    # Flatten axes for easy indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).ravel()

    # Shared color range across all frames
    all_z = (
        np.concatenate(
            [
                hm[1][np.isfinite(hm[1])].ravel()
                for hm in height_maps
                if np.isfinite(hm[1]).any()
            ]
        )
        if height_maps
        else np.array([])
    )

    if len(all_z) > 0:
        vmin, vmax = float(all_z.min()), float(all_z.max())
    else:
        vmin, vmax = 0.0, 1.0

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="0.8")

    for i, (frame_idx, hm, gx, gy) in enumerate(height_maps):
        ax = axes[i]
        extent = [gx[0], gx[-1], gy[-1], gy[0]]
        ax.imshow(
            hm,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            origin="upper",
            aspect="equal",
        )
        ax.set_title(f"Frame {frame_idx}", fontsize=9)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Surface Evolution", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
