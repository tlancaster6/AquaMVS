"""Tests for evaluation plots and time-series gallery."""

import matplotlib
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from aquamvs.visualization.summary import (
    render_distance_map,
    render_error_histogram,
    render_evaluation_summary,
    render_timeseries_gallery,
)


def test_matplotlib_backend():
    """Verify matplotlib is configured for headless rendering."""
    assert matplotlib.get_backend() == "Agg", "Expected Agg backend for headless rendering"


def test_render_error_histogram(tmp_path):
    """Render histogram from synthetic distance array."""
    # Normal distribution: mean=5mm, std=1mm (in meters: 0.005, 0.001)
    distances = np.random.normal(0.005, 0.001, 100)
    output_path = tmp_path / "histogram.png"

    render_error_histogram(distances, output_path)

    # Verify file exists and is a valid image
    assert output_path.exists()
    img = Image.open(output_path)
    assert img.size[0] > 0 and img.size[1] > 0


def test_render_error_histogram_zeros(tmp_path):
    """Render histogram with all-zero distances (no crash)."""
    distances = np.zeros(50)
    output_path = tmp_path / "histogram_zeros.png"

    render_error_histogram(distances, output_path)

    assert output_path.exists()


def test_render_distance_map(tmp_path):
    """Render distance map from synthetic diff_map."""
    # Create a 20x30 grid with mixed positive, negative, and NaN
    diff_map = np.random.randn(20, 30) * 0.002  # ±2mm variation
    # Add some NaN regions
    diff_map[0:5, 0:10] = np.nan
    diff_map[15:20, 20:30] = np.nan

    grid_x = np.linspace(0.0, 0.3, 30)
    grid_y = np.linspace(0.0, 0.2, 20)
    output_path = tmp_path / "distance_map.png"

    render_distance_map(diff_map, grid_x, grid_y, output_path)

    # Verify file exists and is a valid image
    assert output_path.exists()
    img = Image.open(output_path)
    assert img.size[0] > 0 and img.size[1] > 0


def test_render_distance_map_all_nan(tmp_path):
    """Render distance map with all-NaN diff_map (no crash)."""
    diff_map = np.full((10, 15), np.nan)
    grid_x = np.linspace(0.0, 0.15, 15)
    grid_y = np.linspace(0.0, 0.10, 10)
    output_path = tmp_path / "distance_map_nan.png"

    render_distance_map(diff_map, grid_x, grid_y, output_path)

    assert output_path.exists()


def test_render_evaluation_summary_full(tmp_path):
    """Render evaluation summary with both distances and height_diff."""
    # Cloud distances
    distances = np.random.normal(0.003, 0.0005, 80)

    # Height diff result
    diff_map = np.random.randn(15, 20) * 0.001
    grid_x = np.linspace(0.0, 0.2, 20)
    grid_y = np.linspace(0.0, 0.15, 15)
    height_diff_result = {
        "diff_map": diff_map,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "mean_diff": 0.0001,
        "std_diff": 0.001,
    }

    output_dir = tmp_path / "summary"

    render_evaluation_summary(distances, height_diff_result, output_dir)

    # Verify both files created
    assert (output_dir / "eval_histograms.png").exists()
    assert (output_dir / "eval_distance_map.png").exists()


def test_render_evaluation_summary_partial_distances(tmp_path):
    """Render evaluation summary with only distances (height_diff=None)."""
    distances = np.random.normal(0.004, 0.0008, 60)
    output_dir = tmp_path / "summary_partial"

    render_evaluation_summary(distances, None, output_dir)

    # Only histogram should be created
    assert (output_dir / "eval_histograms.png").exists()
    assert not (output_dir / "eval_distance_map.png").exists()


def test_render_evaluation_summary_partial_height_diff(tmp_path):
    """Render evaluation summary with only height_diff (distances=None)."""
    diff_map = np.random.randn(12, 18) * 0.0015
    grid_x = np.linspace(0.0, 0.18, 18)
    grid_y = np.linspace(0.0, 0.12, 12)
    height_diff_result = {
        "diff_map": diff_map,
        "grid_x": grid_x,
        "grid_y": grid_y,
    }
    output_dir = tmp_path / "summary_height_only"

    render_evaluation_summary(None, height_diff_result, output_dir)

    # Only distance map should be created
    assert not (output_dir / "eval_histograms.png").exists()
    assert (output_dir / "eval_distance_map.png").exists()


def test_render_timeseries_gallery(tmp_path):
    """Render time-series gallery with 6 frames, n_cols=3."""
    # Create 6 synthetic height maps
    height_maps = []
    for i in range(6):
        frame_idx = i * 10
        hm = np.random.rand(15, 20) * 0.01 + 0.98  # Z ~0.98-0.99m
        # Add some NaN cells
        hm[0:3, 0:5] = np.nan
        grid_x = np.linspace(0.0, 0.2, 20)
        grid_y = np.linspace(0.0, 0.15, 15)
        height_maps.append((frame_idx, hm, grid_x, grid_y))

    output_path = tmp_path / "timeseries.png"

    render_timeseries_gallery(height_maps, output_path, n_cols=3)

    # Verify file exists and dimensions are reasonable
    assert output_path.exists()
    img = Image.open(output_path)
    # With 6 frames and n_cols=3, we expect 2 rows × 3 cols
    # Each subplot is ~4 inches * dpi=150 = 600 pixels wide
    # Total width should be ~3*600 = ~1800 pixels (allowing some margin)
    assert img.size[0] > 1000
    assert img.size[1] > 500


def test_render_timeseries_gallery_empty(tmp_path):
    """Render time-series gallery with empty list (no crash, no file)."""
    output_path = tmp_path / "timeseries_empty.png"

    render_timeseries_gallery([], output_path, n_cols=4)

    # Should return early without creating file
    assert not output_path.exists()


def test_render_timeseries_gallery_single_frame(tmp_path):
    """Render time-series gallery with single frame."""
    hm = np.random.rand(10, 12) * 0.005 + 0.975
    grid_x = np.linspace(0.0, 0.12, 12)
    grid_y = np.linspace(0.0, 0.10, 10)
    height_maps = [(42, hm, grid_x, grid_y)]

    output_path = tmp_path / "timeseries_single.png"

    render_timeseries_gallery(height_maps, output_path, n_cols=4)

    assert output_path.exists()
    img = Image.open(output_path)
    assert img.size[0] > 0 and img.size[1] > 0


def test_render_timeseries_gallery_many_frames(tmp_path):
    """Render time-series gallery with 10 frames, n_cols=4 (3 rows)."""
    height_maps = []
    for i in range(10):
        hm = np.random.rand(8, 10) * 0.008 + 0.97
        grid_x = np.linspace(0.0, 0.1, 10)
        grid_y = np.linspace(0.0, 0.08, 8)
        height_maps.append((i * 5, hm, grid_x, grid_y))

    output_path = tmp_path / "timeseries_many.png"

    render_timeseries_gallery(height_maps, output_path, n_cols=4)

    assert output_path.exists()
    img = Image.open(output_path)
    # 10 frames with n_cols=4 → 3 rows
    # Expect width ~4*4*150 = ~2400 pixels, height ~3*3*150 = ~1350 pixels
    assert img.size[0] > 1500
    assert img.size[1] > 800


def test_render_error_histogram_custom_params(tmp_path):
    """Render histogram with custom title, xlabel, n_bins, dpi."""
    distances = np.random.exponential(0.002, 75)
    output_path = tmp_path / "histogram_custom.png"

    render_error_histogram(
        distances,
        output_path,
        title="Custom Distance Metric",
        xlabel="Error (mm)",
        n_bins=30,
        dpi=100,
    )

    assert output_path.exists()


def test_render_distance_map_custom_params(tmp_path):
    """Render distance map with custom title and dpi."""
    diff_map = np.random.randn(18, 25) * 0.003
    grid_x = np.linspace(-0.1, 0.15, 25)
    grid_y = np.linspace(-0.05, 0.13, 18)
    output_path = tmp_path / "distance_map_custom.png"

    render_distance_map(
        diff_map,
        grid_x,
        grid_y,
        output_path,
        title="Custom Height Difference",
        dpi=100,
    )

    assert output_path.exists()
