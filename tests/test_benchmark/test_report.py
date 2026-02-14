"""Tests for benchmark report generation."""

import pytest

from aquamvs.benchmark.metrics import BenchmarkResults, ConfigResult
from aquamvs.benchmark.report import generate_report


@pytest.fixture
def sample_results():
    """Create synthetic benchmark results for testing."""
    result1 = ConfigResult(
        config_name="superpoint_clahe_off",
        extractor_type="superpoint",
        clahe_enabled=False,
        keypoint_counts={"cam0": 100, "cam1": 150},
        keypoint_mean_scores={"cam0": 0.5, "cam1": 0.6},
        match_counts={("cam0", "cam1"): 50},
        sparse_point_count=40,
        extraction_time=1.0,
        matching_time=0.5,
        triangulation_time=0.2,
        total_time=1.7,
    )

    result2 = ConfigResult(
        config_name="aliked_clahe_on",
        extractor_type="aliked",
        clahe_enabled=True,
        keypoint_counts={"cam0": 200, "cam1": 250},
        keypoint_mean_scores={"cam0": 0.6, "cam1": 0.7},
        match_counts={("cam0", "cam1"): 80},
        sparse_point_count=70,
        extraction_time=1.2,
        matching_time=0.6,
        triangulation_time=0.3,
        total_time=2.1,
    )

    return BenchmarkResults(
        results=[result1, result2],
        frame_idx=0,
        camera_names=["cam0", "cam1"],
        pair_keys=[("cam0", "cam1")],
    )


def test_generate_report_creates_files(tmp_path, sample_results):
    """Test that generate_report creates all expected files."""
    report_path = generate_report(sample_results, tmp_path)

    # Check that report.md exists
    assert report_path.exists()
    assert report_path.name == "report.md"

    # Check that comparison charts exist
    comparison_dir = tmp_path / "benchmark" / "comparison"
    assert (comparison_dir / "keypoint_counts.png").exists()
    assert (comparison_dir / "match_counts.png").exists()
    assert (comparison_dir / "timing.png").exists()


def test_report_content(tmp_path, sample_results):
    """Test that report contains expected sections and data."""
    report_path = generate_report(sample_results, tmp_path)

    content = report_path.read_text()

    # Check for main sections
    assert "# AquaMVS Benchmark Report" in content
    assert "## Summary" in content
    assert "## Per-Stage Timing" in content
    assert "## Per-Camera Keypoints" in content
    assert "## Visualizations" in content

    # Check that config names appear
    assert "superpoint_clahe_off" in content
    assert "aliked_clahe_on" in content

    # Check that visualizations are linked
    assert "![Keypoint Counts](comparison/keypoint_counts.png)" in content
    assert "![Match Counts](comparison/match_counts.png)" in content
    assert "![Timing Breakdown](comparison/timing.png)" in content


def test_report_with_single_config(tmp_path):
    """Test report generation with only one configuration."""
    result = ConfigResult(
        config_name="superpoint_clahe_off",
        extractor_type="superpoint",
        clahe_enabled=False,
        keypoint_counts={"cam0": 100},
        keypoint_mean_scores={"cam0": 0.5},
        match_counts={},
        sparse_point_count=0,
        extraction_time=1.0,
        matching_time=0.5,
        triangulation_time=0.2,
        total_time=1.7,
    )

    results = BenchmarkResults(
        results=[result],
        frame_idx=0,
        camera_names=["cam0"],
        pair_keys=[],
    )

    # Should not raise an error
    report_path = generate_report(results, tmp_path)
    assert report_path.exists()

    # Charts should still be created (even if only 1 bar each)
    comparison_dir = tmp_path / "benchmark" / "comparison"
    assert (comparison_dir / "keypoint_counts.png").exists()


def test_report_includes_new_grids(tmp_path, sample_results):
    """Test that report includes references to the new comparison grids."""
    report_path = generate_report(sample_results, tmp_path)

    content = report_path.read_text()

    # Check that new grid images are referenced
    assert "![Keypoints Grid](comparison/keypoints_grid.png)" in content
    assert "![Sparse Renders Grid](comparison/sparse_renders_grid.png)" in content

    # Check that comparison grid files were created
    comparison_dir = tmp_path / "benchmark" / "comparison"
    assert (comparison_dir / "keypoints_grid.png").exists()
    assert (comparison_dir / "sparse_renders_grid.png").exists()
