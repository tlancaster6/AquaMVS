"""Tests for benchmark metrics and helper functions."""

from aquamvs.benchmark.metrics import (
    ConfigResult,
    config_name,
    total_keypoints,
    total_matches,
)


def test_config_name():
    """Test config_name generates expected strings."""
    assert config_name("superpoint", True) == "superpoint_clahe_on"
    assert config_name("superpoint", False) == "superpoint_clahe_off"
    assert config_name("aliked", True) == "aliked_clahe_on"
    assert config_name("aliked", False) == "aliked_clahe_off"
    assert config_name("disk", True) == "disk_clahe_on"
    assert config_name("disk", False) == "disk_clahe_off"


def test_total_keypoints():
    """Test total_keypoints sums across cameras."""
    result = ConfigResult(
        config_name="test",
        extractor_type="superpoint",
        clahe_enabled=False,
        keypoint_counts={"cam0": 100, "cam1": 200, "cam2": 50},
        keypoint_mean_scores={"cam0": 0.5, "cam1": 0.6, "cam2": 0.4},
        match_counts={},
        sparse_point_count=0,
        extraction_time=0.0,
        matching_time=0.0,
        triangulation_time=0.0,
        total_time=0.0,
    )

    assert total_keypoints(result) == 350


def test_total_matches():
    """Test total_matches sums across pairs."""
    result = ConfigResult(
        config_name="test",
        extractor_type="superpoint",
        clahe_enabled=False,
        keypoint_counts={},
        keypoint_mean_scores={},
        match_counts={
            ("cam0", "cam1"): 50,
            ("cam0", "cam2"): 30,
            ("cam1", "cam2"): 40,
        },
        sparse_point_count=0,
        extraction_time=0.0,
        matching_time=0.0,
        triangulation_time=0.0,
        total_time=0.0,
    )

    assert total_matches(result) == 120
