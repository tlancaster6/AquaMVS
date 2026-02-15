"""Tests for benchmark runner sweep logic."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from aquamvs.benchmark.runner import run_benchmark
from aquamvs.config import PipelineConfig


@pytest.fixture
def mock_config(tmp_path):
    """Create a minimal PipelineConfig for testing."""
    config = PipelineConfig(
        calibration_path=str(tmp_path / "calibration.json"),
        output_dir=str(tmp_path / "output"),
        camera_video_map={"cam0": str(tmp_path / "video0.mp4")},
    )
    # Override benchmark config for testing
    config.runtime.benchmark_extractors = ["superpoint", "aliked"]
    config.runtime.benchmark_clahe = [True, False]
    return config


@pytest.fixture
def mock_pipeline_context():
    """Create a mock PipelineContext."""
    ctx = MagicMock()
    ctx.calibration.water_z = 1.0
    ctx.calibration.cameras = {"cam0": MagicMock(image_size=(640, 480))}
    ctx.undistortion_maps = {"cam0": MagicMock()}
    ctx.projection_models = {"cam0": MagicMock()}
    ctx.pairs = {"cam0": []}
    ctx.masks = {}
    return ctx


def test_sweep_cross_product(mock_config, mock_pipeline_context):
    """Test that sweep generates cross product of extractors x clahe."""
    with (
        patch("aquamvs.benchmark.runner.setup_pipeline") as mock_setup,
        patch("aquamvs.benchmark.runner.VideoSet") as mock_videoset,
        patch("aquamvs.benchmark.runner.extract_features_batch") as mock_extract,
        patch("aquamvs.benchmark.runner.match_all_pairs") as mock_match,
        patch("aquamvs.benchmark.runner.triangulate_all_pairs") as mock_triangulate,
        patch("aquamvs.benchmark.runner.filter_sparse_cloud") as mock_filter,
        patch("aquamvs.benchmark.runner.render_config_outputs"),
    ):
        # Setup mocks
        mock_setup.return_value = mock_pipeline_context

        # Mock VideoSet context manager and frame iteration
        mock_vs_instance = MagicMock()
        mock_vs_instance.__enter__.return_value = mock_vs_instance
        mock_vs_instance.__exit__.return_value = None
        mock_vs_instance.iterate_frames.return_value = iter(
            [(0, {"cam0": np.zeros((480, 640, 3), dtype=np.uint8)})]
        )
        mock_videoset.return_value = mock_vs_instance

        # Mock undistort_image
        with patch("aquamvs.benchmark.runner.undistort_image") as mock_undistort:
            mock_undistort.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

            # Mock feature extraction
            mock_extract.return_value = {
                "cam0": {
                    "keypoints": torch.zeros((10, 2)),
                    "descriptors": torch.zeros((10, 256)),
                    "scores": torch.ones(10) * 0.5,
                }
            }

            # Mock matching
            mock_match.return_value = {}

            # Mock triangulation
            mock_triangulate.return_value = {
                "points_3d": torch.zeros((5, 3)),
                "scores": torch.ones(5) * 0.8,
            }
            mock_filter.return_value = {
                "points_3d": torch.zeros((4, 3)),
                "scores": torch.ones(4) * 0.8,
            }

            # Run benchmark
            results = run_benchmark(mock_config, frame=0)

            # Check that we got 4 results (2 extractors x 2 clahe settings)
            assert len(results.results) == 4

            # Check config combinations
            config_names = {r.config_name for r in results.results}
            assert config_names == {
                "superpoint_clahe_on",
                "superpoint_clahe_off",
                "aliked_clahe_on",
                "aliked_clahe_off",
            }

            # Check extractor types
            extractors = {r.extractor_type for r in results.results}
            assert extractors == {"superpoint", "aliked"}


def test_timing_is_recorded(mock_config, mock_pipeline_context):
    """Test that timing is recorded for all stages."""
    with (
        patch("aquamvs.benchmark.runner.setup_pipeline") as mock_setup,
        patch("aquamvs.benchmark.runner.VideoSet") as mock_videoset,
        patch("aquamvs.benchmark.runner.extract_features_batch") as mock_extract,
        patch("aquamvs.benchmark.runner.match_all_pairs") as mock_match,
        patch("aquamvs.benchmark.runner.triangulate_all_pairs") as mock_triangulate,
        patch("aquamvs.benchmark.runner.filter_sparse_cloud") as mock_filter,
        patch("aquamvs.benchmark.runner.undistort_image") as mock_undistort,
        patch("aquamvs.benchmark.runner.render_config_outputs"),
    ):
        # Setup mocks
        mock_setup.return_value = mock_pipeline_context
        mock_vs_instance = MagicMock()
        mock_vs_instance.__enter__.return_value = mock_vs_instance
        mock_vs_instance.__exit__.return_value = None
        mock_vs_instance.iterate_frames.return_value = iter(
            [(0, {"cam0": np.zeros((480, 640, 3), dtype=np.uint8)})]
        )
        mock_videoset.return_value = mock_vs_instance
        mock_undistort.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_extract.return_value = {
            "cam0": {
                "keypoints": torch.zeros((10, 2)),
                "descriptors": torch.zeros((10, 256)),
                "scores": torch.ones(10) * 0.5,
            }
        }
        mock_match.return_value = {}
        mock_triangulate.return_value = {
            "points_3d": torch.zeros((5, 3)),
            "scores": torch.ones(5) * 0.8,
        }
        mock_filter.return_value = {
            "points_3d": torch.zeros((4, 3)),
            "scores": torch.ones(4) * 0.8,
        }

        # Run benchmark
        results = run_benchmark(mock_config, frame=0)

        # Check that all results have timing data
        for result in results.results:
            assert result.extraction_time > 0
            assert result.matching_time >= 0  # Can be 0 if mocked fast enough
            assert result.triangulation_time >= 0
            assert (
                result.total_time
                >= result.extraction_time
                + result.matching_time
                + result.triangulation_time
            )


def test_metrics_collected(mock_config, mock_pipeline_context):
    """Test that metrics are collected from pipeline outputs."""
    with (
        patch("aquamvs.benchmark.runner.setup_pipeline") as mock_setup,
        patch("aquamvs.benchmark.runner.VideoSet") as mock_videoset,
        patch("aquamvs.benchmark.runner.extract_features_batch") as mock_extract,
        patch("aquamvs.benchmark.runner.match_all_pairs") as mock_match,
        patch("aquamvs.benchmark.runner.triangulate_all_pairs") as mock_triangulate,
        patch("aquamvs.benchmark.runner.filter_sparse_cloud") as mock_filter,
        patch("aquamvs.benchmark.runner.undistort_image") as mock_undistort,
        patch("aquamvs.benchmark.runner.render_config_outputs"),
    ):
        # Setup mocks
        mock_setup.return_value = mock_pipeline_context
        mock_vs_instance = MagicMock()
        mock_vs_instance.__enter__.return_value = mock_vs_instance
        mock_vs_instance.__exit__.return_value = None
        mock_vs_instance.iterate_frames.return_value = iter(
            [(0, {"cam0": np.zeros((480, 640, 3), dtype=np.uint8)})]
        )
        mock_videoset.return_value = mock_vs_instance
        mock_undistort.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        # Mock with specific counts
        mock_extract.return_value = {
            "cam0": {
                "keypoints": torch.zeros((15, 2)),
                "descriptors": torch.zeros((15, 256)),
                "scores": torch.ones(15) * 0.6,
            }
        }
        mock_match.return_value = {
            ("cam0", "cam1"): {
                "ref_keypoints": torch.zeros((8, 2)),
                "src_keypoints": torch.zeros((8, 2)),
            }
        }
        mock_triangulate.return_value = {
            "points_3d": torch.zeros((7, 3)),
            "scores": torch.ones(7) * 0.8,
        }
        mock_filter.return_value = {
            "points_3d": torch.zeros((6, 3)),
            "scores": torch.ones(6) * 0.8,
        }

        # Run benchmark
        results = run_benchmark(mock_config, frame=0)

        # Check that metrics are populated
        for result in results.results:
            assert len(result.keypoint_counts) > 0
            assert len(result.keypoint_mean_scores) > 0
            assert result.sparse_point_count == 6  # Filtered count


def test_masks_propagate(mock_config, mock_pipeline_context):
    """Test that masks are applied during the sweep."""
    # Add a mask to the context
    mock_pipeline_context.masks = {"cam0": np.ones((480, 640), dtype=np.uint8) * 255}

    with (
        patch("aquamvs.benchmark.runner.setup_pipeline") as mock_setup,
        patch("aquamvs.benchmark.runner.VideoSet") as mock_videoset,
        patch("aquamvs.benchmark.runner.extract_features_batch") as mock_extract,
        patch("aquamvs.benchmark.runner.match_all_pairs") as mock_match,
        patch("aquamvs.benchmark.runner.triangulate_all_pairs") as mock_triangulate,
        patch("aquamvs.benchmark.runner.filter_sparse_cloud") as mock_filter,
        patch("aquamvs.benchmark.runner.undistort_image") as mock_undistort,
        patch("aquamvs.benchmark.runner.apply_mask_to_features") as mock_apply_mask,
        patch("aquamvs.benchmark.runner.render_config_outputs"),
    ):
        # Setup mocks
        mock_setup.return_value = mock_pipeline_context
        mock_vs_instance = MagicMock()
        mock_vs_instance.__enter__.return_value = mock_vs_instance
        mock_vs_instance.__exit__.return_value = None
        mock_vs_instance.iterate_frames.return_value = iter(
            [(0, {"cam0": np.zeros((480, 640, 3), dtype=np.uint8)})]
        )
        mock_videoset.return_value = mock_vs_instance
        mock_undistort.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_extract.return_value = {
            "cam0": {
                "keypoints": torch.zeros((10, 2)),
                "descriptors": torch.zeros((10, 256)),
                "scores": torch.ones(10) * 0.5,
            }
        }
        # apply_mask_to_features should return the same structure
        mock_apply_mask.return_value = {
            "keypoints": torch.zeros((8, 2)),  # Fewer after masking
            "descriptors": torch.zeros((8, 256)),
            "scores": torch.ones(8) * 0.5,
        }
        mock_match.return_value = {}
        mock_triangulate.return_value = {
            "points_3d": torch.zeros((5, 3)),
            "scores": torch.ones(5) * 0.8,
        }
        mock_filter.return_value = {
            "points_3d": torch.zeros((4, 3)),
            "scores": torch.ones(4) * 0.8,
        }

        # Run benchmark
        run_benchmark(mock_config, frame=0)

        # Check that apply_mask_to_features was called
        # Should be called once per config (4 configs)
        assert mock_apply_mask.call_count == 4


def test_single_config_sweep(mock_config, mock_pipeline_context):
    """Test sweep with only one configuration."""
    # Set benchmark to single extractor and single clahe setting
    mock_config.runtime.benchmark_extractors = ["superpoint"]
    mock_config.runtime.benchmark_clahe = [False]

    with (
        patch("aquamvs.benchmark.runner.setup_pipeline") as mock_setup,
        patch("aquamvs.benchmark.runner.VideoSet") as mock_videoset,
        patch("aquamvs.benchmark.runner.extract_features_batch") as mock_extract,
        patch("aquamvs.benchmark.runner.match_all_pairs") as mock_match,
        patch("aquamvs.benchmark.runner.triangulate_all_pairs") as mock_triangulate,
        patch("aquamvs.benchmark.runner.filter_sparse_cloud") as mock_filter,
        patch("aquamvs.benchmark.runner.undistort_image") as mock_undistort,
        patch("aquamvs.benchmark.runner.render_config_outputs"),
    ):
        # Setup mocks
        mock_setup.return_value = mock_pipeline_context
        mock_vs_instance = MagicMock()
        mock_vs_instance.__enter__.return_value = mock_vs_instance
        mock_vs_instance.__exit__.return_value = None
        mock_vs_instance.iterate_frames.return_value = iter(
            [(0, {"cam0": np.zeros((480, 640, 3), dtype=np.uint8)})]
        )
        mock_videoset.return_value = mock_vs_instance
        mock_undistort.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_extract.return_value = {
            "cam0": {
                "keypoints": torch.zeros((10, 2)),
                "descriptors": torch.zeros((10, 256)),
                "scores": torch.ones(10) * 0.5,
            }
        }
        mock_match.return_value = {}
        mock_triangulate.return_value = {
            "points_3d": torch.zeros((5, 3)),
            "scores": torch.ones(5) * 0.8,
        }
        mock_filter.return_value = {
            "points_3d": torch.zeros((4, 3)),
            "scores": torch.ones(4) * 0.8,
        }

        # Run benchmark
        results = run_benchmark(mock_config, frame=0)

        # Should have exactly 1 result
        assert len(results.results) == 1
        assert results.results[0].config_name == "superpoint_clahe_off"
        assert results.results[0].extractor_type == "superpoint"
        assert results.results[0].clahe_enabled is False
