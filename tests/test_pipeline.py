"""Tests for pipeline orchestration."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import open3d as o3d
import pytest
import torch

from aquamvs.calibration import CalibrationData, CameraData, UndistortionData
from aquamvs.config import (
    DenseStereoConfig,
    DeviceConfig,
    FeatureExtractionConfig,
    FrameSamplingConfig,
    FusionConfig,
    MatchingConfig,
    PairSelectionConfig,
    PipelineConfig,
    SurfaceConfig,
)
from aquamvs.pipeline import (
    PipelineContext,
    process_frame,
    run_pipeline,
    setup_pipeline,
)


@pytest.fixture
def mock_calibration_data():
    """Create a minimal CalibrationData with 3 cameras for testing."""
    cameras = {}
    for i, cam_name in enumerate(["cam0", "cam1", "cam2"]):
        cameras[cam_name] = CameraData(
            name=cam_name,
            K=torch.eye(3, dtype=torch.float32),
            dist_coeffs=torch.zeros(5, dtype=torch.float64),
            R=torch.eye(3, dtype=torch.float32),
            t=torch.tensor([i * 0.1, 0.0, 0.0], dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=(cam_name == "cam2"),
        )

    return CalibrationData(
        cameras=cameras,
        water_z=0.978,
        interface_normal=torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),
        n_air=1.0,
        n_water=1.333,
    )


@pytest.fixture
def pipeline_config(tmp_path):
    """Create a minimal PipelineConfig for testing."""
    return PipelineConfig(
        calibration_path="dummy_calibration.json",
        output_dir=str(tmp_path / "output"),
        camera_video_map={
            "cam0": "video0.mp4",
            "cam1": "video1.mp4",
            "cam2": "video2.mp4",
        },
        frame_sampling=FrameSamplingConfig(start=0, stop=10, step=1),
        feature_extraction=FeatureExtractionConfig(),
        pair_selection=PairSelectionConfig(),
        matching=MatchingConfig(),
        dense_stereo=DenseStereoConfig(),
        fusion=FusionConfig(),
        surface=SurfaceConfig(),
        device=DeviceConfig(device="cpu"),
    )


def test_setup_pipeline_structure(pipeline_config, mock_calibration_data, tmp_path):
    """Test that setup_pipeline creates PipelineContext with correct structure."""
    with patch(
        "aquamvs.pipeline.load_calibration_data", return_value=mock_calibration_data
    ):
        with patch("aquamvs.pipeline.select_pairs") as mock_select_pairs:
            # Mock select_pairs to return simple pair structure
            mock_select_pairs.return_value = {
                "cam0": ["cam1", "cam2"],
                "cam1": ["cam0", "cam2"],
            }

            ctx = setup_pipeline(pipeline_config)

            # Verify context structure
            assert isinstance(ctx, PipelineContext)
            assert ctx.config == pipeline_config
            assert ctx.calibration == mock_calibration_data
            assert len(ctx.undistortion_maps) == 3
            assert len(ctx.projection_models) == 3
            assert len(ctx.ring_cameras) == 2  # cam0, cam1
            assert len(ctx.auxiliary_cameras) == 1  # cam2
            assert ctx.device == "cpu"

            # Verify config copy was saved
            config_path = Path(pipeline_config.output_dir) / "config.yaml"
            assert config_path.exists()


def test_setup_pipeline_uses_undistorted_k(pipeline_config, mock_calibration_data):
    """Test that projection models use K_new from undistortion, not original K."""
    with patch(
        "aquamvs.pipeline.load_calibration_data", return_value=mock_calibration_data
    ):
        with patch("aquamvs.pipeline.select_pairs") as mock_select_pairs:
            mock_select_pairs.return_value = {"cam0": ["cam1"]}

            with patch(
                "aquamvs.pipeline.compute_undistortion_maps"
            ) as mock_compute_undist:
                # Mock undistortion to return a different K_new
                def make_undist_data(cam):
                    K_new = cam.K + 0.1  # Make K_new different from K
                    return UndistortionData(
                        K_new=K_new,
                        map_x=np.zeros((480, 640), dtype=np.float32),
                        map_y=np.zeros((480, 640), dtype=np.float32),
                    )

                mock_compute_undist.side_effect = make_undist_data

                with patch(
                    "aquamvs.pipeline.RefractiveProjectionModel"
                ) as mock_proj_model:
                    ctx = setup_pipeline(pipeline_config)

                    # Verify RefractiveProjectionModel was called with K_new
                    assert mock_proj_model.call_count == 3
                    for call in mock_proj_model.call_args_list:
                        K_used = call[1]["K"]
                        # K_new should be different from original K
                        assert not torch.allclose(K_used, torch.eye(3))


def test_process_frame_directory_structure(
    pipeline_config, mock_calibration_data, tmp_path
):
    """Test that process_frame creates correct output directory structure."""
    # Create mock projection models
    mock_proj_model = Mock()
    projection_models = {
        "cam0": mock_proj_model,
        "cam1": mock_proj_model,
        "cam2": mock_proj_model,
    }

    # Create mock undistortion data
    undistortion_maps = {
        "cam0": UndistortionData(
            K_new=torch.eye(3, dtype=torch.float32),
            map_x=np.zeros((480, 640), dtype=np.float32),
            map_y=np.zeros((480, 640), dtype=np.float32),
        ),
        "cam1": UndistortionData(
            K_new=torch.eye(3, dtype=torch.float32),
            map_x=np.zeros((480, 640), dtype=np.float32),
            map_y=np.zeros((480, 640), dtype=np.float32),
        ),
    }

    # Create a mock context
    ctx = PipelineContext(
        config=pipeline_config,
        calibration=mock_calibration_data,
        undistortion_maps=undistortion_maps,
        projection_models=projection_models,
        pairs={"cam0": ["cam1"]},
        ring_cameras=["cam0", "cam1"],
        auxiliary_cameras=["cam2"],
        device="cpu",
    )

    # Mock all pipeline stages
    with patch("aquamvs.pipeline.undistort_image") as mock_undist:
        mock_undist.side_effect = lambda img, _: img  # Pass through

        with patch("aquamvs.pipeline.extract_features_batch") as mock_extract:
            mock_extract.return_value = {}

            with patch("aquamvs.pipeline.match_all_pairs") as mock_match:
                mock_match.return_value = {}

                with patch("aquamvs.pipeline.triangulate_all_pairs") as mock_tri:
                    mock_tri.return_value = {
                        "points_3d": torch.zeros(0, 3),
                        "scores": torch.zeros(0),
                    }

                    with patch("aquamvs.pipeline.compute_depth_ranges") as mock_ranges:
                        mock_ranges.return_value = {"cam0": (0.3, 0.9)}

                        with patch("aquamvs.pipeline.plane_sweep_stereo") as mock_sweep:
                            mock_sweep.return_value = {
                                "cost_volume": torch.zeros(1, 1, 1),
                                "depths": torch.tensor([0.5]),
                            }

                            with patch("aquamvs.pipeline.extract_depth") as mock_extr:
                                mock_extr.return_value = (
                                    torch.full((480, 640), float("nan")),
                                    torch.zeros(480, 640),
                                )

                                with patch(
                                    "aquamvs.pipeline.filter_all_depth_maps"
                                ) as mock_filter:
                                    mock_filter.return_value = {
                                        "cam0": (
                                            torch.full((480, 640), float("nan")),
                                            torch.zeros(480, 640),
                                            torch.zeros(480, 640, dtype=torch.int32),
                                        )
                                    }

                                    with patch(
                                        "aquamvs.pipeline.fuse_depth_maps"
                                    ) as mock_fuse:
                                        mock_fuse.return_value = (
                                            o3d.geometry.PointCloud()
                                        )

                                        with patch(
                                            "aquamvs.pipeline.reconstruct_surface"
                                        ) as mock_surf:
                                            mock_surf.return_value = (
                                                o3d.geometry.TriangleMesh()
                                            )

                                            # Call process_frame
                                            raw_images = {
                                                "cam0": np.zeros(
                                                    (480, 640, 3), dtype=np.uint8
                                                ),
                                                "cam1": np.zeros(
                                                    (480, 640, 3), dtype=np.uint8
                                                ),
                                            }
                                            process_frame(0, raw_images, ctx)

    # Verify directory structure
    frame_dir = Path(pipeline_config.output_dir) / "frame_000000"
    assert frame_dir.exists()
    assert (frame_dir / "sparse").exists()
    assert (frame_dir / "depth_maps").exists()
    assert (frame_dir / "point_cloud").exists()
    assert (frame_dir / "mesh").exists()


def test_process_frame_no_valid_images(pipeline_config, mock_calibration_data, caplog):
    """Test that process_frame handles empty image dict gracefully."""
    ctx = PipelineContext(
        config=pipeline_config,
        calibration=mock_calibration_data,
        undistortion_maps={},
        projection_models={},
        pairs={},
        ring_cameras=[],
        auxiliary_cameras=[],
        device="cpu",
    )

    with caplog.at_level(logging.WARNING):
        # Empty dict (all cameras failed)
        process_frame(0, {}, ctx)

    # Should log a warning and return early
    assert "no valid images" in caplog.text


def test_process_frame_with_none_images(pipeline_config, mock_calibration_data, caplog):
    """Test that process_frame filters out None values from failed camera reads."""
    ctx = PipelineContext(
        config=pipeline_config,
        calibration=mock_calibration_data,
        undistortion_maps={},
        projection_models={},
        pairs={},
        ring_cameras=[],
        auxiliary_cameras=[],
        device="cpu",
    )

    with caplog.at_level(logging.WARNING):
        # Dict with None values (some cameras failed)
        raw_images = {
            "cam0": None,
            "cam1": None,
        }
        process_frame(0, raw_images, ctx)

    # Should log a warning and return early
    assert "no valid images" in caplog.text


def test_run_pipeline_videoset_integration(pipeline_config):
    """Test that run_pipeline iterates over VideoSet frames correctly."""
    with patch("aquamvs.pipeline.setup_pipeline") as mock_setup:
        mock_ctx = Mock()
        mock_setup.return_value = mock_ctx

        with patch("aquamvs.pipeline.VideoSet") as mock_videoset:
            # Mock VideoSet to yield 2 frames
            mock_videos = MagicMock()
            mock_videos.__enter__.return_value = mock_videos
            mock_videos.iterate_frames.return_value = [
                (0, {"cam0": np.zeros((480, 640, 3), dtype=np.uint8)}),
                (5, {"cam0": np.zeros((480, 640, 3), dtype=np.uint8)}),
            ]
            mock_videoset.return_value = mock_videos

            with patch("aquamvs.pipeline.process_frame") as mock_process:
                run_pipeline(pipeline_config)

                # Verify process_frame was called twice with correct indices
                assert mock_process.call_count == 2
                assert mock_process.call_args_list[0][0][0] == 0  # frame_idx
                assert mock_process.call_args_list[1][0][0] == 5  # frame_idx


def test_run_pipeline_handles_frame_failure(pipeline_config, caplog):
    """Test that run_pipeline continues after a frame processing failure."""
    with patch("aquamvs.pipeline.setup_pipeline") as mock_setup:
        mock_ctx = Mock()
        mock_setup.return_value = mock_ctx

        with patch("aquamvs.pipeline.VideoSet") as mock_videoset:
            # Mock VideoSet to yield 3 frames
            mock_videos = MagicMock()
            mock_videos.__enter__.return_value = mock_videos
            mock_videos.iterate_frames.return_value = [
                (0, {"cam0": np.zeros((480, 640, 3), dtype=np.uint8)}),
                (1, {"cam0": np.zeros((480, 640, 3), dtype=np.uint8)}),
                (2, {"cam0": np.zeros((480, 640, 3), dtype=np.uint8)}),
            ]
            mock_videoset.return_value = mock_videos

            with patch("aquamvs.pipeline.process_frame") as mock_process:
                # Make frame 1 fail
                def process_side_effect(idx, images, ctx):
                    if idx == 1:
                        raise RuntimeError("Simulated failure")

                mock_process.side_effect = process_side_effect

                with caplog.at_level(logging.ERROR):
                    run_pipeline(pipeline_config)

                # Should have attempted all 3 frames
                assert mock_process.call_count == 3
                # Should have logged the failure
                assert "processing failed" in caplog.text


def test_pipeline_context_fields(pipeline_config, mock_calibration_data):
    """Test that PipelineContext has all required fields populated."""
    with patch(
        "aquamvs.pipeline.load_calibration_data", return_value=mock_calibration_data
    ):
        with patch("aquamvs.pipeline.select_pairs") as mock_select_pairs:
            mock_select_pairs.return_value = {"cam0": ["cam1"]}

            ctx = setup_pipeline(pipeline_config)

            # Verify all fields are present and correct types
            assert isinstance(ctx.config, PipelineConfig)
            assert isinstance(ctx.calibration, CalibrationData)
            assert isinstance(ctx.undistortion_maps, dict)
            assert isinstance(ctx.projection_models, dict)
            assert isinstance(ctx.pairs, dict)
            assert isinstance(ctx.ring_cameras, list)
            assert isinstance(ctx.auxiliary_cameras, list)
            assert isinstance(ctx.device, str)

            # Verify camera categorization
            assert "cam0" in ctx.ring_cameras
            assert "cam1" in ctx.ring_cameras
            assert "cam2" in ctx.auxiliary_cameras
