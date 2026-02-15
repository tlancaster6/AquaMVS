"""Tests for pipeline orchestration."""

import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import open3d as o3d
import pytest
import torch

from aquamvs.calibration import CalibrationData, CameraData, UndistortionData
from aquamvs.config import (
    PipelineConfig,
    PreprocessingConfig,
    ReconstructionConfig,
    RuntimeConfig,
    SparseMatchingConfig,
)
from aquamvs.pipeline import (
    PipelineContext,
    process_frame,
    run_pipeline,
    setup_pipeline,
)
from aquamvs.pipeline.helpers import _should_viz


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
        preprocessing=PreprocessingConfig(frame_start=0, frame_stop=10, frame_step=1),
        sparse_matching=SparseMatchingConfig(),
        reconstruction=ReconstructionConfig(),
        runtime=RuntimeConfig(device="cpu"),
    )


def test_setup_pipeline_structure(pipeline_config, mock_calibration_data, tmp_path):
    """Test that setup_pipeline creates PipelineContext with correct structure."""
    with (
        patch(
            "aquamvs.pipeline.builder.load_calibration_data",
            return_value=mock_calibration_data,
        ),
        patch("aquamvs.pipeline.builder.select_pairs") as mock_select_pairs,
    ):
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
    with (
        patch(
            "aquamvs.pipeline.builder.load_calibration_data",
            return_value=mock_calibration_data,
        ),
        patch("aquamvs.pipeline.builder.select_pairs") as mock_select_pairs,
    ):
        mock_select_pairs.return_value = {"cam0": ["cam1"]}

        with patch(
            "aquamvs.pipeline.builder.compute_undistortion_maps"
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
                "aquamvs.pipeline.builder.RefractiveProjectionModel"
            ) as mock_proj_model:
                setup_pipeline(pipeline_config)

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
        masks={},
    )

    # Mock all pipeline stages
    with patch("aquamvs.pipeline.stages.undistortion.undistort_image") as mock_undist:
        mock_undist.side_effect = lambda img, _: img  # Pass through

        with patch(
            "aquamvs.pipeline.stages.sparse_matching.extract_features_batch"
        ) as mock_extract:
            mock_extract.return_value = {}

            with patch(
                "aquamvs.pipeline.stages.sparse_matching.match_all_pairs"
            ) as mock_match:
                mock_match.return_value = {}

                with patch(
                    "aquamvs.pipeline.stages.sparse_matching.triangulate_all_pairs"
                ) as mock_tri:
                    mock_tri.return_value = {
                        "points_3d": torch.zeros(0, 3),
                        "scores": torch.zeros(0),
                    }

                    with patch(
                        "aquamvs.pipeline.stages.sparse_matching.compute_depth_ranges"
                    ) as mock_ranges:
                        mock_ranges.return_value = {"cam0": (0.3, 0.9)}

                        with patch(
                            "aquamvs.pipeline.stages.depth_estimation.plane_sweep_stereo"
                        ) as mock_sweep:
                            mock_sweep.return_value = {
                                "cost_volume": torch.zeros(1, 1, 1),
                                "depths": torch.tensor([0.5]),
                            }

                            with patch(
                                "aquamvs.pipeline.stages.depth_estimation.extract_depth"
                            ) as mock_extr:
                                mock_extr.return_value = (
                                    torch.full((480, 640), float("nan")),
                                    torch.zeros(480, 640),
                                )

                                with patch(
                                    "aquamvs.pipeline.stages.fusion.filter_all_depth_maps"
                                ) as mock_filter:
                                    mock_filter.return_value = {
                                        "cam0": (
                                            torch.full((480, 640), float("nan")),
                                            torch.zeros(480, 640),
                                            torch.zeros(480, 640, dtype=torch.int32),
                                        )
                                    }

                                    with patch(
                                        "aquamvs.pipeline.stages.fusion.fuse_depth_maps"
                                    ) as mock_fuse:
                                        # Return non-empty point cloud
                                        pcd = o3d.geometry.PointCloud()
                                        pcd.points = o3d.utility.Vector3dVector(
                                            [[0, 0, 1], [0.1, 0, 1]]
                                        )
                                        mock_fuse.return_value = pcd

                                        with patch(
                                            "aquamvs.pipeline.stages.surface.reconstruct_surface"
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
        masks={},
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
        masks={},
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
    with patch("aquamvs.pipeline.runner.build_pipeline_context") as mock_setup:
        mock_ctx = Mock()
        mock_setup.return_value = mock_ctx

        with patch("aquamvs.pipeline.runner.VideoSet") as mock_videoset:
            # Mock VideoSet to yield 2 frames
            mock_videos = MagicMock()
            mock_videos.__enter__.return_value = mock_videos
            mock_videos.iterate_frames.return_value = [
                (0, {"cam0": np.zeros((480, 640, 3), dtype=np.uint8)}),
                (5, {"cam0": np.zeros((480, 640, 3), dtype=np.uint8)}),
            ]
            mock_videoset.return_value = mock_videos

            with patch("aquamvs.pipeline.runner.process_frame") as mock_process:
                run_pipeline(pipeline_config)

                # Verify process_frame was called twice with correct indices
                assert mock_process.call_count == 2
                assert mock_process.call_args_list[0][0][0] == 0  # frame_idx
                assert mock_process.call_args_list[1][0][0] == 5  # frame_idx


def test_run_pipeline_handles_frame_failure(pipeline_config, caplog):
    """Test that run_pipeline continues after a frame processing failure."""
    with patch("aquamvs.pipeline.runner.build_pipeline_context") as mock_setup:
        mock_ctx = Mock()
        mock_setup.return_value = mock_ctx

        with patch("aquamvs.pipeline.runner.VideoSet") as mock_videoset:
            # Mock VideoSet to yield 3 frames
            mock_videos = MagicMock()
            mock_videos.__enter__.return_value = mock_videos
            mock_videos.iterate_frames.return_value = [
                (0, {"cam0": np.zeros((480, 640, 3), dtype=np.uint8)}),
                (1, {"cam0": np.zeros((480, 640, 3), dtype=np.uint8)}),
                (2, {"cam0": np.zeros((480, 640, 3), dtype=np.uint8)}),
            ]
            mock_videoset.return_value = mock_videos

            with patch("aquamvs.pipeline.runner.process_frame") as mock_process:
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
    with (
        patch(
            "aquamvs.pipeline.builder.load_calibration_data",
            return_value=mock_calibration_data,
        ),
        patch("aquamvs.pipeline.builder.select_pairs") as mock_select_pairs,
    ):
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


# ---------------------------------------------------------------------------
# Helpers for process_frame tests with full mocking
# ---------------------------------------------------------------------------


@contextmanager
def _mock_pipeline_stages():
    """Context manager that patches all pipeline stage functions.

    Yields a dict of mock objects keyed by short names.
    """
    with (
        patch("aquamvs.pipeline.stages.undistortion.undistort_image") as m_undist,
        patch(
            "aquamvs.pipeline.stages.sparse_matching.extract_features_batch"
        ) as m_extract,
        patch("aquamvs.pipeline.stages.sparse_matching.match_all_pairs") as m_match,
        patch("aquamvs.pipeline.stages.sparse_matching.triangulate_all_pairs") as m_tri,
        patch(
            "aquamvs.pipeline.stages.sparse_matching.filter_sparse_cloud"
        ) as m_filter_sparse,
        patch(
            "aquamvs.pipeline.stages.sparse_matching.compute_depth_ranges"
        ) as m_ranges,
        patch(
            "aquamvs.pipeline.stages.surface._sparse_cloud_to_open3d"
        ) as m_sparse_to_o3d,
        patch("aquamvs.pipeline.stages.depth_estimation.plane_sweep_stereo") as m_sweep,
        patch("aquamvs.pipeline.stages.depth_estimation.extract_depth") as m_extr,
        patch("aquamvs.pipeline.stages.fusion.filter_all_depth_maps") as m_filter,
        patch("aquamvs.pipeline.stages.fusion.fuse_depth_maps") as m_fuse,
        patch("aquamvs.pipeline.stages.surface.reconstruct_surface") as m_surf,
        patch(
            "aquamvs.pipeline.stages.sparse_matching.save_sparse_cloud"
        ) as m_save_sparse,
        patch(
            "aquamvs.pipeline.stages.depth_estimation.save_depth_map"
        ) as m_save_depth,
        patch("aquamvs.pipeline.stages.fusion.save_point_cloud") as m_save_pcd,
        patch("aquamvs.pipeline.stages.surface.save_point_cloud") as m_save_pcd_sparse,
        patch("aquamvs.pipeline.stages.surface.save_mesh") as m_save_mesh,
    ):
        m_undist.side_effect = lambda img, _: img  # Pass through
        m_extract.return_value = {
            "cam0": {
                "keypoints": torch.zeros(10, 2),
                "scores": torch.ones(10),
                "descriptors": torch.zeros(10, 256),
            },
        }
        m_match.return_value = {
            ("cam0", "cam1"): {
                "ref_keypoints": torch.zeros(5, 2),
                "src_keypoints": torch.zeros(5, 2),
                "scores": torch.ones(5),
            },
        }
        m_tri.return_value = {
            "points_3d": torch.zeros(0, 3),
            "scores": torch.zeros(0),
        }
        m_filter_sparse.side_effect = lambda cloud, **kwargs: cloud  # Pass through
        m_ranges.return_value = {"cam0": (0.3, 0.9)}
        # Return non-empty point cloud by default for sparse mode
        sparse_pcd = o3d.geometry.PointCloud()
        sparse_pcd.points = o3d.utility.Vector3dVector([[0, 0, 1], [0.1, 0, 1]])
        m_sparse_to_o3d.return_value = sparse_pcd
        m_sweep.return_value = {
            "cost_volume": torch.zeros(1, 1, 1),
            "depths": torch.tensor([0.5]),
        }
        m_extr.return_value = (
            torch.full((480, 640), float("nan")),
            torch.zeros(480, 640),
        )
        m_filter.return_value = {
            "cam0": (
                torch.full((480, 640), float("nan")),
                torch.zeros(480, 640),
                torch.zeros(480, 640, dtype=torch.int32),
            )
        }
        # Return non-empty point cloud by default
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector([[0, 0, 1], [0.1, 0, 1]])
        m_fuse.return_value = pcd
        m_surf.return_value = o3d.geometry.TriangleMesh()

        yield {
            "undistort": m_undist,
            "extract": m_extract,
            "match": m_match,
            "triangulate": m_tri,
            "filter_sparse": m_filter_sparse,
            "depth_ranges": m_ranges,
            "sparse_to_o3d": m_sparse_to_o3d,
            "sweep": m_sweep,
            "extract_depth": m_extr,
            "filter": m_filter,
            "fuse": m_fuse,
            "surface": m_surf,
            "save_sparse": m_save_sparse,
            "save_depth": m_save_depth,
            "save_pcd": m_save_pcd,
            "save_pcd_sparse": m_save_pcd_sparse,
            "save_mesh": m_save_mesh,
        }


def _make_ctx(config, calibration_data, masks=None):
    """Create a PipelineContext for testing with minimal mocks.

    Args:
        config: PipelineConfig instance.
        calibration_data: CalibrationData instance.
        masks: Optional dict of camera_name -> mask array. Defaults to empty dict.
    """
    if masks is None:
        masks = {}

    mock_proj_model = Mock()
    return PipelineContext(
        config=config,
        calibration=calibration_data,
        undistortion_maps={
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
        },
        projection_models={
            "cam0": mock_proj_model,
            "cam1": mock_proj_model,
            "cam2": mock_proj_model,
        },
        pairs={"cam0": ["cam1"]},
        ring_cameras=["cam0", "cam1"],
        auxiliary_cameras=["cam2"],
        device="cpu",
        masks=masks,
    )


_RAW_IMAGES = {
    "cam0": np.zeros((480, 640, 3), dtype=np.uint8),
    "cam1": np.zeros((480, 640, 3), dtype=np.uint8),
}


# ---------------------------------------------------------------------------
# _should_viz helper unit tests
# ---------------------------------------------------------------------------


class TestShouldViz:
    """Unit tests for the _should_viz helper."""

    def test_disabled_always_false(self, tmp_path):
        """enabled=False returns False for any stage."""
        config = PipelineConfig(
            output_dir=str(tmp_path),
            runtime=RuntimeConfig(viz_enabled=False),
        )
        assert _should_viz(config, "depth") is False
        assert _should_viz(config, "features") is False
        assert _should_viz(config, "scene") is False
        assert _should_viz(config, "rig") is False
        assert _should_viz(config, "summary") is False

    def test_enabled_empty_stages_all_true(self, tmp_path):
        """enabled=True with stages=[] returns True for all stages."""
        config = PipelineConfig(
            output_dir=str(tmp_path),
            runtime=RuntimeConfig(viz_enabled=True, viz_stages=[]),
        )
        assert _should_viz(config, "depth") is True
        assert _should_viz(config, "features") is True
        assert _should_viz(config, "scene") is True
        assert _should_viz(config, "rig") is True
        assert _should_viz(config, "summary") is True

    def test_enabled_specific_stages(self, tmp_path):
        """enabled=True with specific stages returns True only for those."""
        config = PipelineConfig(
            output_dir=str(tmp_path),
            runtime=RuntimeConfig(viz_enabled=True, viz_stages=["depth", "scene"]),
        )
        assert _should_viz(config, "depth") is True
        assert _should_viz(config, "scene") is True
        assert _should_viz(config, "features") is False
        assert _should_viz(config, "rig") is False
        assert _should_viz(config, "summary") is False


# ---------------------------------------------------------------------------
# Visualization integration tests
# ---------------------------------------------------------------------------


class TestVizIntegration:
    """Tests for visualization wiring in process_frame."""

    def test_viz_disabled_no_viz_dir(
        self, pipeline_config, mock_calibration_data, tmp_path
    ):
        """When viz is disabled, no viz/ directory is created."""
        # Default config has viz disabled
        assert pipeline_config.runtime.viz_enabled is False

        ctx = _make_ctx(pipeline_config, mock_calibration_data)
        with _mock_pipeline_stages():
            process_frame(0, _RAW_IMAGES.copy(), ctx)

        frame_dir = Path(pipeline_config.output_dir) / "frame_000000"
        assert not (frame_dir / "viz").exists()

    def test_viz_disabled_no_imports(
        self, pipeline_config, mock_calibration_data, tmp_path
    ):
        """When viz is disabled, viz modules are not imported."""
        assert pipeline_config.runtime.viz_enabled is False

        # Remove any cached viz module imports
        viz_modules_before = {
            k for k in sys.modules if k.startswith("aquamvs.visualization.")
        }

        ctx = _make_ctx(pipeline_config, mock_calibration_data)
        with _mock_pipeline_stages():
            process_frame(0, _RAW_IMAGES.copy(), ctx)

        # Check that no new viz modules were imported
        viz_modules_after = {
            k for k in sys.modules if k.startswith("aquamvs.visualization.")
        }
        new_viz_imports = viz_modules_after - viz_modules_before
        assert len(new_viz_imports) == 0, f"Unexpected viz imports: {new_viz_imports}"

    def test_viz_enabled_all_stages(self, tmp_path, mock_calibration_data):
        """When viz is enabled with stages=[], all viz functions are called."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            runtime=RuntimeConfig(viz_enabled=True, viz_stages=[], device="cpu"),
        )

        ctx = _make_ctx(config, mock_calibration_data)

        with (
            _mock_pipeline_stages(),
            patch("aquamvs.visualization.features.render_all_features") as m_feat_viz,
            patch("aquamvs.visualization.depth.render_all_depth_maps") as m_depth_viz,
            patch("aquamvs.visualization.scene.render_all_scenes") as m_scene_viz,
            patch("aquamvs.visualization.rig.render_rig_diagram") as m_rig_viz,
        ):
            process_frame(0, _RAW_IMAGES.copy(), ctx)

            assert m_feat_viz.called, "render_all_features should be called"
            assert m_depth_viz.called, "render_all_depth_maps should be called"
            assert m_scene_viz.called, "render_all_scenes should be called"
            assert m_rig_viz.called, "render_rig_diagram should be called"

        # viz directory should exist
        frame_dir = Path(config.output_dir) / "frame_000000"
        assert (frame_dir / "viz").exists()

    def test_viz_enabled_specific_stages(self, tmp_path, mock_calibration_data):
        """When viz is enabled with stages=["depth"], only depth viz runs."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            runtime=RuntimeConfig(viz_enabled=True, viz_stages=["depth"], device="cpu"),
        )

        ctx = _make_ctx(config, mock_calibration_data)

        with (
            _mock_pipeline_stages(),
            patch("aquamvs.visualization.features.render_all_features") as m_feat_viz,
            patch("aquamvs.visualization.depth.render_all_depth_maps") as m_depth_viz,
            patch("aquamvs.visualization.scene.render_all_scenes") as m_scene_viz,
            patch("aquamvs.visualization.rig.render_rig_diagram") as m_rig_viz,
        ):
            process_frame(0, _RAW_IMAGES.copy(), ctx)

            assert m_depth_viz.called, "render_all_depth_maps should be called"
            assert not m_feat_viz.called, "render_all_features should NOT be called"
            assert not m_scene_viz.called, "render_all_scenes should NOT be called"
            assert not m_rig_viz.called, "render_rig_diagram should NOT be called"

    def test_viz_error_does_not_crash_pipeline(
        self, tmp_path, mock_calibration_data, caplog
    ):
        """Viz errors are caught and logged, pipeline completes."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            runtime=RuntimeConfig(viz_enabled=True, viz_stages=["depth"], device="cpu"),
        )

        ctx = _make_ctx(config, mock_calibration_data)

        with (
            _mock_pipeline_stages(),
            patch(
                "aquamvs.visualization.depth.render_all_depth_maps",
                side_effect=RuntimeError("Viz explosion"),
            ),
            caplog.at_level(logging.INFO),
        ):
            # Should NOT raise
            process_frame(0, _RAW_IMAGES.copy(), ctx)

        # Pipeline should have completed
        assert "depth visualization failed" in caplog.text
        # Frame should still be marked complete
        assert "Frame 0: complete" in caplog.text


# ---------------------------------------------------------------------------
# OutputConfig tests
# ---------------------------------------------------------------------------


class TestOutputConfig:
    """Tests for OutputConfig gating in process_frame."""

    def test_save_features_enabled(self, tmp_path, mock_calibration_data):
        """save_features=True causes features and matches to be saved."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            runtime=RuntimeConfig(save_features=True, device="cpu"),
        )

        ctx = _make_ctx(config, mock_calibration_data)

        with (
            _mock_pipeline_stages(),
            patch("aquamvs.features.save_features") as m_sf,
            patch("aquamvs.features.save_matches") as m_sm,
        ):
            process_frame(0, _RAW_IMAGES.copy(), ctx)

            assert m_sf.called, "save_features should be called"
            assert m_sm.called, "save_matches should be called"

    def test_save_features_disabled(self, tmp_path, mock_calibration_data):
        """save_features=False (default) means no feature files saved."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            runtime=RuntimeConfig(save_features=False, device="cpu"),
        )

        ctx = _make_ctx(config, mock_calibration_data)

        with _mock_pipeline_stages():
            process_frame(0, _RAW_IMAGES.copy(), ctx)

        # No features/ directory should exist
        frame_dir = Path(config.output_dir) / "frame_000000"
        assert not (frame_dir / "features").exists()

    def test_skip_depth_maps(self, tmp_path, mock_calibration_data):
        """save_depth_maps=False means no depth_maps/ directory."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            runtime=RuntimeConfig(save_depth_maps=False, device="cpu"),
        )

        ctx = _make_ctx(config, mock_calibration_data)

        with _mock_pipeline_stages() as mocks:
            process_frame(0, _RAW_IMAGES.copy(), ctx)
            assert not mocks["save_depth"].called

        frame_dir = Path(config.output_dir) / "frame_000000"
        assert not (frame_dir / "depth_maps").exists()

    def test_skip_point_cloud(self, tmp_path, mock_calibration_data):
        """save_point_cloud=False means no point_cloud/ directory."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            runtime=RuntimeConfig(save_point_cloud=False, device="cpu"),
        )

        ctx = _make_ctx(config, mock_calibration_data)

        with _mock_pipeline_stages() as mocks:
            process_frame(0, _RAW_IMAGES.copy(), ctx)
            assert not mocks["save_pcd"].called

        frame_dir = Path(config.output_dir) / "frame_000000"
        assert not (frame_dir / "point_cloud").exists()

    def test_skip_mesh(self, tmp_path, mock_calibration_data):
        """save_mesh=False means no mesh/ directory."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            runtime=RuntimeConfig(save_mesh=False, device="cpu"),
        )

        ctx = _make_ctx(config, mock_calibration_data)

        with _mock_pipeline_stages() as mocks:
            process_frame(0, _RAW_IMAGES.copy(), ctx)
            assert not mocks["save_mesh"].called

        frame_dir = Path(config.output_dir) / "frame_000000"
        assert not (frame_dir / "mesh").exists()

    def test_cleanup_intermediates(self, tmp_path, mock_calibration_data):
        """keep_intermediates=False removes depth maps after fusion."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            runtime=RuntimeConfig(
                save_depth_maps=True, keep_intermediates=False, device="cpu"
            ),
        )

        ctx = _make_ctx(config, mock_calibration_data)

        with _mock_pipeline_stages():
            process_frame(0, _RAW_IMAGES.copy(), ctx)

        frame_dir = Path(config.output_dir) / "frame_000000"
        # Depth maps were saved but then cleaned up
        assert not (frame_dir / "depth_maps").exists()

    def test_sparse_cloud_always_saved(self, tmp_path, mock_calibration_data):
        """Sparse cloud is always saved regardless of OutputConfig flags."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            runtime=RuntimeConfig(
                save_features=False,
                save_depth_maps=False,
                save_point_cloud=False,
                save_mesh=False,
                device="cpu",
            ),
        )

        ctx = _make_ctx(config, mock_calibration_data)

        with _mock_pipeline_stages() as mocks:
            process_frame(0, _RAW_IMAGES.copy(), ctx)
            assert mocks["save_sparse"].called, (
                "save_sparse_cloud should always be called"
            )


# ---------------------------------------------------------------------------
# Summary visualization test
# ---------------------------------------------------------------------------


class TestSummaryViz:
    """Tests for summary visualization in run_pipeline."""

    def test_summary_viz_in_run_pipeline(self, tmp_path):
        """Summary viz is called when 'summary' stage is enabled."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4"},
            runtime=RuntimeConfig(
                viz_enabled=True, viz_stages=["summary"], device="cpu"
            ),
        )

        with patch("aquamvs.pipeline.runner.build_pipeline_context") as mock_setup:
            mock_ctx = Mock()
            mock_ctx.config = config
            mock_setup.return_value = mock_ctx

            with patch("aquamvs.pipeline.runner.VideoSet") as mock_videoset:
                mock_videos = MagicMock()
                mock_videos.__enter__.return_value = mock_videos
                mock_videos.iterate_frames.return_value = []  # No frames
                mock_videoset.return_value = mock_videos

                with (
                    patch("aquamvs.pipeline.runner.process_frame"),
                    patch(
                        "aquamvs.pipeline.runner._collect_height_maps",
                        return_value=[
                            (0, np.zeros((10, 10)), np.arange(10), np.arange(10)),
                        ],
                    ) as m_collect,
                    patch(
                        "aquamvs.visualization.summary.render_timeseries_gallery"
                    ) as m_gallery,
                ):
                    run_pipeline(config)

                    assert m_collect.called
                    assert m_gallery.called

    def test_summary_viz_not_called_when_disabled(self, tmp_path):
        """Summary viz is not called when viz is disabled."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4"},
            runtime=RuntimeConfig(viz_enabled=False, device="cpu"),
        )

        with patch("aquamvs.pipeline.runner.build_pipeline_context") as mock_setup:
            mock_ctx = Mock()
            mock_ctx.config = config
            mock_setup.return_value = mock_ctx

            with patch("aquamvs.pipeline.runner.VideoSet") as mock_videoset:
                mock_videos = MagicMock()
                mock_videos.__enter__.return_value = mock_videos
                mock_videos.iterate_frames.return_value = []
                mock_videoset.return_value = mock_videos

                with patch("aquamvs.pipeline.runner.process_frame"):
                    with patch(
                        "aquamvs.pipeline.runner._collect_height_maps"
                    ) as m_collect:
                        run_pipeline(config)
                        assert not m_collect.called


# ---------------------------------------------------------------------------
# BF.3 tests: Empty cloud guards and sparse cloud filtering
# ---------------------------------------------------------------------------


class TestEmptyCloudHandling:
    """Tests for empty point cloud guards (B.3)."""

    def test_process_frame_empty_fusion_no_crash(
        self, tmp_path, mock_calibration_data, caplog
    ):
        """Empty fused point cloud skips surface reconstruction without crash."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            runtime=RuntimeConfig(device="cpu"),
        )

        ctx = _make_ctx(config, mock_calibration_data)

        with _mock_pipeline_stages() as mocks:
            # Override fusion to return an empty point cloud
            empty_pcd = o3d.geometry.PointCloud()
            mocks["fuse"].return_value = empty_pcd

            with caplog.at_level(logging.WARNING):
                # Should NOT raise
                process_frame(0, _RAW_IMAGES.copy(), ctx)

            # Verify warnings logged
            assert "fused point cloud is empty" in caplog.text
            # Surface reconstruction should not have been called
            assert not mocks["surface"].called
            # Mesh and point cloud should not have been saved
            assert not mocks["save_mesh"].called
            assert not mocks["save_pcd"].called

        # Verify no mesh/ or point_cloud/ directories created
        frame_dir = Path(config.output_dir) / "frame_000000"
        assert not (frame_dir / "mesh").exists()
        assert not (frame_dir / "point_cloud").exists()

    def test_process_frame_nonempty_fusion_proceeds_normally(
        self, tmp_path, mock_calibration_data
    ):
        """Non-empty fused point cloud proceeds to surface reconstruction."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            runtime=RuntimeConfig(device="cpu"),
        )

        ctx = _make_ctx(config, mock_calibration_data)

        with _mock_pipeline_stages() as mocks:
            # Override fusion to return a non-empty point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector([[0, 0, 1], [0.1, 0, 1]])
            mocks["fuse"].return_value = pcd

            process_frame(0, _RAW_IMAGES.copy(), ctx)

            # Surface reconstruction should have been called
            assert mocks["surface"].called
            # Point cloud and mesh should have been saved
            assert mocks["save_pcd"].called
            assert mocks["save_mesh"].called


class TestSparseCloudFiltering:
    """Tests for sparse cloud filtering wiring (B.6)."""

    def test_process_frame_calls_filter_sparse_cloud(
        self, tmp_path, mock_calibration_data
    ):
        """filter_sparse_cloud is called with water_z from calibration."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            runtime=RuntimeConfig(device="cpu"),
        )

        ctx = _make_ctx(config, mock_calibration_data)

        with _mock_pipeline_stages():
            with patch(
                "aquamvs.pipeline.stages.sparse_matching.filter_sparse_cloud"
            ) as mock_filter:
                # Mock filter to return the input unchanged
                mock_filter.side_effect = lambda cloud, **kwargs: cloud

                process_frame(0, _RAW_IMAGES.copy(), ctx)

                # Verify filter was called
                assert mock_filter.called
                call_args = mock_filter.call_args
                # Check water_z is passed from calibration
                assert call_args[1]["water_z"] == mock_calibration_data.water_z

    def test_process_frame_filter_reduces_points(
        self, tmp_path, mock_calibration_data, caplog
    ):
        """Filtered sparse cloud with fewer points is passed to depth range estimation."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            runtime=RuntimeConfig(device="cpu"),
        )

        ctx = _make_ctx(config, mock_calibration_data)

        with _mock_pipeline_stages() as mocks:
            # Override triangulate to return a cloud with 100 points
            mocks["triangulate"].return_value = {
                "points_3d": torch.zeros(100, 3),
                "scores": torch.zeros(100),
            }

            with patch(
                "aquamvs.pipeline.stages.sparse_matching.filter_sparse_cloud"
            ) as mock_filter:
                # Mock filter to reduce points to 50
                def filter_side_effect(cloud, **kwargs):
                    return {
                        "points_3d": cloud["points_3d"][:50],
                        "scores": cloud["scores"][:50],
                    }

                mock_filter.side_effect = filter_side_effect

                with caplog.at_level(logging.INFO):
                    process_frame(0, _RAW_IMAGES.copy(), ctx)

                # Verify filtering logged
                assert "100 -> 50 points (50 removed)" in caplog.text

                # Verify compute_depth_ranges received the filtered cloud
                depth_range_call = mocks["depth_ranges"].call_args
                filtered_cloud = depth_range_call[0][1]
                assert filtered_cloud["points_3d"].shape[0] == 50

    def test_process_frame_all_points_filtered(
        self, tmp_path, mock_calibration_data, caplog
    ):
        """Pipeline completes when all sparse points are filtered out."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            runtime=RuntimeConfig(device="cpu"),
        )

        ctx = _make_ctx(config, mock_calibration_data)

        with _mock_pipeline_stages() as mocks:
            # Override triangulate to return a cloud with some points
            mocks["triangulate"].return_value = {
                "points_3d": torch.zeros(100, 3),
                "scores": torch.zeros(100),
            }

            with patch(
                "aquamvs.pipeline.stages.sparse_matching.filter_sparse_cloud"
            ) as mock_filter:
                # Mock filter to remove all points
                def filter_side_effect(cloud, **kwargs):
                    return {
                        "points_3d": torch.zeros(0, 3),
                        "scores": torch.zeros(0),
                    }

                mock_filter.side_effect = filter_side_effect

                with caplog.at_level(logging.INFO):
                    # Should NOT crash
                    process_frame(0, _RAW_IMAGES.copy(), ctx)

                # Verify filtering logged
                assert "100 -> 0 points (100 removed)" in caplog.text

                # Verify compute_depth_ranges was called (with empty cloud)
                assert mocks["depth_ranges"].called
                depth_range_call = mocks["depth_ranges"].call_args
                filtered_cloud = depth_range_call[0][1]
                assert filtered_cloud["points_3d"].shape[0] == 0

                # Pipeline should complete
                assert "Frame 0: complete" in caplog.text


# ---------------------------------------------------------------------------
# P.34 tests: Sparse mode branch
# ---------------------------------------------------------------------------


class TestSparseMode:
    """Tests for sparse pipeline mode (P.34)."""

    def test_sparse_mode_skips_dense_stereo(self, tmp_path, mock_calibration_data):
        """Sparse mode skips plane sweep, extract_depth, filter, and fusion."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            pipeline_mode="sparse",
            runtime=RuntimeConfig(device="cpu"),
        )

        ctx = _make_ctx(config, mock_calibration_data)

        with _mock_pipeline_stages() as mocks:
            process_frame(0, _RAW_IMAGES.copy(), ctx)

            # Sparse mode should call triangulate and compute_depth_ranges
            assert mocks["triangulate"].called
            assert mocks["depth_ranges"].called

            # But should NOT call dense stereo stages
            assert not mocks["sweep"].called
            assert not mocks["extract_depth"].called
            assert not mocks["filter"].called
            assert not mocks["fuse"].called

    def test_sparse_mode_calls_surface_reconstruction(
        self, tmp_path, mock_calibration_data
    ):
        """Sparse mode calls surface reconstruction with sparse cloud."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            pipeline_mode="sparse",
            runtime=RuntimeConfig(device="cpu"),
        )

        ctx = _make_ctx(config, mock_calibration_data)

        with _mock_pipeline_stages() as mocks:
            # Override triangulate to return non-empty cloud
            mocks["triangulate"].return_value = {
                "points_3d": torch.zeros(10, 3),
                "scores": torch.ones(10),
            }

            process_frame(0, _RAW_IMAGES.copy(), ctx)

            # Sparse-to-O3D conversion should be called
            assert mocks["sparse_to_o3d"].called

            # Surface reconstruction should be called
            assert mocks["surface"].called

            # Point cloud and mesh should be saved (sparse mode uses save_pcd_sparse)
            assert mocks["save_pcd_sparse"].called
            assert mocks["save_mesh"].called

    def test_sparse_mode_saves_sparse_ply(self, tmp_path, mock_calibration_data):
        """Sparse mode saves point cloud as sparse.ply, not fused.ply."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            pipeline_mode="sparse",
            runtime=RuntimeConfig(device="cpu"),
        )

        ctx = _make_ctx(config, mock_calibration_data)

        with _mock_pipeline_stages() as mocks:
            # Override triangulate to return non-empty cloud
            mocks["triangulate"].return_value = {
                "points_3d": torch.zeros(10, 3),
                "scores": torch.ones(10),
            }

            process_frame(0, _RAW_IMAGES.copy(), ctx)

            # Verify save_point_cloud was called with path ending in sparse.ply
            assert mocks["save_pcd_sparse"].called
            call_args = mocks["save_pcd_sparse"].call_args
            save_path = call_args[0][1]  # Second positional arg is the path
            assert str(save_path).endswith("sparse.ply")
            assert "fused.ply" not in str(save_path)

    def test_sparse_mode_empty_cloud_no_crash(
        self, tmp_path, mock_calibration_data, caplog
    ):
        """Sparse mode with empty sparse cloud logs warning and skips surface reconstruction."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            pipeline_mode="sparse",
            runtime=RuntimeConfig(device="cpu"),
        )

        ctx = _make_ctx(config, mock_calibration_data)

        with _mock_pipeline_stages() as mocks:
            # Override triangulate to return empty cloud
            mocks["triangulate"].return_value = {
                "points_3d": torch.zeros(0, 3),
                "scores": torch.zeros(0),
            }

            with caplog.at_level(logging.WARNING):
                # Should NOT raise
                process_frame(0, _RAW_IMAGES.copy(), ctx)

            # Should log warning
            assert "sparse cloud is empty" in caplog.text

            # sparse_to_o3d should NOT be called (no points)
            assert not mocks["sparse_to_o3d"].called

            # Surface reconstruction should NOT be called
            assert not mocks["surface"].called

            # Save functions should NOT be called
            assert not mocks["save_pcd"].called
            assert not mocks["save_mesh"].called

    def test_sparse_mode_scene_viz(self, tmp_path, mock_calibration_data):
        """Sparse mode with scene viz enabled renders the sparse cloud."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            pipeline_mode="sparse",
            runtime=RuntimeConfig(viz_enabled=True, viz_stages=["scene"], device="cpu"),
        )

        ctx = _make_ctx(config, mock_calibration_data)

        with _mock_pipeline_stages() as mocks:
            # Override triangulate to return non-empty cloud
            mocks["triangulate"].return_value = {
                "points_3d": torch.zeros(10, 3),
                "scores": torch.ones(10),
            }

            with patch("aquamvs.visualization.scene.render_all_scenes") as m_scene_viz:
                process_frame(0, _RAW_IMAGES.copy(), ctx)

                # Scene viz should be called
                assert m_scene_viz.called

                # Depth viz should NOT be called (no depth maps in sparse mode)
                # (tested implicitly by _mock_pipeline_stages not having depth viz patched)

    def test_sparse_mode_rig_viz(self, tmp_path, mock_calibration_data):
        """Sparse mode with rig viz enabled renders the sparse cloud."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            pipeline_mode="sparse",
            runtime=RuntimeConfig(viz_enabled=True, viz_stages=["rig"], device="cpu"),
        )

        ctx = _make_ctx(config, mock_calibration_data)

        with _mock_pipeline_stages() as mocks:
            # Override triangulate to return non-empty cloud
            mocks["triangulate"].return_value = {
                "points_3d": torch.zeros(10, 3),
                "scores": torch.ones(10),
            }

            with patch("aquamvs.visualization.rig.render_rig_diagram") as m_rig_viz:
                process_frame(0, _RAW_IMAGES.copy(), ctx)

                # Rig viz should be called
                assert m_rig_viz.called

    def test_full_mode_unchanged(self, tmp_path, mock_calibration_data):
        """Full mode (default) runs the complete pipeline as before."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            pipeline_mode="full",
            runtime=RuntimeConfig(device="cpu"),
        )

        ctx = _make_ctx(config, mock_calibration_data)

        with _mock_pipeline_stages() as mocks:
            process_frame(0, _RAW_IMAGES.copy(), ctx)

            # Full mode should call all stages
            assert mocks["triangulate"].called
            assert mocks["depth_ranges"].called
            assert mocks["sweep"].called
            assert mocks["extract_depth"].called
            assert mocks["filter"].called
            assert mocks["fuse"].called
            assert mocks["surface"].called

            # sparse_to_o3d should NOT be called in full mode
            assert not mocks["sparse_to_o3d"].called

    def test_sparse_mode_logs_completion(self, tmp_path, mock_calibration_data, caplog):
        """Sparse mode logs 'complete (sparse mode)' at end."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            pipeline_mode="sparse",
            runtime=RuntimeConfig(device="cpu"),
        )

        ctx = _make_ctx(config, mock_calibration_data)

        with _mock_pipeline_stages():
            with caplog.at_level(logging.INFO):
                process_frame(0, _RAW_IMAGES.copy(), ctx)

            # Should log sparse mode completion
            assert "complete (sparse mode)" in caplog.text


# ---------------------------------------------------------------------------
# P.35 tests: ROI mask integration
# ---------------------------------------------------------------------------


class TestMaskIntegration:
    """Tests for ROI mask integration in pipeline (P.35)."""

    def test_masks_loaded_in_setup(self, tmp_path, mock_calibration_data):
        """setup_pipeline calls load_all_masks and stores result in ctx.masks."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            mask_dir=str(tmp_path / "masks"),
            runtime=RuntimeConfig(device="cpu"),
        )

        with (
            patch(
                "aquamvs.pipeline.builder.load_calibration_data",
                return_value=mock_calibration_data,
            ),
            patch("aquamvs.pipeline.builder.select_pairs") as mock_select_pairs,
        ):
            mock_select_pairs.return_value = {"cam0": ["cam1"]}

            with patch("aquamvs.pipeline.builder.load_all_masks") as mock_load_masks:
                mock_load_masks.return_value = {
                    "cam0": np.ones((480, 640), dtype=np.uint8)
                }

                ctx = setup_pipeline(config)

                # Verify load_all_masks was called with correct args
                assert mock_load_masks.called
                call_args = mock_load_masks.call_args
                assert call_args[0][0] == config.mask_dir
                assert call_args[0][1] == mock_calibration_data.cameras

                # Verify result is stored in ctx
                assert "cam0" in ctx.masks
                assert np.array_equal(
                    ctx.masks["cam0"],
                    np.ones((480, 640), dtype=np.uint8),
                )

    def test_masks_filter_features(self, tmp_path, mock_calibration_data, caplog):
        """Masks filter features before matching in process_frame."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            runtime=RuntimeConfig(device="cpu"),
        )

        # Create a mask for cam0 (right half valid)
        mask_cam0 = np.zeros((480, 640), dtype=np.uint8)
        mask_cam0[:, 320:] = 255  # Right half valid

        ctx = _make_ctx(config, mock_calibration_data, masks={"cam0": mask_cam0})

        with _mock_pipeline_stages() as mocks:
            # Override extract to return features spanning both halves
            mocks["extract"].return_value = {
                "cam0": {
                    "keypoints": torch.tensor(
                        [
                            [100.0, 240.0],  # Left (should be filtered)
                            [400.0, 240.0],  # Right (should survive)
                            [500.0, 300.0],  # Right (should survive)
                        ]
                    ),
                    "descriptors": torch.randn(3, 256),
                    "scores": torch.ones(3),
                },
            }

            with patch("aquamvs.masks.apply_mask_to_features") as mock_apply:
                # Pass through but record that it was called
                mock_apply.side_effect = lambda feats, mask: feats

                with caplog.at_level(logging.DEBUG):
                    process_frame(0, _RAW_IMAGES.copy(), ctx)

                # Verify apply_mask_to_features was called
                assert mock_apply.called
                call_args = mock_apply.call_args
                # Verify mask was passed
                assert np.array_equal(call_args[0][1], mask_cam0)

    def test_masks_filter_depth(self, tmp_path, mock_calibration_data):
        """Masks filter depth maps after extraction in process_frame."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            runtime=RuntimeConfig(device="cpu"),
        )

        # Create a mask for cam0 (top half excluded)
        mask_cam0 = np.ones((480, 640), dtype=np.uint8) * 255
        mask_cam0[:240, :] = 0  # Top half excluded

        ctx = _make_ctx(config, mock_calibration_data, masks={"cam0": mask_cam0})

        with _mock_pipeline_stages():
            with patch("aquamvs.masks.apply_mask_to_depth") as mock_apply:
                # Pass through but record that it was called
                mock_apply.side_effect = lambda depth, conf, mask: (depth, conf)

                process_frame(0, _RAW_IMAGES.copy(), ctx)

                # Verify apply_mask_to_depth was called
                assert mock_apply.called
                call_args = mock_apply.call_args
                # Verify mask was passed
                assert np.array_equal(call_args[0][2], mask_cam0)

    def test_no_masks_unchanged(self, tmp_path, mock_calibration_data):
        """Pipeline runs identically when masks={} (no masking configured)."""
        config = PipelineConfig(
            calibration_path="dummy.json",
            output_dir=str(tmp_path / "output"),
            camera_video_map={"cam0": "v0.mp4", "cam1": "v1.mp4", "cam2": "v2.mp4"},
            runtime=RuntimeConfig(device="cpu"),
        )

        # Empty masks dict (no masking)
        ctx = _make_ctx(config, mock_calibration_data, masks={})

        with _mock_pipeline_stages() as mocks:
            with patch("aquamvs.masks.apply_mask_to_features") as mock_apply_feat:
                with patch("aquamvs.masks.apply_mask_to_depth") as mock_apply_depth:
                    process_frame(0, _RAW_IMAGES.copy(), ctx)

                    # Mask application should NOT be called when masks is empty
                    assert not mock_apply_feat.called
                    assert not mock_apply_depth.called

                    # Pipeline should still run normally
                    assert mocks["extract"].called
                    assert mocks["match"].called
