"""Pipeline context builder for one-time initialization."""

import logging
from pathlib import Path

from ..calibration import (
    compute_undistortion_maps,
    load_calibration_data,
)
from ..config import PipelineConfig
from ..features import select_pairs
from ..masks import load_all_masks
from ..projection.refractive import RefractiveProjectionModel
from .context import PipelineContext

logger = logging.getLogger(__name__)


def build_pipeline_context(config: PipelineConfig) -> PipelineContext:
    """Perform one-time pipeline initialization.

    Loads calibration, creates projection models, computes undistortion
    maps, and selects camera pairs. All returned data is constant for
    the entire video session.

    Args:
        config: Full pipeline configuration.

    Returns:
        PipelineContext with all precomputed data.
    """
    device = config.runtime.device

    # 1. Load calibration
    logger.info("Loading calibration from %s", config.calibration_path)
    calibration = load_calibration_data(config.calibration_path)

    # Filter calibration cameras to those with video files
    video_cameras = set(config.camera_video_map.keys())

    all_ring = calibration.ring_cameras
    all_auxiliary = calibration.auxiliary_cameras

    ring_cameras = [c for c in all_ring if c in video_cameras]
    auxiliary_cameras = [c for c in all_auxiliary if c in video_cameras]

    skipped_ring = [c for c in all_ring if c not in video_cameras]
    skipped_aux = [c for c in all_auxiliary if c not in video_cameras]
    if skipped_ring:
        logger.warning(
            "Ring cameras in calibration but missing video (skipped): %s",
            skipped_ring,
        )
    if skipped_aux:
        logger.warning(
            "Auxiliary cameras in calibration but missing video (skipped): %s",
            skipped_aux,
        )

    active_cameras = set(ring_cameras) | set(auxiliary_cameras)
    logger.info(
        "Found %d ring cameras, %d auxiliary cameras (of %d/%d in calibration)",
        len(ring_cameras),
        len(auxiliary_cameras),
        len(all_ring),
        len(all_auxiliary),
    )

    # 2. Compute undistortion maps (only for cameras with video)
    logger.info("Computing undistortion maps")
    undistortion_maps = {}
    for name in active_cameras:
        cam = calibration.cameras[name]
        undistortion_maps[name] = compute_undistortion_maps(cam)

    # 3. Create projection models (only for cameras with video)
    logger.info("Creating projection models")
    projection_models = {}
    for name in active_cameras:
        cam = calibration.cameras[name]
        K_new = undistortion_maps[name].K_new
        projection_models[name] = RefractiveProjectionModel(
            K=K_new,
            R=cam.R,
            t=cam.t,
            water_z=calibration.water_z,
            normal=calibration.interface_normal,
            n_air=calibration.n_air,
            n_water=calibration.n_water,
        ).to(device)

    # 4. Select pairs (using only active cameras)
    logger.info("Selecting camera pairs")
    camera_positions = {
        name: pos
        for name, pos in calibration.camera_positions().items()
        if name in active_cameras
    }
    pairs = select_pairs(
        camera_positions,
        ring_cameras,
        auxiliary_cameras,
        config.sparse_matching,
    )

    # 4b. Load ROI masks
    masks = load_all_masks(config.mask_dir, calibration.cameras)

    # 5. Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 6. Save config copy
    config.to_yaml(output_dir / "config.yaml")
    logger.info("Config saved to %s", output_dir / "config.yaml")

    return PipelineContext(
        config=config,
        calibration=calibration,
        undistortion_maps=undistortion_maps,
        projection_models=projection_models,
        pairs=pairs,
        ring_cameras=ring_cameras,
        auxiliary_cameras=auxiliary_cameras,
        device=device,
        masks=masks,
    )


# Backward compatibility alias
setup_pipeline = build_pipeline_context
