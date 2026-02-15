"""Fusion stage (geometric consistency filtering + depth map fusion + outlier removal)."""

import logging
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from torch.profiler import record_function

from ...fusion import filter_all_depth_maps, fuse_depth_maps, save_point_cloud
from ..context import PipelineContext
from ..helpers import _save_consistency_map

logger = logging.getLogger(__name__)


def run_fusion_stage(
    depth_maps: dict[str, torch.Tensor],
    confidence_maps: dict[str, torch.Tensor],
    undistorted: dict[str, np.ndarray],
    ctx: PipelineContext,
    frame_dir: Path,
    frame_idx: int,
    skip_filter: bool = False,
) -> o3d.geometry.PointCloud:
    """Run geometric consistency filtering, depth map fusion, and outlier removal.

    Args:
        depth_maps: Dict mapping camera name to depth map tensor.
        confidence_maps: Dict mapping camera name to confidence map tensor.
        undistorted: Dict of undistorted BGR images (H, W, 3) uint8 numpy arrays.
        ctx: Pipeline context.
        frame_dir: Frame output directory.
        frame_idx: Frame index (for logging).
        skip_filter: If True, skip geometric consistency filtering (used by roma+full path).

    Returns:
        Fused point cloud (after outlier removal if enabled).
    """
    with record_function("fusion"):
        config = ctx.config

        # --- Stage 7: Geometric Consistency Filtering ---
        # Skip for RoMa+full: aggregate_pairwise_depths already enforces multi-view
        # consistency at warp level. Applying cross-camera filtering on top causes
        # cascading sparsification (sparse maps can't cross-validate each other's
        # edges), producing the star/wedge pattern in the fused cloud. (B.16)
        if skip_filter:
            logger.info(
                "Frame %d: skipping geometric consistency filter (RoMa path)", frame_idx
            )
            filtered_depths = depth_maps
            filtered_confs = confidence_maps
        else:
            logger.info("Frame %d: filtering depth maps", frame_idx)
            filtered = filter_all_depth_maps(
                ctx.ring_cameras,
                ctx.projection_models,
                depth_maps,
                confidence_maps,
                config.reconstruction,
            )

            filtered_depths = {name: f[0] for name, f in filtered.items()}
            filtered_confs = {name: f[1] for name, f in filtered.items()}

            # Save consistency maps (opt-in)
            if config.runtime.save_consistency_maps:
                consistency_dir = frame_dir / "consistency_maps"
                consistency_dir.mkdir(parents=True, exist_ok=True)
                for cam_name, (_, _, consistency) in filtered.items():
                    _save_consistency_map(
                        consistency=consistency,
                        output_stem=consistency_dir / cam_name,
                        max_value=len(ctx.pairs[cam_name]),
                    )
                logger.info(
                    "Frame %d: saved consistency maps for %d cameras",
                    frame_idx,
                    len(filtered),
                )

        # --- Stage 8: Depth Map Fusion ---
        logger.info("Frame %d: fusing depth maps", frame_idx)
        # Convert undistorted images to tensors for color sampling
        undistorted_for_fusion = {
            name: torch.from_numpy(img) for name, img in undistorted.items()
        }
        fused_pcd = fuse_depth_maps(
            ctx.ring_cameras,
            ctx.projection_models,
            filtered_depths,
            filtered_confs,
            undistorted_for_fusion,
            config.reconstruction,
        )

        # --- Clean up intermediates after successful fusion ---
        if not config.runtime.keep_intermediates:
            depth_dir = frame_dir / "depth_maps"
            if depth_dir.exists():
                import shutil

                shutil.rmtree(depth_dir)
                logger.debug("Frame %d: removed intermediate depth maps", frame_idx)

        # --- Save fused point cloud (opt-out) ---
        if config.runtime.save_point_cloud:
            if fused_pcd.has_points():
                pcd_dir = frame_dir / "point_cloud"
                pcd_dir.mkdir(exist_ok=True)
                save_point_cloud(fused_pcd, pcd_dir / "fused.ply")
            else:
                logger.warning(
                    "Frame %d: fused point cloud is empty, skipping point cloud save",
                    frame_idx,
                )

        # --- Statistical Outlier Removal (after fusion, before surface reconstruction) ---
        # Skip if too few points for meaningful neighbor statistics
        if (
            config.reconstruction.outlier_removal_enabled
            and fused_pcd.has_points()
            and len(fused_pcd.points) > config.reconstruction.outlier_nb_neighbors
        ):
            original_count = len(fused_pcd.points)
            fused_pcd, _ = fused_pcd.remove_statistical_outlier(
                nb_neighbors=config.reconstruction.outlier_nb_neighbors,
                std_ratio=config.reconstruction.outlier_std_ratio,
            )
            removed = original_count - len(fused_pcd.points)
            logger.info(
                "Frame %d: removed %d outliers (%.1f%%) from fused cloud",
                frame_idx,
                removed,
                removed / original_count * 100 if original_count > 0 else 0.0,
            )

        return fused_pcd
