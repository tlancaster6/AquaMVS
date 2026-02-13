"""ICP alignment for ground truth comparison."""

from typing import Any

import numpy as np
import open3d as o3d

from aquamvs.config import EvaluationConfig


def icp_align(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    config: EvaluationConfig,
    init_transform: np.ndarray | None = None,
) -> dict[str, Any]:
    """Align a source point cloud to a target using ICP.

    Uses Open3D's point-to-plane ICP for robust alignment. Useful for
    comparing reconstructions across runs or aligning to ground truth.

    Args:
        source: Source point cloud to be transformed.
        target: Target (reference) point cloud.
        config: Evaluation configuration.
        init_transform: Optional initial 4x4 transformation matrix.
            Defaults to identity.

    Returns:
        Dict with keys:
            "transformation": (4, 4) ndarray, float64 -- best-fit rigid transform
            "aligned": PointCloud -- source transformed by the result
            "fitness": float -- fraction of source points with a correspondence
                within icp_max_distance
            "inlier_rmse": float -- RMSE of inlier correspondences (meters)
    """
    if init_transform is None:
        init_transform = np.eye(4)

    # Ensure both clouds have normals (needed for point-to-plane)
    if not target.has_normals():
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=config.icp_max_distance * 5,
                max_nn=30,
            )
        )
    if not source.has_normals():
        source.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=config.icp_max_distance * 5,
                max_nn=30,
            )
        )

    # Run point-to-plane ICP
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance=config.icp_max_distance,
        init=init_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )

    # Apply transformation to a copy of source (transform is in-place)
    import copy
    aligned = copy.deepcopy(source)
    aligned.transform(result.transformation)

    return {
        "transformation": np.array(result.transformation),
        "aligned": aligned,
        "fitness": result.fitness,
        "inlier_rmse": result.inlier_rmse,
    }
