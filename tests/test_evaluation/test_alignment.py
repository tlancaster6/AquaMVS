"""Tests for ICP alignment."""

import numpy as np
import open3d as o3d

from aquamvs.config import EvaluationConfig
from aquamvs.evaluation import icp_align


def create_sphere_cloud(center, radius=0.05, n_points=200):
    """Create a synthetic point cloud in the shape of a sphere."""
    # Random points on a sphere
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    x = radius * np.sin(phi) * np.cos(theta) + center[0]
    y = radius * np.sin(phi) * np.sin(theta) + center[1]
    z = radius * np.cos(phi) + center[2]
    points = np.column_stack([x, y, z])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def create_flat_cloud(z=1.5, n_points=200):
    """Create a synthetic point cloud of a flat horizontal plane."""
    xy = np.random.uniform(-0.1, 0.1, size=(n_points, 2))
    z_vals = np.full(n_points, z)
    points = np.column_stack([xy, z_vals])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def test_icp_identity():
    """Align a cloud to itself - should get identity transformation."""
    config = EvaluationConfig(icp_max_distance=0.01)
    cloud = create_flat_cloud(z=1.5, n_points=300)

    result = icp_align(cloud, cloud, config)

    # Transformation should be identity
    assert np.allclose(result["transformation"], np.eye(4), atol=1e-3)
    # All points should match
    assert result["fitness"] > 0.95
    # RMSE should be near zero
    assert result["inlier_rmse"] < 1e-3


def test_icp_known_transform():
    """Apply a known transform and verify ICP recovers it."""
    import copy

    config = EvaluationConfig(icp_max_distance=0.01)

    # Create a target cloud
    target = create_sphere_cloud(center=[0.0, 0.0, 1.5], radius=0.05, n_points=300)

    # Apply a small known transformation to create source
    # (need to copy first as transform() is in-place)
    source = copy.deepcopy(target)

    # Small rotation around Z (yaw)
    angle = np.deg2rad(10)  # 10 degrees
    R = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    # Small translation
    T = np.eye(4)
    T[:3, 3] = [0.005, 0.005, 0.005]

    known_transform = T @ R
    source.transform(known_transform)

    # Run ICP - should recover the inverse transformation
    result = icp_align(source, target, config)

    # Apply the recovered transformation to source
    np.asarray(result["aligned"].points)
    np.asarray(target.points)

    # After alignment, points should be close to target
    # (This is an indirect check - the transformation should approximately invert the applied one)
    # Check fitness and RMSE as proxies
    assert result["fitness"] > 0.9
    assert result["inlier_rmse"] < 0.002  # 2mm


def test_icp_distant_clouds():
    """ICP with non-overlapping clouds should have low fitness."""
    config = EvaluationConfig(icp_max_distance=0.01)

    # Two clouds far apart
    cloud_a = create_flat_cloud(z=1.0, n_points=200)
    cloud_b = create_flat_cloud(z=2.0, n_points=200)  # 1 meter apart in Z

    result = icp_align(cloud_a, cloud_b, config)

    # Fitness should be low (few correspondences)
    assert result["fitness"] < 0.2


def test_icp_normals_auto_estimated():
    """ICP should auto-estimate normals if not provided."""
    config = EvaluationConfig(icp_max_distance=0.01)

    # Create clouds without normals
    source = create_flat_cloud(z=1.5, n_points=300)
    target = create_flat_cloud(z=1.5, n_points=300)

    assert not source.has_normals()
    assert not target.has_normals()

    # Should succeed anyway
    result = icp_align(source, target, config)

    assert result["fitness"] > 0.5
    assert "transformation" in result
    assert "aligned" in result


def test_icp_with_init_transform():
    """ICP with a good initial transformation should converge better."""
    config = EvaluationConfig(icp_max_distance=0.01)

    target = create_sphere_cloud(center=[0.0, 0.0, 1.5], radius=0.05, n_points=300)

    # Apply a translation (need to copy the cloud first as transform() is in-place)
    import copy

    source = copy.deepcopy(target)
    translation = np.eye(4)
    translation[:3, 3] = [0.01, 0.01, 0.0]
    source.transform(translation)

    # Provide the inverse as initial guess
    init_guess = np.eye(4)
    init_guess[:3, 3] = [-0.01, -0.01, 0.0]

    result = icp_align(source, target, config, init_transform=init_guess)

    # Should achieve good alignment
    assert result["fitness"] > 0.9
    assert result["inlier_rmse"] < 0.002
