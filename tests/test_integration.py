"""End-to-end integration tests for the AquaMVS pipeline.

Runs process_frame() with synthetic but geometrically realistic inputs
(no mocking) to verify the full reconstruction chain produces valid
outputs: sparse cloud, depth maps, fused point cloud, and surface mesh.
"""

import math

import numpy as np
import open3d as o3d
import pytest
import torch

from aquamvs.calibration import CalibrationData, CameraData, UndistortionData
from aquamvs.config import (
    DenseStereoConfig,
    FusionConfig,
    PairSelectionConfig,
    PipelineConfig,
    SurfaceConfig,
)
from aquamvs.features import select_pairs
from aquamvs.pipeline import PipelineContext, process_frame
from aquamvs.projection.refractive import RefractiveProjectionModel

# --- Scene constants ---

IMAGE_W, IMAGE_H = 320, 240
WATER_Z = 0.5
SAND_Z = 1.0
RADIUS = 0.3
N_CAMS = 3


# --- Helper functions ---


def make_K(w: int, h: int, fov_deg: float = 60.0) -> torch.Tensor:
    """Create a pinhole intrinsic matrix.

    Args:
        w: Image width in pixels.
        h: Image height in pixels.
        fov_deg: Horizontal field of view in degrees.

    Returns:
        Intrinsic matrix, shape (3, 3), float32.
    """
    f = w / (2 * math.tan(math.radians(fov_deg / 2)))
    return torch.tensor(
        [
            [f, 0, w / 2],
            [0, f, h / 2],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )


def make_identity_undistortion(
    K: torch.Tensor,
    height: int,
    width: int,
) -> UndistortionData:
    """Create identity undistortion maps (no distortion to correct).

    Args:
        K: Intrinsic matrix, shape (3, 3), float32.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        UndistortionData with identity remap tables.
    """
    v, u = np.mgrid[:height, :width].astype(np.float32)
    return UndistortionData(K_new=K.clone(), map_x=u, map_y=v)


def checkerboard(
    x: torch.Tensor,
    y: torch.Tensor,
    cell_size: float = 0.02,
) -> torch.Tensor:
    """Checkerboard texture returning float in [0, 1].

    Args:
        x: X world coordinates, shape (N,).
        y: Y world coordinates, shape (N,).
        cell_size: Checkerboard cell size in meters.

    Returns:
        Intensity values in [0.1, 0.9], shape (N,).
    """
    cx = (x / cell_size).floor().long()
    cy = (y / cell_size).floor().long()
    return ((cx + cy) % 2).float() * 0.8 + 0.1


def render_synthetic_image(
    model: RefractiveProjectionModel,
    height: int,
    width: int,
    sand_z: float,
    texture_fn,
) -> np.ndarray:
    """Render a synthetic image of a textured plane.

    Casts rays through the projection model, intersects with the
    Z = sand_z plane, and samples a texture function at hit XY coords.

    Args:
        model: Refractive projection model for this camera.
        height: Image height in pixels.
        width: Image width in pixels.
        sand_z: Z-coordinate of the sand plane.
        texture_fn: Callable mapping (x, y) tensors to intensity.

    Returns:
        BGR image, shape (H, W, 3), uint8.
    """
    # Create pixel grid: (H*W, 2)
    v, u = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    pixels = torch.stack([u.reshape(-1), v.reshape(-1)], dim=-1)

    # Cast rays through the refractive model
    origins, directions = model.cast_ray(pixels)

    # Ray-plane intersection at Z = sand_z
    # origin_z + d * direction_z = sand_z
    d = (sand_z - origins[:, 2]) / directions[:, 2]
    hit_points = origins + d.unsqueeze(-1) * directions

    # Sample texture at hit XY
    intensity = texture_fn(hit_points[:, 0], hit_points[:, 1])
    intensity = (intensity * 255).clamp(0, 255).byte()

    # Create BGR image (grayscale -> 3 identical channels)
    gray = intensity.reshape(height, width)
    image = torch.stack([gray, gray, gray], dim=-1)
    return image.numpy()


def build_synthetic_scene(tmp_path):
    """Build a complete synthetic scene for integration testing.

    Creates 3 cameras in an equilateral ring at radius 0.3m, all at Z=0
    looking straight down (+Z) with identity rotation. Water surface at
    Z=0.5, textured sand plane at Z=1.0. Renders checkerboard images
    and assembles a PipelineContext ready for process_frame().

    Args:
        tmp_path: Pytest temporary directory for output.

    Returns:
        Dict with keys: ctx, images, calibration, projection_models, tmp_path.
    """
    K = make_K(IMAGE_W, IMAGE_H)

    # Build 3 cameras in an equilateral ring
    cameras = {}
    for i in range(N_CAMS):
        angle = 2 * math.pi * i / N_CAMS
        cx = RADIUS * math.cos(angle)
        cy = RADIUS * math.sin(angle)

        cam_name = f"cam{i}"
        R = torch.eye(3, dtype=torch.float32)
        # Extrinsics: p_cam = R @ p_world + t
        # Camera center: C = -R^T @ t = -t (when R=I)
        # To place camera at (cx, cy, 0): t = (-cx, -cy, 0)
        t = torch.tensor([-cx, -cy, 0.0], dtype=torch.float32)

        cameras[cam_name] = CameraData(
            name=cam_name,
            K=K.clone(),
            dist_coeffs=torch.zeros(5, dtype=torch.float64),
            R=R,
            t=t,
            image_size=(IMAGE_W, IMAGE_H),
            is_fisheye=False,
            is_auxiliary=False,
        )

    calibration = CalibrationData(
        cameras=cameras,
        water_z=WATER_Z,
        interface_normal=torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),
        n_air=1.0,
        n_water=1.333,
    )

    # Build projection models and identity undistortion maps
    projection_models = {}
    undistortion_maps = {}
    for name, cam in calibration.cameras.items():
        projection_models[name] = RefractiveProjectionModel(
            K=cam.K,
            R=cam.R,
            t=cam.t,
            water_z=calibration.water_z,
            normal=calibration.interface_normal,
            n_air=calibration.n_air,
            n_water=calibration.n_water,
        )
        undistortion_maps[name] = make_identity_undistortion(cam.K, IMAGE_H, IMAGE_W)

    # Select camera pairs (each camera paired with both others)
    pairs = select_pairs(
        calibration.camera_positions(),
        calibration.ring_cameras,
        calibration.auxiliary_cameras,
        PairSelectionConfig(num_neighbors=2, include_center=False),
    )

    # Pipeline config with relaxed parameters for speed and synthetic data
    config = PipelineConfig(
        output_dir=str(tmp_path / "output"),
        dense_stereo=DenseStereoConfig(num_depths=16, window_size=5),
        fusion=FusionConfig(
            min_consistent_views=1,
            min_confidence=0.1,
            depth_tolerance=0.05,
        ),
        surface=SurfaceConfig(method="heightfield", grid_resolution=0.01),
    )

    ctx = PipelineContext(
        config=config,
        calibration=calibration,
        undistortion_maps=undistortion_maps,
        projection_models=projection_models,
        pairs=pairs,
        ring_cameras=calibration.ring_cameras,
        auxiliary_cameras=[],
        device="cpu",
        masks=None,
    )

    # Render synthetic checkerboard images
    images = {}
    for name in calibration.ring_cameras:
        images[name] = render_synthetic_image(
            projection_models[name], IMAGE_H, IMAGE_W, SAND_Z, checkerboard
        )

    return {
        "ctx": ctx,
        "images": images,
        "calibration": calibration,
        "projection_models": projection_models,
        "tmp_path": tmp_path,
    }


@pytest.fixture
def synthetic_scene(tmp_path):
    """Pytest fixture wrapping build_synthetic_scene."""
    return build_synthetic_scene(tmp_path)


# --- Tests ---


@pytest.mark.slow
def test_end_to_end_reconstruction(synthetic_scene):
    """Run the full pipeline on synthetic data and verify outputs.

    Verifies directory structure, sparse cloud, depth maps, fused point
    cloud, and mesh existence and basic validity. Uses no mocking -- runs
    real SuperPoint, LightGlue, plane sweep, Open3D fusion, and surface
    reconstruction.
    """
    ctx = synthetic_scene["ctx"]
    images = synthetic_scene["images"]
    tmp_path = synthetic_scene["tmp_path"]

    # Run the full pipeline
    process_frame(0, images, ctx)

    # --- Verify output directory structure ---
    frame_dir = tmp_path / "output" / "frame_000000"
    assert frame_dir.exists()
    assert (frame_dir / "sparse").is_dir()
    assert (frame_dir / "depth_maps").is_dir()
    assert (frame_dir / "point_cloud").is_dir()
    assert (frame_dir / "mesh").is_dir()

    # --- Verify sparse cloud ---
    sparse_path = frame_dir / "sparse" / "sparse_cloud.pt"
    assert sparse_path.exists()
    sparse_cloud = torch.load(sparse_path, weights_only=True)
    assert "points_3d" in sparse_cloud
    sparse_cloud["points_3d"].shape[0]

    # --- Verify depth maps ---
    depth_files = list((frame_dir / "depth_maps").glob("*.npz"))
    assert len(depth_files) > 0, "No depth maps produced"

    # Check structure and content of at least one depth map
    data = np.load(depth_files[0])
    assert "depth" in data
    assert "confidence" in data
    depth = data["depth"]
    assert depth.shape == (IMAGE_H, IMAGE_W)

    # At least some pixels should have valid (non-NaN) depth
    valid_count = np.isfinite(depth).sum()
    assert valid_count > 0, f"Depth map is entirely NaN: {depth_files[0].name}"

    # --- Verify fused point cloud ---
    pcd_path = frame_dir / "point_cloud" / "fused.ply"
    assert pcd_path.exists()
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    n_points = len(pcd.points)
    assert n_points > 0, "Fused point cloud has 0 points"

    # --- Verify mesh ---
    mesh_path = frame_dir / "mesh" / "surface.ply"
    assert mesh_path.exists()
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    n_triangles = len(mesh.triangles)
    assert n_triangles > 0, "Mesh has 0 triangles"

    # --- Sanity check: fused point Z range ---
    points = np.asarray(pcd.points)
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    # Points should be near SAND_Z (1.0); definitely below water surface
    # and not absurdly deep
    assert z_min > WATER_Z - 0.5, f"Points above water: z_min={z_min}"
    assert z_max < SAND_Z + 1.0, f"Points way too deep: z_max={z_max}"


@pytest.mark.slow
def test_end_to_end_with_evaluation(synthetic_scene):
    """Run pipeline and verify evaluation tools work with pipeline output.

    Runs the full pipeline, then loads the sparse cloud and uses the
    evaluation API (reprojection_error) to verify it functions correctly
    with real pipeline output.
    """
    ctx = synthetic_scene["ctx"]
    images = synthetic_scene["images"]
    tmp_path = synthetic_scene["tmp_path"]
    projection_models = synthetic_scene["projection_models"]

    # Run the full pipeline
    process_frame(0, images, ctx)

    # Load sparse cloud
    sparse_cloud = torch.load(
        tmp_path / "output" / "frame_000000" / "sparse" / "sparse_cloud.pt",
        weights_only=True,
    )

    # If we have sparse points, test evaluation tools
    if sparse_cloud["points_3d"].shape[0] > 0:
        from aquamvs.evaluation import reprojection_error

        points_3d = sparse_cloud["points_3d"]

        # Build observations by projecting sparse points through each camera.
        # This creates a self-consistent set of observations to verify the
        # evaluation API works without crashing on pipeline output.
        observations = {}
        for cam_name, model in projection_models.items():
            projected, valid = model.project(points_3d)
            # model.project already sets invalid projections to NaN
            observations[cam_name] = projected

        # Compute reprojection error (self-reprojection should be near zero)
        result = reprojection_error(points_3d, observations, projection_models)

        # Verify result structure
        assert "mean_error" in result
        assert "per_camera" in result
        assert "errors" in result

        # Self-reprojection error should be very small (< 1 pixel)
        if not np.isnan(result["mean_error"]):
            assert result["mean_error"] < 1.0, (
                f"Self-reprojection error too high: {result['mean_error']:.4f}px"
            )
