"""Synthetic scene generation for benchmark ground truth."""

import logging
from typing import Callable

import numpy as np
import open3d as o3d
from numpy.typing import NDArray

from ..projection.protocol import ProjectionModel

logger = logging.getLogger(__name__)


def get_reference_geometry() -> dict:
    """Get reference 12-camera ring geometry constants.

    Returns standard experimental setup geometry for consistent
    synthetic scene generation across tests.

    Returns:
        Dictionary with keys:
            - ring_radius_m: Radius of camera ring (meters)
            - water_z_m: Water surface Z coordinate (meters)
            - n_water: Refractive index of water
            - num_cameras: Number of ring cameras
    """
    return {
        "ring_radius_m": 0.635,
        "water_z_m": 0.978,
        "n_water": 1.333,
        "num_cameras": 12,
    }


def create_flat_plane_scene(
    depth_z: float,
    bounds: tuple[float, float, float, float],
    resolution: float = 0.005,
) -> tuple[o3d.geometry.TriangleMesh, Callable[[NDArray, NDArray], NDArray]]:
    """Create a flat plane scene with analytic ground truth.

    Args:
        depth_z: World Z coordinate of the plane (underwater target depth).
        bounds: (x_min, x_max, y_min, y_max) bounds in world coordinates.
        resolution: Mesh vertex spacing in meters (default 5mm).

    Returns:
        Tuple of (mesh, analytic_depth_function).
        - mesh: Open3D triangle mesh of the plane
        - analytic_depth_function: f(x, y) -> z returning constant depth_z
    """
    x_min, x_max, y_min, y_max = bounds

    # Generate grid of vertices
    x = np.arange(x_min, x_max + resolution, resolution)
    y = np.arange(y_min, y_max + resolution, resolution)
    xx, yy = np.meshgrid(x, y)

    # Flatten to vertex list
    vertices = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, depth_z)])

    # Generate triangle indices (two triangles per grid cell)
    nx = len(x)
    ny = len(y)
    triangles = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            # Vertex indices for grid cell corners
            v0 = j * nx + i
            v1 = j * nx + (i + 1)
            v2 = (j + 1) * nx + i
            v3 = (j + 1) * nx + (i + 1)

            # Two triangles
            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])

    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh.compute_vertex_normals()

    # Analytic depth function (trivially constant)
    def analytic_fn(x: NDArray, y: NDArray) -> NDArray:
        """Return constant depth for any (x, y) position.

        Args:
            x: X coordinates (meters).
            y: Y coordinates (meters).

        Returns:
            Z coordinates (all equal to depth_z).
        """
        return np.full_like(x, depth_z)

    return mesh, analytic_fn


def create_undulating_scene(
    base_depth_z: float,
    amplitude: float,
    wavelength: float,
    bounds: tuple[float, float, float, float],
    resolution: float = 0.005,
) -> tuple[o3d.geometry.TriangleMesh, Callable[[NDArray, NDArray], NDArray]]:
    """Create an undulating sand-like surface with known analytic form.

    Surface equation: z(x,y) = base_depth_z + amplitude * (sin(2πx/λ) + sin(2πy/(1.3λ)))

    Args:
        base_depth_z: Base world Z coordinate (mean depth).
        amplitude: Amplitude of undulation (meters, e.g., 0.005 = 5mm).
        wavelength: Wavelength of undulation (meters, e.g., 0.05 = 5cm).
        bounds: (x_min, x_max, y_min, y_max) bounds in world coordinates.
        resolution: Mesh vertex spacing in meters (default 5mm).

    Returns:
        Tuple of (mesh, analytic_depth_function).
        - mesh: Open3D triangle mesh of the undulating surface
        - analytic_depth_function: f(x, y) -> z returning analytic Z values
    """
    x_min, x_max, y_min, y_max = bounds

    # Generate grid of vertices
    x = np.arange(x_min, x_max + resolution, resolution)
    y = np.arange(y_min, y_max + resolution, resolution)
    xx, yy = np.meshgrid(x, y)

    # Analytic surface function
    def analytic_fn(x: NDArray, y: NDArray) -> NDArray:
        """Compute Z coordinate for given (x, y) positions.

        Args:
            x: X coordinates (meters).
            y: Y coordinates (meters).

        Returns:
            Z coordinates following undulating surface equation.
        """
        return base_depth_z + amplitude * (
            np.sin(2 * np.pi * x / wavelength)
            + np.sin(2 * np.pi * y / (wavelength * 1.3))
        )

    # Compute Z values at grid points
    zz = analytic_fn(xx, yy)

    # Flatten to vertex list
    vertices = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # Generate triangle indices (same pattern as flat plane)
    nx = len(x)
    ny = len(y)
    triangles = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v0 = j * nx + i
            v1 = j * nx + (i + 1)
            v2 = (j + 1) * nx + i
            v3 = (j + 1) * nx + (i + 1)

            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])

    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh.compute_vertex_normals()

    return mesh, analytic_fn


def generate_ground_truth_depth_maps(
    scene_mesh: o3d.geometry.TriangleMesh,
    projection_models: dict[str, ProjectionModel],
    image_shape: tuple[int, int],
) -> dict[str, NDArray[np.float64]]:
    """Generate ground truth depth maps by ray-casting scene mesh.

    Uses Open3D RaycastingScene for efficient ray-mesh intersection.
    Converts hit distances to ray-depth parameterization matching pipeline convention.

    Args:
        scene_mesh: Ground truth surface mesh.
        projection_models: Dict mapping camera name to ProjectionModel.
        image_shape: (height, width) of output depth maps.

    Returns:
        Dict mapping camera name to H × W depth map (NaN for invalid pixels).
    """
    # Create raycasting scene
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(scene_mesh))

    depth_maps = {}

    for cam_name, model in projection_models.items():
        height, width = image_shape
        depth_map = np.full((height, width), np.nan, dtype=np.float64)

        # Generate pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        pixels = np.column_stack([u.ravel(), v.ravel()])

        # Cast rays for all pixels
        origins, directions = model.cast_ray(pixels)

        # Convert to Open3D format (float32)
        rays = np.column_stack([origins, directions]).astype(np.float32)
        rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

        # Ray cast
        result = scene.cast_rays(rays_tensor)
        t_hit = result["t_hit"].numpy()

        # Convert hit distances to depth map
        # t_hit is distance along ray from origin
        # This matches our ray-depth parameterization: point = origin + t * direction
        valid = np.isfinite(t_hit)
        depths = t_hit.reshape(height, width)

        # Mark invalid pixels (no hit) as NaN
        depth_map[~valid.reshape(height, width)] = np.nan
        depth_map[valid.reshape(height, width)] = depths[valid.reshape(height, width)]

        depth_maps[cam_name] = depth_map

        logger.info(
            "Generated depth map for %s: %.1f%% valid pixels",
            cam_name,
            100.0 * np.sum(valid) / valid.size,
        )

    return depth_maps


__all__ = [
    "create_flat_plane_scene",
    "create_undulating_scene",
    "generate_ground_truth_depth_maps",
    "get_reference_geometry",
]
