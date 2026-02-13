"""Point cloud to mesh conversion and surface reconstruction."""

from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.interpolate import griddata

from .config import SurfaceConfig


def _transfer_colors(
    pcd: o3d.geometry.PointCloud,
    mesh: o3d.geometry.TriangleMesh,
) -> o3d.geometry.TriangleMesh:
    """Transfer colors from a point cloud to mesh vertices via nearest-neighbor lookup.

    Args:
        pcd: Point cloud with colors.
        mesh: Triangle mesh (vertices only, no colors yet).

    Returns:
        The same mesh with vertex_colors set.
    """
    if not pcd.has_colors():
        return mesh

    # Build KD-tree on point cloud
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    pcd_colors = np.asarray(pcd.colors)
    mesh_vertices = np.asarray(mesh.vertices)

    vertex_colors = np.zeros_like(mesh_vertices)
    for i in range(len(mesh_vertices)):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(mesh_vertices[i], 1)
        vertex_colors[i] = pcd_colors[idx[0]]

    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return mesh


def reconstruct_poisson(
    pcd: o3d.geometry.PointCloud,
    config: SurfaceConfig,
) -> o3d.geometry.TriangleMesh:
    """Reconstruct a surface mesh using Poisson surface reconstruction.

    Produces a watertight mesh from an oriented point cloud. Best for
    general-purpose use but may hallucinate geometry in regions without
    point coverage.

    Args:
        pcd: Fused point cloud with points, colors, and normals.
        config: Surface configuration.

    Returns:
        Triangle mesh with vertex colors.

    Raises:
        ValueError: If the point cloud has no normals.
    """
    if not pcd.has_normals():
        raise ValueError("Point cloud must have normals for Poisson reconstruction.")

    # Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=config.poisson_depth
    )

    # Remove low-density vertices (Poisson fills gaps with hallucinated geometry)
    # Use a percentile-based threshold to trim the mesh to the data extent
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.01)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # Transfer colors from point cloud to mesh vertices
    # For each mesh vertex, find the nearest point cloud point and copy its color
    mesh = _transfer_colors(pcd, mesh)

    return mesh


def reconstruct_heightfield(
    pcd: o3d.geometry.PointCloud,
    config: SurfaceConfig,
) -> o3d.geometry.TriangleMesh:
    """Reconstruct a surface mesh using height-field interpolation.

    Projects the point cloud onto a regular XY grid and interpolates Z
    values using scipy's griddata. Best for surfaces that are approximately
    single-valued in Z (e.g., a sand bed viewed from above).

    Args:
        pcd: Fused point cloud with points and colors.
        config: Surface configuration.

    Returns:
        Triangle mesh with vertex colors.

    Raises:
        ValueError: If the point cloud has fewer than 4 points.
    """
    points = np.asarray(pcd.points)  # (N, 3)
    colors = np.asarray(pcd.colors)  # (N, 3)

    if len(points) < 4:
        raise ValueError(
            f"Need at least 4 points for height-field interpolation, got {len(points)}."
        )

    # Step 1: Build a regular XY grid
    x_min, y_min = points[:, 0].min(), points[:, 1].min()
    x_max, y_max = points[:, 0].max(), points[:, 1].max()

    res = config.grid_resolution
    grid_x = np.arange(x_min, x_max + res, res)
    grid_y = np.arange(y_min, y_max + res, res)
    gx, gy = np.meshgrid(grid_x, grid_y)  # (Ny, Nx)

    # Step 2: Interpolate Z values onto the grid
    gz = griddata(
        points[:, :2],  # XY coordinates
        points[:, 2],  # Z values
        (gx, gy),
        method="linear",
        fill_value=np.nan,
    )  # (Ny, Nx)

    # Step 3: Interpolate colors onto the grid (per-channel)
    gc_r = griddata(points[:, :2], colors[:, 0], (gx, gy), method="linear", fill_value=0)
    gc_g = griddata(points[:, :2], colors[:, 1], (gx, gy), method="linear", fill_value=0)
    gc_b = griddata(points[:, :2], colors[:, 2], (gx, gy), method="linear", fill_value=0)

    # Step 4: Build mesh from the grid
    Ny, Nx = gz.shape
    valid = ~np.isnan(gz)

    # Create vertex array: only valid grid cells
    # Map grid (i, j) -> vertex index
    vertex_map = np.full((Ny, Nx), -1, dtype=np.int64)
    vertices = []
    vertex_colors = []

    for i in range(Ny):
        for j in range(Nx):
            if valid[i, j]:
                vertex_map[i, j] = len(vertices)
                vertices.append([gx[i, j], gy[i, j], gz[i, j]])
                vertex_colors.append([gc_r[i, j], gc_g[i, j], gc_b[i, j]])

    if len(vertices) == 0:
        return o3d.geometry.TriangleMesh()

    vertices = np.array(vertices)
    vertex_colors = np.array(vertex_colors)

    # Create triangles: for each 2x2 grid cell with all 4 corners valid,
    # create 2 triangles
    triangles = []
    for i in range(Ny - 1):
        for j in range(Nx - 1):
            v00 = vertex_map[i, j]
            v01 = vertex_map[i, j + 1]
            v10 = vertex_map[i + 1, j]
            v11 = vertex_map[i + 1, j + 1]

            if v00 >= 0 and v01 >= 0 and v10 >= 0 and v11 >= 0:
                triangles.append([v00, v10, v01])  # lower-left triangle
                triangles.append([v01, v10, v11])  # upper-right triangle

    if len(triangles) == 0:
        return o3d.geometry.TriangleMesh()

    triangles = np.array(triangles, dtype=np.int32)

    # Step 5: Assemble Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(vertex_colors, 0, 1))

    # Compute normals from geometry
    mesh.compute_vertex_normals()

    return mesh


def reconstruct_surface(
    pcd: o3d.geometry.PointCloud,
    config: SurfaceConfig,
) -> o3d.geometry.TriangleMesh:
    """Reconstruct a surface mesh from a fused point cloud.

    Dispatches to the appropriate method based on config.

    Args:
        pcd: Fused point cloud (from fuse_depth_maps).
        config: Surface configuration.

    Returns:
        Triangle mesh with vertex colors.

    Raises:
        ValueError: If config.method is not "poisson" or "heightfield".
    """
    match config.method:
        case "poisson":
            return reconstruct_poisson(pcd, config)
        case "heightfield":
            return reconstruct_heightfield(pcd, config)
        case _:
            raise ValueError(
                f"Unknown surface method: {config.method!r}. "
                "Expected 'poisson' or 'heightfield'."
            )


def save_mesh(
    mesh: o3d.geometry.TriangleMesh,
    path: str | Path,
) -> None:
    """Save a triangle mesh to a PLY file (binary).

    Args:
        mesh: Open3D TriangleMesh with vertices, triangles, and colors.
        path: Output file path (should end with .ply).
    """
    o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False)


def load_mesh(
    path: str | Path,
) -> o3d.geometry.TriangleMesh:
    """Load a triangle mesh from a PLY file.

    Args:
        path: Path to .ply file.

    Returns:
        Open3D TriangleMesh.
    """
    return o3d.io.read_triangle_mesh(str(path))
