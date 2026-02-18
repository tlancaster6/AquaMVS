"""Point cloud to mesh conversion and surface reconstruction."""

import logging
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.interpolate import griddata

from .config import ReconstructionConfig

logger = logging.getLogger(__name__)


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
    config: ReconstructionConfig,
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
    config: ReconstructionConfig,
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
    gc_r = griddata(
        points[:, :2], colors[:, 0], (gx, gy), method="linear", fill_value=0
    )
    gc_g = griddata(
        points[:, :2], colors[:, 1], (gx, gy), method="linear", fill_value=0
    )
    gc_b = griddata(
        points[:, :2], colors[:, 2], (gx, gy), method="linear", fill_value=0
    )

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


def reconstruct_bpa(
    pcd: o3d.geometry.PointCloud,
    config: ReconstructionConfig,
) -> o3d.geometry.TriangleMesh:
    """Reconstruct a surface mesh using the Ball Pivoting Algorithm.

    Rolls a ball of varying radii over the point cloud surface, creating
    triangles where the ball touches three points without enclosing others.
    Best for incomplete point clouds where Poisson would hallucinate beyond
    the data extent.

    Args:
        pcd: Fused point cloud with points, colors, and normals.
        config: Surface configuration.

    Returns:
        Triangle mesh with vertex colors.

    Raises:
        ValueError: If the point cloud has no normals.
    """
    if not pcd.has_normals():
        raise ValueError("Point cloud must have normals for BPA reconstruction.")

    # Handle empty point clouds gracefully
    if len(pcd.points) == 0:
        return o3d.geometry.TriangleMesh()

    # Determine ball radii
    if config.bpa_radii is None:
        # Auto-estimate from nearest-neighbor distance
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [1.0 * avg_dist, 2.0 * avg_dist, 4.0 * avg_dist]
    else:
        radii = config.bpa_radii

    # Create a DoubleVector for Open3D
    radii_vec = o3d.utility.DoubleVector(radii)

    # Run Ball Pivoting Algorithm
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, radii_vec
    )

    # Transfer colors from point cloud to mesh vertices
    mesh = _transfer_colors(pcd, mesh)

    # Compute vertex normals
    mesh.compute_vertex_normals()

    return mesh


def reconstruct_surface(
    pcd: o3d.geometry.PointCloud,
    config: ReconstructionConfig,
) -> o3d.geometry.TriangleMesh:
    """Reconstruct a surface mesh from a fused point cloud.

    Dispatches to the appropriate method based on config.

    Args:
        pcd: Fused point cloud (from fuse_depth_maps).
        config: Surface configuration.

    Returns:
        Triangle mesh with vertex colors.

    Raises:
        ValueError: If config.method is not "poisson", "heightfield", or "bpa".
    """
    match config.surface_method:
        case "poisson":
            mesh = reconstruct_poisson(pcd, config)
        case "heightfield":
            mesh = reconstruct_heightfield(pcd, config)
        case "bpa":
            mesh = reconstruct_bpa(pcd, config)
        case _:
            raise ValueError(
                f"Unknown surface method: {config.surface_method!r}. "
                "Expected 'poisson', 'heightfield', or 'bpa'."
            )

    # Apply simplification if configured
    if config.target_faces is not None:
        mesh = simplify_mesh(mesh, config.target_faces)

    return mesh


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


def simplify_mesh(
    mesh: o3d.geometry.TriangleMesh,
    target_faces: int,
) -> o3d.geometry.TriangleMesh:
    """Simplify a triangle mesh using quadric decimation.

    Reduces mesh complexity while preserving shape and features.
    Useful for reducing file size or improving rendering performance.

    Args:
        mesh: Input triangle mesh.
        target_faces: Target number of triangles after simplification.

    Returns:
        Simplified triangle mesh.
    """
    original_faces = len(mesh.triangles)
    logger.info(
        f"Simplifying mesh: {original_faces} faces -> target {target_faces} faces"
    )

    simplified = mesh.simplify_quadric_decimation(
        target_number_of_triangles=target_faces
    )

    actual_faces = len(simplified.triangles)
    logger.info(f"Simplification result: {actual_faces} faces")

    if actual_faces > target_faces * 1.1:
        logger.warning(
            f"Simplification fell short: {actual_faces} faces "
            f"(target was {target_faces})"
        )

    return simplified


def export_mesh(
    input_path: str | Path,
    output_path: str | Path,
    simplify: int | None = None,
) -> None:
    """Export a mesh to a different format with optional simplification.

    Supports conversion between PLY, OBJ, STL, GLTF, and GLB formats.
    STL export automatically computes vertex normals if missing.

    Args:
        input_path: Path to input mesh file (typically .ply).
        output_path: Path to output mesh file. Format determined by extension.
        simplify: Optional target face count for mesh simplification.

    Raises:
        ValueError: If input mesh is empty or invalid.
        RuntimeError: If mesh export fails.
    """
    # Load mesh
    logger.info(f"Loading mesh from {input_path}")
    mesh = o3d.io.read_triangle_mesh(str(input_path))

    # Validate
    if not mesh.has_vertices() or len(mesh.vertices) == 0:
        raise ValueError(f"Input mesh has no vertices: {input_path}")

    logger.info(
        f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces"
    )

    # Simplify if requested
    if simplify is not None:
        mesh = simplify_mesh(mesh, simplify)

    # Format-specific preprocessing
    output_path = Path(output_path)
    output_format = output_path.suffix.lower()

    if output_format == ".stl":
        # STL requires triangle normals (and vertex normals for shading)
        if not mesh.has_triangle_normals():
            logger.info("Computing triangle normals for STL export")
            mesh.compute_triangle_normals()
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        # Warn about color loss
        if mesh.has_vertex_colors():
            logger.warning(
                "STL format does not support vertex colors (colors will be lost)"
            )

    # Export
    logger.info(f"Exporting to {output_path} (format: {output_format})")
    success = o3d.io.write_triangle_mesh(str(output_path), mesh)

    if not success:
        raise RuntimeError(f"Failed to write mesh to {output_path}")

    logger.info(
        f"Export complete: {len(mesh.triangles)} faces written to {output_path}"
    )
