"""3D point cloud and mesh rendering."""

import logging
import os
from pathlib import Path

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)


def _offscreen_available() -> bool:
    """Check if Open3D offscreen rendering is available.

    Suppresses native stderr during the probe because Open3D's C++
    layer prints an ``[Open3D Error]`` message to fd 2 before raising
    the Python exception, which ``try/except`` alone cannot catch.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    try:
        renderer = o3d.visualization.rendering.OffscreenRenderer(64, 64)
        del renderer
        return True
    except Exception:
        return False
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)
        os.close(devnull_fd)


OFFSCREEN_AVAILABLE = _offscreen_available()


def compute_canonical_viewpoints(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute canonical camera viewpoints from a geometry's bounding box.

    Derives three viewpoints that provide complementary views of the
    reconstruction: top-down (plan view), oblique (3/4 view), and
    side (profile view).

    Args:
        bbox_min: Minimum corner of the bounding box, shape (3,).
        bbox_max: Maximum corner of the bounding box, shape (3,).

    Returns:
        Dict mapping viewpoint name to camera parameters:
            "top": Looking straight down at the center.
            "oblique": 45-degree view from one corner.
            "side": Horizontal view from one side.
        Each value is a dict with:
            "eye": Camera position, shape (3,).
            "center": Look-at point, shape (3,).
            "up": Up vector, shape (3,).
    """
    center = (bbox_min + bbox_max) / 2.0
    extent = bbox_max - bbox_min
    max_extent = max(extent[0], extent[1], extent[2], 0.1)  # avoid zero

    # Top-down: eye above center, looking down (+Z direction in world = into water)
    # Z-down world: eye at smaller Z than center (above the water)
    # Up vector is +Y (forward in world frame)
    top = {
        "eye": np.array([center[0], center[1], bbox_min[2] - max_extent * 1.5]),
        "center": center.copy(),
        "up": np.array([0.0, 1.0, 0.0]),
    }

    # Oblique: 45 degrees from one corner
    # Up vector is -Z (pointing "up" in real world = negative Z)
    offset = max_extent * 1.0
    oblique = {
        "eye": np.array([
            center[0] + offset,
            center[1] - offset,
            bbox_min[2] - offset,
        ]),
        "center": center.copy(),
        "up": np.array([0.0, 0.0, -1.0]),  # Z-down world
    }

    # Side: horizontal view from one side
    # Up vector is -Z (pointing "up" in real world)
    side = {
        "eye": np.array([center[0], center[1] - max_extent * 2.0, center[2]]),
        "center": center.copy(),
        "up": np.array([0.0, 0.0, -1.0]),
    }

    return {"top": top, "oblique": oblique, "side": side}


def render_geometry(
    geometry: o3d.geometry.Geometry,
    eye: np.ndarray,
    center: np.ndarray,
    up: np.ndarray,
    output_path: str | Path,
    width: int = 1280,
    height: int = 960,
    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    point_size: float = 2.0,
) -> None:
    """Render an Open3D geometry from a given viewpoint using offscreen rendering.

    Uses Open3D's OffscreenRenderer for headless rendering. Works without
    a display server.

    Args:
        geometry: Open3D PointCloud or TriangleMesh to render.
        eye: Camera position, shape (3,).
        center: Look-at point, shape (3,).
        up: Up vector, shape (3,).
        output_path: Path to save the PNG image.
        width: Image width in pixels.
        height: Image height in pixels.
        background_color: RGB background color, each in [0, 1].
        point_size: Point size for point cloud rendering (ignored for meshes).
    """
    if not OFFSCREEN_AVAILABLE:
        logger.warning(
            "Open3D offscreen rendering unavailable - skipping render to %s",
            output_path,
        )
        return

    # Create offscreen renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background(np.array([*background_color, 1.0]))

    # Material
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    if isinstance(geometry, o3d.geometry.PointCloud):
        mat.point_size = point_size

    renderer.scene.add_geometry("scene", geometry, mat)

    # Set camera
    renderer.setup_camera(60.0, center, eye, up)

    # Render and save
    img = renderer.render_to_image()
    o3d.io.write_image(str(output_path), img)
    logger.info("Rendered to %s", output_path)


def render_scene(
    geometry: o3d.geometry.Geometry,
    output_dir: str | Path,
    prefix: str = "scene",
    width: int = 1280,
    height: int = 960,
) -> None:
    """Render a geometry from all canonical viewpoints.

    Saves {prefix}_{viewpoint}.png for each viewpoint (top, oblique, side).

    Args:
        geometry: Open3D PointCloud or TriangleMesh.
        output_dir: Directory to save images.
        prefix: Filename prefix (e.g., "fused" or "mesh").
        width: Image width in pixels.
        height: Image height in pixels.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute bounding box
    bbox = geometry.get_axis_aligned_bounding_box()
    bbox_min = np.asarray(bbox.min_bound)
    bbox_max = np.asarray(bbox.max_bound)

    # Compute viewpoints
    viewpoints = compute_canonical_viewpoints(bbox_min, bbox_max)

    # Render each viewpoint
    for viewpoint_name, params in viewpoints.items():
        output_path = output_dir / f"{prefix}_{viewpoint_name}.png"
        render_geometry(
            geometry,
            eye=params["eye"],
            center=params["center"],
            up=params["up"],
            output_path=output_path,
            width=width,
            height=height,
        )


def render_all_scenes(
    point_cloud: o3d.geometry.PointCloud | None,
    mesh: o3d.geometry.TriangleMesh | None,
    output_dir: str | Path,
    width: int = 1280,
    height: int = 960,
) -> None:
    """Render both point cloud and mesh from canonical viewpoints.

    Saves to output_dir:
        fused_{viewpoint}.png -- point cloud renders
        mesh_{viewpoint}.png -- mesh renders

    Args:
        point_cloud: Fused point cloud (may be None if fusion failed).
        mesh: Reconstructed mesh (may be None if reconstruction failed).
        output_dir: Directory to save images.
        width: Image width.
        height: Image height.
    """
    if point_cloud is None and mesh is None:
        logger.warning("No geometry to render - skipping scene rendering")
        return

    if point_cloud is not None:
        logger.info("Rendering fused point cloud from canonical viewpoints")
        render_scene(point_cloud, output_dir, prefix="fused", width=width, height=height)
    else:
        logger.warning("Point cloud is None - skipping fused renders")

    if mesh is not None:
        logger.info("Rendering mesh from canonical viewpoints")
        render_scene(mesh, output_dir, prefix="mesh", width=width, height=height)
    else:
        logger.warning("Mesh is None - skipping mesh renders")