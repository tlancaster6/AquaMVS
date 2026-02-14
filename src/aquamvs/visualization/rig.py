"""Camera rig diagram generation."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def render_rig_diagram(
    camera_positions: dict[str, np.ndarray],
    camera_rotations: dict[str, np.ndarray],
    water_z: float,
    output_path: str | Path,
    K: np.ndarray | None = None,
    image_size: tuple[int, int] | None = None,
    point_cloud_points: np.ndarray | None = None,
    frustum_scale: float = 0.1,
    dpi: int = 150,
) -> None:
    """Render a 3D camera rig diagram.

    Shows camera positions with frustums, the water surface plane,
    and optionally reconstructed points for spatial context.

    Args:
        camera_positions: Camera name to world-frame position (3,) mapping.
            Computed from CalibrationData.camera_positions().
        camera_rotations: Camera name to rotation matrix R (3, 3) mapping.
            World-to-camera rotation (p_cam = R @ p_world + t).
        water_z: Z-coordinate of the water surface in world frame.
        output_path: Path to save the PNG image.
        K: Optional intrinsic matrix (3, 3) for frustum aspect ratio.
            If None, uses a square frustum.
        image_size: Optional (width, height) for frustum aspect ratio.
        point_cloud_points: Optional (N, 3) array of 3D points to overlay.
        frustum_scale: Scale factor for frustum wireframe size (meters).
        dpi: Output resolution.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw each camera
    for name, pos in camera_positions.items():
        # Camera marker
        ax.scatter(*pos, c="red", s=40, marker="^", zorder=5)
        ax.text(pos[0], pos[1], pos[2], f"  {name}", fontsize=6)

        # Frustum wireframe
        if name in camera_rotations:
            R = camera_rotations[name]
            _draw_frustum(ax, pos, R, frustum_scale, K, image_size)

    # Water surface plane
    _draw_water_plane(ax, camera_positions, water_z)

    # Optional point cloud overlay
    if point_cloud_points is not None and len(point_cloud_points) > 0:
        # Subsample if too many points
        pts = point_cloud_points
        if len(pts) > 5000:
            idx = np.random.default_rng(42).choice(len(pts), 5000, replace=False)
            pts = pts[idx]
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap="viridis", s=1, alpha=0.3
        )

    # Labels and formatting
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Camera Rig Diagram")

    # Invert Z axis so Z-down appears intuitive (cameras on top, water below)
    ax.invert_zaxis()

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def _draw_frustum(
    ax,
    camera_center: np.ndarray,
    R: np.ndarray,
    scale: float,
    K: np.ndarray | None = None,
    image_size: tuple[int, int] | None = None,
) -> None:
    """Draw a camera frustum wireframe on a 3D axis.

    The frustum is a wireframe pyramid from the camera center
    to four corners representing the image plane.

    Args:
        ax: Matplotlib 3D axis.
        camera_center: Camera position in world frame, shape (3,).
        R: World-to-camera rotation matrix, shape (3, 3).
        scale: Frustum depth (distance from center to image plane) in meters.
        K: Intrinsic matrix for aspect ratio. If None, square frustum.
        image_size: (width, height) for aspect ratio.
    """
    # Determine aspect ratio
    if K is not None and image_size is not None or image_size is not None:
        width, height = image_size
        aspect = width / height
    else:
        aspect = 4.0 / 3.0  # Default aspect ratio

    # Create 4 corner points on a virtual image plane in camera frame
    # Camera frame: +X right, +Y down, +Z forward (optical axis)
    # Place corners at distance `scale` along the optical axis
    # For a simple wireframe, use normalized image coordinates
    half_width = aspect / 2.0
    half_height = 0.5

    # Four corners in camera frame (unnormalized, at distance scale)
    corners_cam = np.array(
        [
            [half_width * scale, half_height * scale, scale],  # Top-right
            [-half_width * scale, half_height * scale, scale],  # Top-left
            [-half_width * scale, -half_height * scale, scale],  # Bottom-left
            [half_width * scale, -half_height * scale, scale],  # Bottom-right
        ]
    )

    # Transform corners from camera frame to world frame
    # p_world = R^T @ (p_cam - 0) + camera_center
    # Since p_cam is relative to camera center, just rotate
    R_inv = R.T
    corners_world = (R_inv @ corners_cam.T).T + camera_center

    # Draw lines from camera center to each corner
    for corner in corners_world:
        ax.plot(
            [camera_center[0], corner[0]],
            [camera_center[1], corner[1]],
            [camera_center[2], corner[2]],
            color="black",
            alpha=0.5,
            linewidth=0.5,
        )

    # Draw lines connecting corners to form the image plane rectangle
    for i in range(4):
        j = (i + 1) % 4
        ax.plot(
            [corners_world[i, 0], corners_world[j, 0]],
            [corners_world[i, 1], corners_world[j, 1]],
            [corners_world[i, 2], corners_world[j, 2]],
            color="black",
            alpha=0.5,
            linewidth=0.5,
        )


def _draw_water_plane(
    ax,
    camera_positions: dict[str, np.ndarray],
    water_z: float,
) -> None:
    """Draw a semi-transparent water surface plane.

    Sizes the plane to cover the camera positions with some padding.

    Args:
        ax: Matplotlib 3D axis.
        camera_positions: Camera name to world-frame position (3,) mapping.
        water_z: Z-coordinate of the water surface in world frame.
    """
    # Compute bounding box of camera positions
    positions = np.array(list(camera_positions.values()))
    x_min, y_min = positions[:, :2].min(axis=0)
    x_max, y_max = positions[:, :2].max(axis=0)

    # Add 20% padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding_x = 0.2 * x_range if x_range > 0 else 0.2
    padding_y = 0.2 * y_range if y_range > 0 else 0.2

    x_min -= padding_x
    x_max += padding_x
    y_min -= padding_y
    y_max += padding_y

    # Create rectangular plane at Z = water_z
    corners = [
        [x_min, y_min, water_z],
        [x_max, y_min, water_z],
        [x_max, y_max, water_z],
        [x_min, y_max, water_z],
    ]

    # Draw as a semi-transparent polygon
    poly = Poly3DCollection([corners], alpha=0.2, facecolor="cyan", edgecolor="blue")
    ax.add_collection3d(poly)
