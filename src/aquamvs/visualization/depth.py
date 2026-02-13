"""Depth and confidence map rendering."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def render_depth_map(
    depth_map: np.ndarray,
    output_path: str | Path,
    camera_name: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    dpi: int = 150,
) -> None:
    """Render a depth map as a colormapped image with colorbar.

    Args:
        depth_map: Depth map, shape (H, W), float32. NaN for invalid pixels.
        output_path: Path to save the PNG image.
        camera_name: Camera name for the title.
        vmin: Colormap minimum. If None, auto from valid data.
        vmax: Colormap maximum. If None, auto from valid data.
        dpi: Output resolution.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Set NaN background color
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="0.8")  # gray for NaN

    # Auto-range from valid data
    valid = depth_map[np.isfinite(depth_map)]
    if vmin is None and len(valid) > 0:
        vmin = float(valid.min())
    if vmax is None and len(valid) > 0:
        vmax = float(valid.max())

    im = ax.imshow(depth_map, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Depth (m)")

    title = "Depth Map"
    if camera_name:
        title += f" - {camera_name}"
    ax.set_title(title)
    ax.axis("off")

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def render_confidence_map(
    confidence: np.ndarray,
    output_path: str | Path,
    camera_name: str = "",
    dpi: int = 150,
) -> None:
    """Render a confidence map as a colormapped image with colorbar.

    Args:
        confidence: Confidence map, shape (H, W), float32 in [0, 1].
        output_path: Path to save the PNG image.
        camera_name: Camera name for the title.
        dpi: Output resolution.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Use plasma colormap, fixed range [0, 1]
    cmap = plt.cm.plasma
    im = ax.imshow(confidence, cmap=cmap, vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Confidence")

    title = "Confidence Map"
    if camera_name:
        title += f" - {camera_name}"
    ax.set_title(title)
    ax.axis("off")

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def render_all_depth_maps(
    depth_maps: dict[str, np.ndarray],
    confidence_maps: dict[str, np.ndarray],
    output_dir: str | Path,
    dpi: int = 150,
) -> None:
    """Render depth and confidence maps for all cameras.

    Saves to output_dir/depth_{cam}.png and output_dir/confidence_{cam}.png.

    Args:
        depth_maps: Camera name to depth map (H, W) mapping.
        confidence_maps: Camera name to confidence map (H, W) mapping.
        output_dir: Directory to save images.
        dpi: Output resolution.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute shared vmin/vmax across all cameras for consistent coloring
    all_valid_depths = []
    for depth_map in depth_maps.values():
        valid = depth_map[np.isfinite(depth_map)]
        if len(valid) > 0:
            all_valid_depths.append(valid)

    if len(all_valid_depths) > 0:
        all_valid_depths = np.concatenate(all_valid_depths)
        vmin = float(all_valid_depths.min())
        vmax = float(all_valid_depths.max())
    else:
        vmin = None
        vmax = None

    # Render depth and confidence for each camera
    for camera_name in depth_maps.keys():
        depth_map = depth_maps[camera_name]
        confidence = confidence_maps[camera_name]

        # Depth map
        depth_path = output_dir / f"depth_{camera_name}.png"
        render_depth_map(
            depth_map,
            depth_path,
            camera_name=camera_name,
            vmin=vmin,
            vmax=vmax,
            dpi=dpi,
        )

        # Confidence map
        confidence_path = output_dir / f"confidence_{camera_name}.png"
        render_confidence_map(
            confidence,
            confidence_path,
            camera_name=camera_name,
            dpi=dpi,
        )
