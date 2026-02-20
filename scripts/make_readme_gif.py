"""Generate a turntable GIF of a PLY mesh for the README header."""

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyvista as pv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ply", type=Path, help="Path to .ply mesh file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("docs/header.gif"),
        help="Output GIF path",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument(
        "--n-frames", type=int, default=90, help="Frames per full rotation"
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--elevation",
        type=float,
        default=45.0,
        help="Camera elevation angle in degrees",
    )
    parser.add_argument(
        "--bg-color", type=str, default="white", help="Background color (hex or name)"
    )
    args = parser.parse_args()

    # Load mesh
    mesh = pv.read(str(args.ply))
    centroid = mesh.center

    # Set up offscreen plotter
    plotter = pv.Plotter(off_screen=True, window_size=[args.width, args.height])
    plotter.background_color = args.bg_color
    plotter.add_mesh(mesh, scalars="RGB", rgb=True, smooth_shading=True)

    # Position camera at oblique angle looking at centroid
    plotter.camera.focal_point = centroid
    radius = mesh.length * 1.5  # 1.5x bounding box diagonal for comfortable framing
    # +Z is down in our coordinate system, so "up" for the camera is -Z
    plotter.camera.up = (0.0, 0.0, -1.0)

    # Render turntable frames
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plotter.open_gif(str(args.output), fps=args.fps)

    for i in range(args.n_frames):
        azimuth = 360.0 * i / args.n_frames
        plotter.camera.position = _spherical_to_cartesian(
            centroid, radius, elevation_deg=args.elevation, azimuth_deg=azimuth
        )
        plotter.reset_camera_clipping_range()
        plotter.write_frame()

    plotter.close()

    raw_size = args.output.stat().st_size
    print(
        f"Wrote {args.output} ({args.n_frames} frames, {args.n_frames / args.fps:.1f}s, {raw_size / 1e6:.1f} MB)"
    )

    # Compress with gifsicle if available
    if shutil.which("gifsicle"):
        subprocess.run(
            ["gifsicle", "-O3", "--lossy=80", str(args.output), "-o", str(args.output)],
            check=True,
        )
        compressed_size = args.output.stat().st_size
        print(
            f"Compressed with gifsicle: {compressed_size / 1e6:.1f} MB ({compressed_size / raw_size:.0%} of original)"
        )
    else:
        print("gifsicle not found — skipping compression")


def _spherical_to_cartesian(
    center: np.ndarray,
    radius: float,
    elevation_deg: float,
    azimuth_deg: float,
) -> tuple[float, float, float]:
    """Convert spherical coordinates to Cartesian camera position.

    Elevation is measured from the horizontal plane (positive = looking down,
    consistent with Z-down convention). Azimuth rotates about +Z.
    """
    el = np.radians(elevation_deg)
    az = np.radians(azimuth_deg)
    # Horizontal radius at this elevation
    r_horiz = radius * np.cos(el)
    x = center[0] + r_horiz * np.cos(az)
    y = center[1] + r_horiz * np.sin(az)
    # Positive elevation -> camera above target -> smaller Z (Z-down)
    z = center[2] - radius * np.sin(el)
    return x, y, z


if __name__ == "__main__":
    main()
