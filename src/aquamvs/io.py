"""I/O adapters for image directory and video input sources."""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def detect_input_type(camera_map: dict[str, str]) -> str:
    """Detect whether camera_map contains image directories or video files.

    Args:
        camera_map: Mapping from camera name to path (directory or video file).

    Returns:
        "images" if all paths are directories, "video" if all paths are files.

    Raises:
        ValueError: If camera_map is empty or contains mixed types.
    """
    if not camera_map:
        raise ValueError("camera_map is empty")

    paths = [Path(p) for p in camera_map.values()]
    is_dir = [p.is_dir() for p in paths]
    is_file = [p.is_file() for p in paths]

    if all(is_dir):
        return "images"
    elif all(is_file):
        return "video"
    else:
        raise ValueError(
            "camera_map contains mixed types (directories and files). "
            "All paths must be either directories or files."
        )


class ImageDirectorySet:
    """Adapter providing VideoSet-compatible interface for image directories.

    Reads synchronized frames from per-camera image directories. All cameras
    must have the same number of images with matching filenames (sorted order).

    Args:
        image_dirs: Mapping from camera name to directory path.

    Raises:
        ValueError: If directories have mismatched file counts or filenames.
    """

    def __init__(self, image_dirs: dict[str, str]):
        self.image_dirs = {name: Path(path) for name, path in image_dirs.items()}
        self._validate_and_index()

    def _validate_and_index(self) -> None:
        """Validate directories and build frame index."""
        # Glob for images in each directory
        image_exts = ("*.png", "*.jpg", "*.jpeg", "*.tiff", "*.tif")
        self.frame_files: dict[str, list[Path]] = {}

        for cam_name, cam_dir in self.image_dirs.items():
            if not cam_dir.exists():
                raise ValueError(f"Camera directory does not exist: {cam_dir}")
            if not cam_dir.is_dir():
                raise ValueError(f"Camera path is not a directory: {cam_dir}")

            # Collect all image files
            files = []
            for ext in image_exts:
                files.extend(cam_dir.glob(ext))

            if not files:
                raise ValueError(f"No images found in directory: {cam_dir}")

            # Sort by filename (not full path)
            files = sorted(files, key=lambda p: p.name)
            self.frame_files[cam_name] = files

        # Validate all cameras have the same number of frames
        frame_counts = {name: len(files) for name, files in self.frame_files.items()}
        if len(set(frame_counts.values())) != 1:
            raise ValueError(
                f"Mismatched frame counts across cameras: {frame_counts}. "
                "All cameras must have the same number of images."
            )

        # Validate all cameras have the same filenames (in sorted order)
        filenames_per_cam = {
            name: [f.name for f in files] for name, files in self.frame_files.items()
        }
        first_cam = next(iter(filenames_per_cam))
        first_filenames = filenames_per_cam[first_cam]

        for cam_name, filenames in filenames_per_cam.items():
            if filenames != first_filenames:
                raise ValueError(
                    f"Filenames do not match between {first_cam} and {cam_name}. "
                    "All cameras must have images with the same filenames."
                )

        self._frame_count = len(first_filenames)
        logger.info(
            "Detected %d frames across %d cameras (image directory input)",
            self._frame_count,
            len(self.image_dirs),
        )

    @property
    def frame_count(self) -> int:
        """Total number of frames available."""
        return self._frame_count

    def __enter__(self):
        """Context manager entry (no-op for image directories)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit (no resources to release)."""
        pass

    def iterate_frames(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> tuple[int, dict[str, np.ndarray]]:
        """Iterate over frames, yielding frame index and per-camera images.

        Args:
            start: First frame index to yield.
            stop: Last frame index (exclusive). None = end of sequence.
            step: Frame step interval.

        Yields:
            Tuple of (frame_idx, images_dict) where images_dict maps
            camera name to BGR image (H, W, 3) uint8.
        """
        if stop is None:
            stop = self._frame_count

        for frame_idx in range(start, min(stop, self._frame_count), step):
            images = {}
            for cam_name, files in self.frame_files.items():
                img_path = files[frame_idx]
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning(
                        "Failed to read image: %s (camera %s, frame %d)",
                        img_path,
                        cam_name,
                        frame_idx,
                    )
                    continue
                images[cam_name] = img

            yield frame_idx, images
