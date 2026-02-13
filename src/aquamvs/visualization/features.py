"""Feature and match overlay rendering."""

from pathlib import Path

import cv2
import numpy as np


def render_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray | None = None,
    output_path: str | Path | None = None,
    marker_size: int = 3,
) -> np.ndarray:
    """Draw keypoint markers on an image.

    Args:
        image: BGR image, shape (H, W, 3), uint8.
        keypoints: Pixel coordinates, shape (N, 2), float32. (u, v) format.
        scores: Optional detection scores, shape (N,), float32 in [0, 1].
            Used to color markers (green = high, red = low). If None, all green.
        output_path: If given, save the result as PNG.
        marker_size: Radius of circle markers in pixels.

    Returns:
        Annotated image, shape (H, W, 3), uint8.
    """
    vis = image.copy()
    n = len(keypoints)
    if n == 0:
        if output_path is not None:
            cv2.imwrite(str(output_path), vis)
        return vis

    for i in range(n):
        u, v = int(round(keypoints[i, 0])), int(round(keypoints[i, 1]))
        if scores is not None:
            # Green (high score) to red (low score) via BGR
            s = float(scores[i])
            color = (0, int(255 * s), int(255 * (1 - s)))  # BGR
        else:
            color = (0, 255, 0)
        cv2.circle(vis, (u, v), marker_size, color, thickness=-1, lineType=cv2.LINE_AA)

    if output_path is not None:
        cv2.imwrite(str(output_path), vis)
    return vis


def render_matches(
    image_ref: np.ndarray,
    image_src: np.ndarray,
    ref_keypoints: np.ndarray,
    src_keypoints: np.ndarray,
    scores: np.ndarray | None = None,
    output_path: str | Path | None = None,
    line_thickness: int = 1,
    marker_size: int = 3,
) -> np.ndarray:
    """Draw match lines on a side-by-side image pair.

    Creates a horizontal concatenation of reference and source images,
    then draws lines connecting matched keypoints. Lines are colored
    by match confidence (green = high, red = low).

    Args:
        image_ref: Reference BGR image, shape (H, W, 3), uint8.
        image_src: Source BGR image, shape (H, W, 3), uint8.
        ref_keypoints: Reference pixel coords, shape (M, 2), float32.
        src_keypoints: Source pixel coords, shape (M, 2), float32.
        scores: Match confidence scores, shape (M,), float32 in [0, 1].
            If None, all lines are green.
        output_path: If given, save the result as PNG.
        line_thickness: Line width in pixels.
        marker_size: Keypoint circle radius in pixels.

    Returns:
        Side-by-side annotated image, shape (H, 2*W, 3), uint8.
    """
    h_ref, w_ref = image_ref.shape[:2]
    h_src, w_src = image_src.shape[:2]

    # Pad to same height if needed
    h = max(h_ref, h_src)
    canvas = np.zeros((h, w_ref + w_src, 3), dtype=np.uint8)
    canvas[:h_ref, :w_ref] = image_ref
    canvas[:h_src, w_ref:] = image_src

    m = len(ref_keypoints)
    for i in range(m):
        ru, rv = int(round(ref_keypoints[i, 0])), int(round(ref_keypoints[i, 1]))
        su, sv = int(round(src_keypoints[i, 0])) + w_ref, int(round(src_keypoints[i, 1]))

        if scores is not None:
            s = float(scores[i])
            color = (0, int(255 * s), int(255 * (1 - s)))
        else:
            color = (0, 255, 0)

        cv2.line(canvas, (ru, rv), (su, sv), color, line_thickness, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (ru, rv), marker_size, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (su, sv), marker_size, color, -1, lineType=cv2.LINE_AA)

    if output_path is not None:
        cv2.imwrite(str(output_path), canvas)
    return canvas


def render_sparse_overlay(
    image: np.ndarray,
    projected_points: np.ndarray,
    errors: np.ndarray | None = None,
    output_path: str | Path | None = None,
    marker_size: int = 5,
    error_threshold: float = 5.0,
) -> np.ndarray:
    """Draw reprojected sparse 3D points on a camera image.

    Args:
        image: BGR image, shape (H, W, 3), uint8.
        projected_points: Reprojected pixel coordinates, shape (N, 2), float32.
        errors: Optional reprojection errors in pixels, shape (N,), float32.
            Used to color markers (green = 0, red >= error_threshold).
            If None, all markers are cyan.
        output_path: If given, save the result as PNG.
        marker_size: Circle radius in pixels.
        error_threshold: Error value at which markers become fully red (pixels).

    Returns:
        Annotated image, shape (H, W, 3), uint8.
    """
    vis = image.copy()
    n = len(projected_points)
    if n == 0:
        if output_path is not None:
            cv2.imwrite(str(output_path), vis)
        return vis

    for i in range(n):
        u, v = int(round(projected_points[i, 0])), int(round(projected_points[i, 1]))

        if errors is not None:
            # Green (low error) to red (high error) via BGR
            # Clamp error to [0, error_threshold]
            e = float(errors[i])
            e_norm = min(e / error_threshold, 1.0)
            color = (0, int(255 * (1 - e_norm)), int(255 * e_norm))  # BGR
        else:
            # Cyan if no error information
            color = (255, 255, 0)  # BGR: cyan

        # Unfilled circle (thickness=2) to stand out
        cv2.circle(vis, (u, v), marker_size, color, thickness=2, lineType=cv2.LINE_AA)

    if output_path is not None:
        cv2.imwrite(str(output_path), vis)
    return vis


def render_all_features(
    images: dict[str, np.ndarray],
    all_features: dict[str, dict[str, np.ndarray]],
    all_matches: dict[tuple[str, str], dict[str, np.ndarray]],
    sparse_cloud: dict[str, np.ndarray] | None = None,
    projection_models: dict | None = None,
    output_dir: str | Path = ".",
) -> None:
    """Render keypoints, matches, and sparse overlays for all cameras.

    Saves to output_dir:
        - sparse_{cam}.png for each camera (keypoints overlay)
        - matches_{ref}_{src}.png for each matched pair
        - If sparse_cloud and projection_models provided:
          sparse_{cam}.png with reprojected 3D points

    Note:
        Sparse reprojection requires PyTorch and is deferred to the caller (P.31).
        If projection_models is provided, this function will skip sparse reprojection
        to maintain NumPy-only dependencies. The caller should use render_sparse_overlay
        directly after projecting points with PyTorch.

    Args:
        images: Camera name to BGR image (H, W, 3) uint8 mapping.
        all_features: Camera name to features dict mapping.
            Each has "keypoints" (N, 2) and "scores" (N,) as NumPy arrays.
        all_matches: (ref_name, src_name) to match dict mapping.
            Each has "ref_keypoints" (M, 2), "src_keypoints" (M, 2), "scores" (M,).
        sparse_cloud: Optional sparse cloud dict with "points_3d" (P, 3).
            Currently unused - kept for future API compatibility.
        projection_models: Optional camera name to ProjectionModel mapping.
            Currently unused - kept for future API compatibility.
        output_dir: Directory to save images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Render keypoints for each camera
    for cam_name, image in images.items():
        if cam_name not in all_features:
            continue

        features = all_features[cam_name]
        keypoints = features["keypoints"]
        scores = features.get("scores")

        output_path = output_dir / f"sparse_{cam_name}.png"
        render_keypoints(image, keypoints, scores, output_path)

    # Render matches for each pair
    for (ref_name, src_name), match_dict in all_matches.items():
        if ref_name not in images or src_name not in images:
            continue

        ref_image = images[ref_name]
        src_image = images[src_name]
        ref_kpts = match_dict["ref_keypoints"]
        src_kpts = match_dict["src_keypoints"]
        scores = match_dict.get("scores")

        output_path = output_dir / f"matches_{ref_name}_{src_name}.png"
        render_matches(ref_image, src_image, ref_kpts, src_kpts, scores, output_path)

    # Note: Sparse reprojection overlay rendering is skipped here to maintain
    # NumPy-only dependencies. The caller (P.31) should use render_sparse_overlay
    # directly after projecting 3D points to 2D with PyTorch projection models.
