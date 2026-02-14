"""Best-view color selection and cross-camera color normalization."""

import numpy as np
import torch

from .projection.protocol import ProjectionModel


def best_view_colors(
    points: np.ndarray,
    normals: np.ndarray,
    projection_models: dict[str, ProjectionModel],
    images: dict[str, torch.Tensor],
    camera_centers: dict[str, np.ndarray],
) -> np.ndarray:
    """Assign colors to 3D points using the best viewing angle per point.

    For each point, projects into all cameras, computes the alignment between
    the viewing direction and the surface normal, and picks the camera with
    the highest alignment score (most perpendicular view).

    The viewing direction is the straight-line direction from camera center to
    point. This is an approximation that ignores refraction at the water surface,
    but is accurate enough for camera selection since all cameras look roughly
    downward and refraction angles are small.

    Args:
        points: 3D points in world frame, shape (N, 3), float64.
        normals: Surface normals at each point, shape (N, 3), float64.
        projection_models: Dict of ProjectionModel by camera name.
        images: Dict of undistorted BGR images (H, W, 3) uint8 tensors by camera name.
        camera_centers: Dict of camera centers in world frame, shape (3,) float64 per camera.

    Returns:
        RGB colors in [0, 1], shape (N, 3), float64. Points with no valid camera
        projection receive default gray (0.5, 0.5, 0.5).
    """
    N = points.shape[0]

    if N == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # Initialize: best_score = -1 for all points, colors = gray (0.5)
    best_scores = np.full(N, -1.0, dtype=np.float64)
    colors = np.full((N, 3), 0.5, dtype=np.float64)

    # Convert points and normals to torch tensors
    # Infer device from first projection model (all models share the same device)
    first_model = next(iter(projection_models.values()))
    device = first_model.K.device
    points_torch = torch.from_numpy(points).float().to(device)
    normals_torch = torch.from_numpy(normals).float().to(device)

    # For each camera, project all points and update best colors
    for cam_name, model in projection_models.items():
        if cam_name not in images or cam_name not in camera_centers:
            continue

        image = images[cam_name]  # (H, W, 3) uint8
        H, W = image.shape[:2]
        cam_center = camera_centers[cam_name]  # (3,) float64

        pixels, valid = model.project(points_torch)  # (N, 2), (N,)

        # Compute viewing direction: point - cam_center
        cam_center_torch = torch.from_numpy(cam_center).float().to(device)
        view_dir = points_torch - cam_center_torch  # (N, 3)
        view_dir = view_dir / (torch.linalg.norm(view_dir, dim=-1, keepdim=True) + 1e-8)

        # Compute alignment score: |dot(view_dir, normal)|
        # Higher score = more perpendicular to surface
        score = torch.abs((view_dir * normals_torch).sum(dim=-1))  # (N,)

        # Convert to numpy for comparison and indexing
        score_np = score.cpu().numpy()
        valid_np = valid.cpu().numpy()
        pixels_np = pixels.cpu().numpy()

        # Check image bounds: reject pixels outside [0, W) x [0, H)
        in_bounds = (
            (pixels_np[:, 0] >= 0) & (pixels_np[:, 0] < W) &
            (pixels_np[:, 1] >= 0) & (pixels_np[:, 1] < H)
        )

        # Find points where this camera is better than current best
        better = valid_np & in_bounds & (score_np > best_scores)

        # For better points: sample image color and update
        if better.sum() > 0:
            better_indices = np.where(better)[0]
            for i in better_indices:
                u, v = pixels_np[i]
                # Convert to integer indices with bounds checking
                v_int = int(np.clip(np.round(v), 0, H - 1))
                u_int = int(np.clip(np.round(u), 0, W - 1))

                # Sample BGR pixel and convert to RGB [0, 1]
                bgr = image[v_int, u_int].cpu().numpy()
                rgb = bgr[[2, 1, 0]] / 255.0

                # Update color and best score
                colors[i] = rgb
                best_scores[i] = score_np[i]

    return colors


def normalize_colors(
    images: dict[str, np.ndarray],
    method: str = "gain",
) -> dict[str, np.ndarray]:
    """Normalize colors across cameras to reduce white balance and exposure differences.

    Args:
        images: Camera name to BGR image (H, W, 3) uint8 mapping.
        method: Normalization method ("gain" or "histogram").

    Returns:
        Normalized images, same format as input. Original dict is not modified.

    Raises:
        ValueError: If method is not "gain" or "histogram".
    """
    if method not in ["gain", "histogram"]:
        raise ValueError(
            f"Invalid normalization method: {method!r}. "
            "Must be 'gain' or 'histogram'."
        )

    if not images:
        return {}

    if method == "gain":
        return _normalize_colors_gain(images)
    else:  # histogram
        return _normalize_colors_histogram(images)


def _normalize_colors_gain(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Normalize colors using per-channel multiplicative gain.

    Args:
        images: Camera name to BGR image (H, W, 3) uint8 mapping.

    Returns:
        Normalized images with per-channel gains applied.
    """
    # 1. Compute per-channel mean for each camera
    means = {name: img.mean(axis=(0, 1)).astype(float) for name, img in images.items()}

    # 2. Compute global target mean (average across cameras)
    target_mean = np.mean(list(means.values()), axis=0)  # (3,) BGR

    # 3. Apply gain to each camera
    result = {}
    for name, img in images.items():
        gain = target_mean / (means[name] + 1e-8)  # (3,) per-channel gain
        normalized = np.clip(img.astype(float) * gain, 0, 255).astype(np.uint8)
        result[name] = normalized

    return result


def _normalize_colors_histogram(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Normalize colors using per-channel histogram matching.

    For each channel, match each camera's CDF to the average CDF across all cameras.

    Args:
        images: Camera name to BGR image (H, W, 3) uint8 mapping.

    Returns:
        Normalized images with per-channel histogram matching applied.
    """
    num_cameras = len(images)
    num_bins = 256

    # 1. Compute per-channel histograms for all cameras
    histograms = {}
    for name, img in images.items():
        histograms[name] = np.zeros((3, num_bins), dtype=np.float64)
        for c in range(3):
            histograms[name][c] = np.histogram(
                img[:, :, c], bins=num_bins, range=(0, 256)
            )[0]

    # 2. Compute average histogram (mean across cameras)
    avg_histogram = np.mean(list(histograms.values()), axis=0)  # (3, 256)

    # 3. Compute CDF of average histogram -> reference CDF
    ref_cdf = np.zeros((3, num_bins), dtype=np.float64)
    for c in range(3):
        ref_cdf[c] = np.cumsum(avg_histogram[c])
        ref_cdf[c] = ref_cdf[c] / (ref_cdf[c][-1] + 1e-8)  # Normalize to [0, 1]

    # 4. Build LUT for each camera and channel
    result = {}
    for name, img in images.items():
        # Compute source CDF
        src_cdf = np.zeros((3, num_bins), dtype=np.float64)
        for c in range(3):
            src_hist = np.histogram(img[:, :, c], bins=num_bins, range=(0, 256))[0]
            src_cdf[c] = np.cumsum(src_hist)
            src_cdf[c] = src_cdf[c] / (src_cdf[c][-1] + 1e-8)  # Normalize to [0, 1]

        # Build LUT: for each intensity i, find j where ref_CDF[j] is closest to src_CDF[i]
        luts = np.zeros((3, num_bins), dtype=np.uint8)
        for c in range(3):
            for i in range(num_bins):
                # Find the closest ref_CDF value
                idx = np.argmin(np.abs(ref_cdf[c] - src_cdf[c][i]))
                luts[c, i] = idx

        # Apply LUT to each channel
        normalized = np.zeros_like(img)
        for c in range(3):
            normalized[:, :, c] = luts[c, img[:, :, c]]

        result[name] = normalized

    return result
