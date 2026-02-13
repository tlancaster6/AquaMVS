"""Sparse triangulation for matched features."""

from pathlib import Path

import torch

from .projection.protocol import ProjectionModel


def triangulate_rays(
    rays: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Find the 3D point minimizing sum of squared distances to all rays.

    Uses the closed-form linear least squares solution. For each ray
    (origin o_i, direction d_i), the distance from a point P to the ray is
    minimized by solving the normal equations.

    Args:
        rays: List of (origin, direction) tuples. Each origin is shape (3,),
            each direction is shape (3,) unit vector. Must have at least 2 rays.

    Returns:
        3D point in world frame, shape (3,), float32.

    Raises:
        ValueError: If fewer than 2 rays provided or system is degenerate
            (e.g., parallel rays).
    """
    if len(rays) < 2:
        raise ValueError("Need at least 2 rays")

    # Infer device and dtype from first ray
    device = rays[0][0].device
    dtype = rays[0][0].dtype

    A_sum = torch.zeros(3, 3, device=device, dtype=dtype)
    b_sum = torch.zeros(3, device=device, dtype=dtype)

    for origin, direction in rays:
        # Ensure unit direction
        d = direction / torch.linalg.norm(direction)
        I_minus_ddT = torch.eye(3, device=device, dtype=dtype) - torch.outer(d, d)
        A_sum += I_minus_ddT
        b_sum += I_minus_ddT @ origin

    # Solve A_sum @ P = b_sum
    try:
        P = torch.linalg.solve(A_sum, b_sum)
    except torch.linalg.LinAlgError:
        raise ValueError("Degenerate ray configuration")

    return P


def _triangulate_two_rays_batch(
    origins_a: torch.Tensor,
    dirs_a: torch.Tensor,
    origins_b: torch.Tensor,
    dirs_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized closest-approach triangulation for pairs of rays.

    For two rays, the normal equations simplify. Each ray contributes
    I - d*d^T to the 3x3 system. We sum two such matrices and solve.

    Args:
        origins_a: Ray origins for first set, shape (M, 3).
        dirs_a: Unit ray directions for first set, shape (M, 3).
        origins_b: Ray origins for second set, shape (M, 3).
        dirs_b: Unit ray directions for second set, shape (M, 3).

    Returns:
        points: Triangulated 3D points, shape (M, 3).
        valid: Boolean mask, shape (M,). False if system is degenerate
            (nearly parallel rays).
    """
    M = origins_a.shape[0]
    device = origins_a.device
    dtype = origins_a.dtype

    # Build normal equation matrices for each ray pair
    # For ray i: A_i = I - d_i @ d_i^T, b_i = A_i @ o_i
    # A_sum = A_a + A_b, b_sum = b_a + b_b

    # Outer products: (M, 3, 3)
    ddT_a = torch.einsum("mi,mj->mij", dirs_a, dirs_a)
    ddT_b = torch.einsum("mi,mj->mij", dirs_b, dirs_b)

    I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)  # (1, 3, 3)

    A_a = I - ddT_a  # (M, 3, 3)
    A_b = I - ddT_b  # (M, 3, 3)

    A_sum = A_a + A_b  # (M, 3, 3)

    b_a = torch.einsum("mij,mj->mi", A_a, origins_a)  # (M, 3)
    b_b = torch.einsum("mij,mj->mi", A_b, origins_b)  # (M, 3)
    b_sum = b_a + b_b  # (M, 3)

    # Solve batched system: A_sum @ P = b_sum
    # Detect degenerate cases via determinant
    det = torch.linalg.det(A_sum)  # (M,)
    valid = det.abs() > 1e-6

    # Initialize points as zeros
    points = torch.zeros(M, 3, device=device, dtype=dtype)

    # Solve only for valid entries
    if valid.any():
        try:
            points[valid] = torch.linalg.solve(A_sum[valid], b_sum[valid])
        except torch.linalg.LinAlgError:
            # Fallback: solve individually
            for i in range(M):
                if valid[i]:
                    try:
                        points[i] = torch.linalg.solve(A_sum[i], b_sum[i])
                    except torch.linalg.LinAlgError:
                        valid[i] = False

    return points, valid


def triangulate_pair(
    model_ref: ProjectionModel,
    model_src: ProjectionModel,
    matches: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Triangulate 3D points from matched features between two cameras.

    For each pair of matched keypoints, casts rays through both projection
    models and finds the point of closest approach.

    Args:
        model_ref: Projection model for the reference camera.
        model_src: Projection model for the source camera.
        matches: Matches dict from match_pair(). Keys:
            "ref_keypoints": (M, 2) pixel coords in reference image.
            "src_keypoints": (M, 2) pixel coords in source image.
            "scores": (M,) match confidence scores.

    Returns:
        Dict with keys:
            "points_3d": shape (M, 3), float32 -- triangulated world points
            "scores": shape (M,), float32 -- match confidence (passed through)
            "ref_pixels": shape (M, 2), float32 -- reference pixel coords
            "src_pixels": shape (M, 2), float32 -- source pixel coords
            "valid": shape (M,), bool -- True if triangulation succeeded
        Invalid entries in points_3d are zeros (masked by valid).
        Returns empty tensors if M=0.
    """
    ref_keypoints = matches["ref_keypoints"]
    src_keypoints = matches["src_keypoints"]
    scores = matches["scores"]

    M = ref_keypoints.shape[0]

    # Handle empty matches
    if M == 0:
        device = ref_keypoints.device
        dtype = ref_keypoints.dtype
        return {
            "points_3d": torch.empty(0, 3, device=device, dtype=dtype),
            "scores": torch.empty(0, device=device, dtype=dtype),
            "ref_pixels": torch.empty(0, 2, device=device, dtype=dtype),
            "src_pixels": torch.empty(0, 2, device=device, dtype=dtype),
            "valid": torch.empty(0, device=device, dtype=torch.bool),
        }

    # Cast rays for all keypoints in batch
    origins_ref, dirs_ref = model_ref.cast_ray(ref_keypoints)
    origins_src, dirs_src = model_src.cast_ray(src_keypoints)

    # Triangulate all pairs in batch
    points_3d, valid = _triangulate_two_rays_batch(
        origins_ref, dirs_ref, origins_src, dirs_src
    )

    return {
        "points_3d": points_3d,
        "scores": scores,
        "ref_pixels": ref_keypoints,
        "src_pixels": src_keypoints,
        "valid": valid,
    }


def triangulate_all_pairs(
    projection_models: dict[str, ProjectionModel],
    all_matches: dict[tuple[str, str], dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Triangulate 3D points from all matched camera pairs.

    Iterates over all pairs, triangulates each, and concatenates the results
    into a single sparse point cloud. Points that fail triangulation are
    excluded.

    Args:
        projection_models: Camera name to ProjectionModel mapping.
        all_matches: (ref_cam, src_cam) tuple to matches dict mapping
            (from match_all_pairs).

    Returns:
        Dict with keys:
            "points_3d": shape (N_total, 3), float32 -- all valid triangulated points
            "scores": shape (N_total,), float32 -- match confidence scores
        Where N_total is the sum of valid triangulated points across all pairs.
        Returns empty tensors if no valid points.
    """
    all_points = []
    all_scores = []

    for (ref_name, src_name), matches in all_matches.items():
        model_ref = projection_models[ref_name]
        model_src = projection_models[src_name]

        result = triangulate_pair(model_ref, model_src, matches)

        # Filter by validity mask
        valid_mask = result["valid"]
        if valid_mask.any():
            all_points.append(result["points_3d"][valid_mask])
            all_scores.append(result["scores"][valid_mask])

    # Concatenate all results
    if len(all_points) == 0:
        # No valid points - return empty tensors
        # Infer device from first projection model if possible
        device = "cpu"
        if len(projection_models) > 0:
            # Try to get device from any model (assuming they expose it)
            # For safety, just use cpu as default
            pass
        return {
            "points_3d": torch.empty(0, 3, dtype=torch.float32, device=device),
            "scores": torch.empty(0, dtype=torch.float32, device=device),
        }

    points_3d = torch.cat(all_points, dim=0)
    scores = torch.cat(all_scores, dim=0)

    return {
        "points_3d": points_3d,
        "scores": scores,
    }


def compute_depth_ranges(
    projection_models: dict[str, ProjectionModel],
    sparse_cloud: dict[str, torch.Tensor],
    margin: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Compute per-camera depth ranges from sparse triangulated points.

    For each camera, casts rays from the camera to each sparse 3D point,
    computes the ray depth (distance along the refracted ray from origin),
    and returns the [d_min - margin, d_max + margin] range.

    Args:
        projection_models: Camera name to ProjectionModel mapping.
        sparse_cloud: Sparse point cloud from triangulate_all_pairs().
            Must contain "points_3d" (N, 3).
        margin: Margin added to each side of the depth range (meters).
            Corresponds to DenseStereoConfig.depth_margin.

    Returns:
        Dict mapping camera name to (d_min, d_max) tuple.
        d_min and d_max are in ray-depth space (meters along the refracted ray).
        If a camera has no visible points, uses a fallback range.
    """
    points_3d = sparse_cloud["points_3d"]
    N = points_3d.shape[0]

    depth_ranges = {}

    # Fallback range if no points visible
    # From reference geometry: water depth ~0.530m, surface variation +/- 0.150m
    # Safe fallback: 0.3m to 0.9m ray depth
    FALLBACK_RANGE = (0.3, 0.9)

    if N == 0:
        # Empty cloud - return fallback for all cameras
        return {name: FALLBACK_RANGE for name in projection_models.keys()}

    for cam_name, model in projection_models.items():
        # Project all 3D points to this camera
        pixels, valid = model.project(points_3d)

        # Filter to only valid (visible) points
        if not valid.any():
            # No visible points for this camera
            depth_ranges[cam_name] = FALLBACK_RANGE
            continue

        valid_pixels = pixels[valid]
        valid_points = points_3d[valid]

        # Cast rays from the valid pixels
        origins, directions = model.cast_ray(valid_pixels)

        # Compute ray depth for each valid point
        # Since point = origin + d * direction and direction is unit:
        # d = dot(point - origin, direction)
        diff = valid_points - origins  # (K, 3)
        depths = (diff * directions).sum(dim=-1)  # (K,)

        # Compute range with margin
        d_min = float(depths.min().item()) - margin
        d_max = float(depths.max().item()) + margin

        # Clamp d_min to be >= 0 (ray depth cannot be negative)
        d_min = max(0.0, d_min)

        depth_ranges[cam_name] = (d_min, d_max)

    return depth_ranges


def save_sparse_cloud(cloud: dict[str, torch.Tensor], path: str | Path) -> None:
    """Save sparse point cloud to a .pt file."""
    torch.save(cloud, path)


def load_sparse_cloud(path: str | Path) -> dict[str, torch.Tensor]:
    """Load sparse point cloud from a .pt file."""
    return torch.load(path, weights_only=True)