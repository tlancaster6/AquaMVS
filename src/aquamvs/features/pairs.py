"""Camera pair selection for feature matching."""

import torch

from ..config import PairSelectionConfig


def select_pairs(
    camera_positions: dict[str, torch.Tensor],
    ring_cameras: list[str],
    auxiliary_cameras: list[str],
    config: PairSelectionConfig,
) -> dict[str, list[str]]:
    """Select source cameras for each reference camera.

    For each ring camera, selects the N closest ring cameras by Euclidean
    distance between camera centers. Optionally includes the center
    (auxiliary) camera as an additional source.

    Args:
        camera_positions: Camera name to world-frame center position (3,) tensor.
        ring_cameras: List of ring (non-auxiliary) camera names.
        auxiliary_cameras: List of auxiliary camera names.
        config: Pair selection configuration.

    Returns:
        Dict mapping each reference camera name to its list of source camera
        names. Only ring cameras appear as keys. Source lists are ordered by
        distance (nearest first), with auxiliary cameras appended at the end.
    """
    pairs = {}

    for ref_cam in ring_cameras:
        ref_pos = camera_positions[ref_cam]

        # Compute distances to all other ring cameras
        distances = []
        for src_cam in ring_cameras:
            if src_cam == ref_cam:
                # A camera is never paired with itself
                continue

            src_pos = camera_positions[src_cam]
            distance = torch.linalg.norm(ref_pos - src_pos).item()
            distances.append((distance, src_cam))

        # Sort by distance (ascending), then by name for deterministic tiebreaking
        distances.sort(key=lambda x: (x[0], x[1]))

        # Take the first num_neighbors cameras
        num_neighbors = min(config.num_neighbors, len(distances))
        sources = [name for _, name in distances[:num_neighbors]]

        # Optionally append auxiliary cameras
        if config.include_center:
            sources.extend(auxiliary_cameras)

        pairs[ref_cam] = sources

    return pairs
