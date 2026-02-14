"""Dense stereo reconstruction via plane sweep."""

from .cost import aggregate_costs, compute_cost, compute_ncc, compute_ssim
from .plane_sweep import (
    build_cost_volume,
    extract_depth,
    generate_depth_hypotheses,
    load_depth_map,
    plane_sweep_stereo,
    save_depth_map,
    warp_source_image,
)
from .roma_depth import (
    aggregate_pairwise_depths,
    roma_warps_to_depth_maps,
    warp_to_pairwise_depth,
)

__all__ = [
    "compute_ncc",
    "compute_ssim",
    "compute_cost",
    "aggregate_costs",
    "generate_depth_hypotheses",
    "warp_source_image",
    "build_cost_volume",
    "plane_sweep_stereo",
    "extract_depth",
    "save_depth_map",
    "load_depth_map",
    "warp_to_pairwise_depth",
    "aggregate_pairwise_depths",
    "roma_warps_to_depth_maps",
]
