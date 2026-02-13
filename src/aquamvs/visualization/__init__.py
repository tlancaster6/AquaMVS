"""Visualization outputs for reconstruction results."""

from .depth import (
    render_all_depth_maps,
    render_confidence_map,
    render_depth_map,
)
from .features import (
    render_all_features,
    render_keypoints,
    render_matches,
    render_sparse_overlay,
)
from .rig import render_rig_diagram
from .scene import (
    compute_canonical_viewpoints,
    render_all_scenes,
    render_geometry,
    render_scene,
)
from .summary import (
    render_distance_map,
    render_error_histogram,
    render_evaluation_summary,
    render_timeseries_gallery,
)

__all__ = [
    "compute_canonical_viewpoints",
    "render_all_depth_maps",
    "render_all_features",
    "render_all_scenes",
    "render_confidence_map",
    "render_depth_map",
    "render_distance_map",
    "render_error_histogram",
    "render_evaluation_summary",
    "render_geometry",
    "render_keypoints",
    "render_matches",
    "render_rig_diagram",
    "render_scene",
    "render_sparse_overlay",
    "render_timeseries_gallery",
]
