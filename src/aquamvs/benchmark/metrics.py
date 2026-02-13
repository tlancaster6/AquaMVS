"""Benchmark metrics and result data structures."""

from dataclasses import dataclass


@dataclass
class ConfigResult:
    """Results from one configuration in the benchmark sweep."""

    config_name: str  # e.g., "superpoint_clahe_on"
    extractor_type: str
    clahe_enabled: bool

    # Per-camera keypoint counts and mean scores
    keypoint_counts: dict[str, int]
    keypoint_mean_scores: dict[str, float]

    # Per-pair match counts
    match_counts: dict[tuple[str, str], int]

    # Sparse cloud stats
    sparse_point_count: int

    # Timing (seconds)
    extraction_time: float
    matching_time: float
    triangulation_time: float
    total_time: float


@dataclass
class BenchmarkResults:
    """Aggregated results from the full benchmark sweep."""

    results: list[ConfigResult]
    frame_idx: int
    camera_names: list[str]  # ring + auxiliary
    pair_keys: list[tuple[str, str]]


def config_name(extractor_type: str, clahe_enabled: bool) -> str:
    """Generate a human-readable config name like 'superpoint_clahe_on'.

    Args:
        extractor_type: Feature extractor backend name.
        clahe_enabled: Whether CLAHE preprocessing is enabled.

    Returns:
        Config name string.
    """
    clahe_suffix = "clahe_on" if clahe_enabled else "clahe_off"
    return f"{extractor_type}_{clahe_suffix}"


def total_keypoints(result: ConfigResult) -> int:
    """Sum keypoints across all cameras.

    Args:
        result: Configuration result to aggregate.

    Returns:
        Total keypoint count.
    """
    return sum(result.keypoint_counts.values())


def total_matches(result: ConfigResult) -> int:
    """Sum matches across all pairs.

    Args:
        result: Configuration result to aggregate.

    Returns:
        Total match count.
    """
    return sum(result.match_counts.values())


__all__ = [
    "ConfigResult",
    "BenchmarkResults",
    "config_name",
    "total_keypoints",
    "total_matches",
]
