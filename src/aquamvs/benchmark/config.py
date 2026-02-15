"""Configuration models for benchmark testing."""

import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

logger = logging.getLogger(__name__)


class BenchmarkDataset(BaseModel):
    """Configuration for a benchmark dataset.

    Attributes:
        name: Human-readable dataset name.
        type: Dataset type (charuco, synthetic_plane, or synthetic_surface).
        path: Path to dataset directory or config file.
        ground_truth_tolerance_mm: Tolerance for accurate completeness metric (None = skip).
    """

    model_config = ConfigDict(extra="allow")

    name: str
    type: Literal["charuco", "synthetic_plane", "synthetic_surface"]
    path: str
    ground_truth_tolerance_mm: float | None = None

    @model_validator(mode="after")
    def warn_extra_fields(self) -> "BenchmarkDataset":
        """Warn about unknown configuration keys."""
        if self.__pydantic_extra__:
            logger.warning(
                "Unknown config keys in BenchmarkDataset '%s' (ignored): %s",
                self.name,
                list(self.__pydantic_extra__.keys()),
            )
        return self


class BenchmarkTests(BaseModel):
    """Configuration for which benchmark tests to run.

    Attributes:
        clahe_comparison: Compare CLAHE on vs off.
        execution_mode_comparison: Compare sparse vs full reconstruction modes.
        surface_reconstruction_comparison: Compare surface reconstruction methods.
    """

    model_config = ConfigDict(extra="allow")

    clahe_comparison: bool = True
    execution_mode_comparison: bool = True
    surface_reconstruction_comparison: bool = True

    @model_validator(mode="after")
    def warn_extra_fields(self) -> "BenchmarkTests":
        """Warn about unknown configuration keys."""
        if self.__pydantic_extra__:
            logger.warning(
                "Unknown config keys in BenchmarkTests (ignored): %s",
                list(self.__pydantic_extra__.keys()),
            )
        return self


class BenchmarkConfig(BaseModel):
    """Configuration for benchmark testing.

    Attributes:
        output_dir: Directory for benchmark results and reports.
        datasets: List of datasets to test against.
        tests: Which benchmark tests to run.
        regression_thresholds: Per-metric percentage thresholds for regression detection.
        frames: Number of frames to process per dataset.
        lightglue_extractor: Feature extractor to use with LightGlue matcher.
    """

    model_config = ConfigDict(extra="allow")

    output_dir: str
    datasets: list[BenchmarkDataset] = Field(default_factory=list)
    tests: BenchmarkTests = Field(default_factory=BenchmarkTests)
    regression_thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "mean_error_mm": 5.0,
            "median_error_mm": 5.0,
            "raw_completeness_pct": 5.0,
            "accurate_completeness_pct": 5.0,
            "extraction_time_s": 10.0,
            "matching_time_s": 10.0,
            "reconstruction_time_s": 10.0,
        }
    )
    frames: int = 1
    lightglue_extractor: Literal["superpoint", "aliked", "disk"] = "superpoint"

    @model_validator(mode="after")
    def warn_extra_fields(self) -> "BenchmarkConfig":
        """Warn about unknown configuration keys."""
        if self.__pydantic_extra__:
            logger.warning(
                "Unknown config keys in BenchmarkConfig (ignored): %s",
                list(self.__pydantic_extra__.keys()),
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchmarkConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Loaded configuration with defaults filled in.

        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If the file is not valid YAML.
            ValueError: If validation fails.
        """
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        try:
            config = cls.model_validate(data)
        except ValidationError as e:
            # Format validation errors with YAML paths
            formatted_errors = _format_validation_errors(e)
            raise ValueError(
                f"Benchmark configuration validation failed:\n{formatted_errors}"
            ) from None

        return config

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to output YAML file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict using Pydantic v2 model_dump
        data = self.model_dump()

        with open(path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def _format_validation_errors(error: ValidationError) -> str:
    """Format Pydantic validation errors with YAML-style paths.

    Args:
        error: Pydantic ValidationError.

    Returns:
        Formatted error string with YAML paths and messages.
    """
    lines = []
    for err in error.errors():
        # Build YAML path from location tuple
        loc = err["loc"]
        path_parts = []
        for part in loc:
            if isinstance(part, int):
                path_parts.append(f"[{part}]")
            else:
                if path_parts:
                    path_parts.append(f".{part}")
                else:
                    path_parts.append(str(part))
        yaml_path = "".join(path_parts)

        # Format error message
        msg = err["msg"]
        lines.append(f"  - {yaml_path}: {msg}")

    return "\n".join(lines)


__all__ = [
    "BenchmarkConfig",
    "BenchmarkDataset",
    "BenchmarkTests",
]
