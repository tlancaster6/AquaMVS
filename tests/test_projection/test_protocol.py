"""Tests for ProjectionModel protocol structural compliance."""

import torch

from aquamvs.projection import ProjectionModel


class _DummyProjectionModel:
    """Minimal implementation for protocol compliance testing."""

    def project(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = points.shape[0]
        return torch.zeros(n, 2), torch.ones(n, dtype=torch.bool)

    def cast_ray(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = pixels.shape[0]
        return torch.zeros(n, 3), torch.zeros(n, 3)


class _MissingProject:
    """Class missing the project method."""

    def cast_ray(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = pixels.shape[0]
        return torch.zeros(n, 3), torch.zeros(n, 3)


class _MissingCastRay:
    """Class missing the cast_ray method."""

    def project(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = points.shape[0]
        return torch.zeros(n, 2), torch.ones(n, dtype=torch.bool)


def test_protocol_compliance_positive():
    """Verify that a class with both methods passes isinstance check."""
    dummy = _DummyProjectionModel()
    assert isinstance(dummy, ProjectionModel)


def test_protocol_compliance_missing_project():
    """Verify that a class missing project fails isinstance check."""
    obj = _MissingProject()
    assert not isinstance(obj, ProjectionModel)


def test_protocol_compliance_missing_cast_ray():
    """Verify that a class missing cast_ray fails isinstance check."""
    obj = _MissingCastRay()
    assert not isinstance(obj, ProjectionModel)


def test_importability():
    """Verify that ProjectionModel can be imported from aquamvs.projection."""
    # Import already happened at top of file, but verify attributes exist
    assert hasattr(ProjectionModel, "project")
    assert hasattr(ProjectionModel, "cast_ray")
