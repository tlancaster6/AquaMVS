"""Protocol definition for projection models."""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class ProjectionModel(Protocol):
    """Protocol for geometric projection models.

    Defines the interface for mapping between 3D world points and 2D pixel
    coordinates. Implementations handle different geometric models (refractive,
    pinhole, etc.) while maintaining a common interface for the pipeline.

    Both methods are batched (N points/pixels in, N results out),
    differentiable (support PyTorch autograd), and device-agnostic
    (output tensors are on the same device as input tensors).
    """

    def project(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project 3D world points to 2D pixel coordinates.

        Args:
            points: 3D points in world frame, shape (N, 3), float32.

        Returns:
            pixels: 2D pixel coordinates (u, v), shape (N, 2), float32.
            valid: Boolean validity mask, shape (N,). False for points that
                cannot be projected (behind camera, total internal reflection,
                above water surface, etc.). Invalid entries in pixels are
                undefined (may be NaN or arbitrary values).
        """
        ...

    def cast_ray(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cast rays from pixel coordinates into the scene.

        For refractive models, rays originate at the water surface and
        point into the water (refracted direction). A 3D point at ray
        depth d is recovered as: point = origin + d * direction.

        Args:
            pixels: 2D pixel coordinates (u, v), shape (N, 2), float32.

        Returns:
            origins: Ray origin points, shape (N, 3), float32. For the
                refractive model, these lie on the water surface (Z = water_z).
            directions: Unit ray direction vectors, shape (N, 3), float32.
                For the refractive model, these point into the water
                (positive Z component).
        """
        ...
