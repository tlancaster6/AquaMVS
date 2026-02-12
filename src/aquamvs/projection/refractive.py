"""Refractive projection model with Snell's law ray tracing."""

import torch


class RefractiveProjectionModel:
    """Refractive projection model for air-water interface.

    Implements the ProjectionModel protocol for the case of cameras in air
    viewing through a flat water surface. Ray casting traces through the
    pinhole model, intersects the water surface, and applies Snell's law.

    Args:
        K: Intrinsic matrix (post-undistortion), shape (3, 3), float32.
        R: Rotation matrix (world to camera), shape (3, 3), float32.
        t: Translation vector (world to camera), shape (3,), float32.
        water_z: Z-coordinate of the water surface in world frame (meters).
        normal: Interface normal vector, shape (3,), float32. Points from
            water toward air, typically [0, 0, -1].
        n_air: Refractive index of air (typically 1.0).
        n_water: Refractive index of water (typically 1.333).
    """

    def __init__(
        self,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        water_z: float,
        normal: torch.Tensor,
        n_air: float,
        n_water: float,
    ) -> None:
        # Store all parameters
        self.K = K
        self.R = R
        self.t = t
        self.water_z = water_z
        self.normal = normal
        self.n_air = n_air
        self.n_water = n_water

        # Precompute derived quantities
        self.K_inv = torch.linalg.inv(K)  # shape (3, 3)
        self.C = -R.T @ t  # camera center in world frame, shape (3,)
        self.n_ratio = n_air / n_water  # scalar float

    def cast_ray(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cast rays from pixel coordinates into the scene.

        For refractive models, rays originate at the water surface and
        point into the water (refracted direction). A 3D point at ray
        depth d is recovered as: point = origin + d * direction.

        Args:
            pixels: 2D pixel coordinates (u, v), shape (N, 2), float32.

        Returns:
            origins: Ray origin points, shape (N, 3), float32. These lie
                on the water surface (Z = water_z).
            directions: Unit ray direction vectors, shape (N, 3), float32.
                These point into the water (positive Z component).
        """
        N = pixels.shape[0]

        # Step 1: Pinhole back-projection (pixels to rays in camera frame)
        # Homogeneous pixel coords: (N, 3)
        ones = torch.ones(N, 1, device=pixels.device, dtype=pixels.dtype)
        pixels_h = torch.cat([pixels, ones], dim=-1)  # (N, 3)

        # Normalized camera coords: (N, 3)
        rays_cam = (self.K_inv @ pixels_h.T).T  # (N, 3)

        # Normalize to unit vectors
        rays_cam = rays_cam / torch.linalg.norm(rays_cam, dim=-1, keepdim=True)

        # Step 2: Transform to world frame
        # Camera-to-world rotation is R^T (since R is world-to-camera)
        rays_world = (self.R.T @ rays_cam.T).T  # (N, 3)

        # Step 3: Ray-plane intersection (camera center to water surface)
        # Parametric: point = C + t_param * rays_world
        # At intersection: point_z = water_z
        # So: C_z + t_param * rays_world_z = water_z
        # t_param = (water_z - C_z) / rays_world_z
        t_param = (self.water_z - self.C[2]) / rays_world[:, 2]  # (N,)
        origins = self.C.unsqueeze(0) + t_param.unsqueeze(-1) * rays_world  # (N, 3)

        # Step 4: Snell's law (vectorized)
        # Interface normal points water->air: [0, 0, -1]
        # For air-to-water rays (going +Z), cos_i = dot(ray, normal) < 0
        # We need n pointing into destination medium (water), so n = -normal = [0, 0, 1]

        # cos(theta_i) = |dot(incident, normal)|
        cos_i = -(rays_world * self.normal).sum(
            dim=-1
        )  # (N,) -- negate because dot is negative

        # Oriented normal pointing into water
        n_oriented = -self.normal.unsqueeze(0)  # (1, 3)

        # sin^2(theta_t) = n_ratio^2 * (1 - cos_i^2)
        sin_t_sq = self.n_ratio**2 * (1.0 - cos_i**2)  # (N,)

        # cos(theta_t) = sqrt(1 - sin_t_sq)
        cos_t = torch.sqrt(torch.clamp(1.0 - sin_t_sq, min=0.0))  # (N,)

        # Refracted direction: n_ratio * d + (cos_t - n_ratio * cos_i) * n_oriented
        directions = (
            self.n_ratio * rays_world
            + (cos_t - self.n_ratio * cos_i).unsqueeze(-1) * n_oriented
        )

        # Normalize to unit vectors
        directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)

        return origins, directions

    def project(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project 3D world points to 2D pixel coordinates.

        This method will be implemented in P.8.

        Args:
            points: 3D points in world frame, shape (N, 3), float32.

        Returns:
            pixels: 2D pixel coordinates (u, v), shape (N, 2), float32.
            valid: Boolean validity mask, shape (N,).

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        raise NotImplementedError("Forward projection is implemented in P.8")
