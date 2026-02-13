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

    def to(self, device: str | torch.device) -> "RefractiveProjectionModel":
        """Move all internal tensors to the specified device.

        Args:
            device: Target device (e.g., "cpu", "cuda", "cuda:0").

        Returns:
            Self, for method chaining.
        """
        self.K = self.K.to(device)
        self.K_inv = self.K_inv.to(device)
        self.R = self.R.to(device)
        self.t = self.t.to(device)
        self.C = self.C.to(device)
        self.normal = self.normal.to(device)
        return self

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

        Uses Newton-Raphson iteration to find the water surface intersection
        point that satisfies Snell's law, then projects through the pinhole
        camera model.

        Args:
            points: 3D points in world frame, shape (N, 3), float32.
                Points should be underwater (Z > water_z).

        Returns:
            pixels: 2D pixel coordinates (u, v), shape (N, 2), float32.
                Invalid pixels are NaN.
            valid: Boolean validity mask, shape (N,). False for points above
                water surface (Z <= water_z) or when interface point is behind
                camera.
        """
        N = points.shape[0]
        device = points.device
        dtype = points.dtype

        Q = points  # (N, 3)
        C = self.C  # (3,)

        # Vertical distances
        h_c = self.water_z - C[2]  # scalar
        h_q = Q[:, 2] - self.water_z  # (N,)

        # Horizontal displacement and distance
        dx = Q[:, 0] - C[0]  # (N,)
        dy = Q[:, 1] - C[1]  # (N,)
        r_q = torch.sqrt(
            dx * dx + dy * dy + 1e-12
        )  # (N,) -- epsilon avoids div-by-zero grad

        # Direction unit vector in XY plane (camera toward point)
        dir_x = dx / r_q  # (N,)
        dir_y = dy / r_q  # (N,)

        # Initial guess: pinhole (straight-line) intersection
        r_p = r_q * h_c / (h_c + h_q + 1e-12)  # (N,)

        # Newton-Raphson iterations (fixed count for differentiability)
        for _ in range(10):
            d_air_sq = r_p * r_p + h_c * h_c
            d_air = torch.sqrt(d_air_sq)

            r_diff = r_q - r_p
            d_water_sq = r_diff * r_diff + h_q * h_q
            d_water = torch.sqrt(d_water_sq)

            sin_air = r_p / d_air
            sin_water = r_diff / d_water

            f = self.n_air * sin_air - self.n_water * sin_water

            f_prime = self.n_air * h_c * h_c / (
                d_air_sq * d_air
            ) + self.n_water * h_q * h_q / (d_water_sq * d_water)

            r_p = r_p - f / (f_prime + 1e-12)

            # Clamp to valid range [0, r_q] -- use torch.clamp, not in-place
            r_p = torch.clamp(r_p, min=0.0)
            r_p = torch.minimum(r_p, r_q)

        # Compute interface point P
        px = C[0] + r_p * dir_x  # (N,)
        py = C[1] + r_p * dir_y  # (N,)
        pz = torch.full_like(px, self.water_z)  # (N,)
        P = torch.stack([px, py, pz], dim=-1)  # (N, 3)

        # Project P through pinhole model: p_cam = R @ P_world + t
        p_cam = (self.R @ P.T).T + self.t.unsqueeze(0)  # (N, 3)

        # Perspective division + intrinsics
        p_norm = p_cam[:, :2] / p_cam[:, 2:3]  # (N, 2)
        pixels = (self.K[:2, :2] @ p_norm.T).T + self.K[:2, 2].unsqueeze(0)  # (N, 2)

        # Validity mask
        valid = (h_q > 0) & (
            p_cam[:, 2] > 0
        )  # point below water & interface in front of camera

        # Set invalid pixels to NaN
        pixels = torch.where(
            valid.unsqueeze(-1),
            pixels,
            torch.tensor(float("nan"), device=device, dtype=dtype),
        )

        return pixels, valid
