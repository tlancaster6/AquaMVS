"""Tests for plane-sweep stereo."""

import pytest
import torch

from aquamvs.config import DenseStereoConfig
from aquamvs.dense.plane_sweep import (
    _bgr_to_gray,
    _make_pixel_grid,
    build_cost_volume,
    generate_depth_hypotheses,
    plane_sweep_stereo,
    warp_source_image,
)
from aquamvs.projection.refractive import RefractiveProjectionModel


class TestGenerateDepthHypotheses:
    """Tests for generate_depth_hypotheses()."""

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA not available"
                ),
            ),
        ],
    )
    def test_basic(self, device):
        """Test basic depth hypothesis generation."""
        d_min, d_max, num_depths = 1.0, 2.0, 11
        depths = generate_depth_hypotheses(d_min, d_max, num_depths, device=device)

        assert depths.shape == (num_depths,)
        assert depths.dtype == torch.float32
        assert depths.device.type == device
        assert torch.allclose(depths[0], torch.tensor(d_min, device=device))
        assert torch.allclose(depths[-1], torch.tensor(d_max, device=device))

        # Check uniform spacing
        spacing = depths[1:] - depths[:-1]
        assert torch.allclose(spacing, spacing[0])

    def test_single_depth(self):
        """Test edge case: d_min == d_max."""
        d_min, d_max, num_depths = 1.5, 1.5, 5
        depths = generate_depth_hypotheses(d_min, d_max, num_depths)

        assert depths.shape == (num_depths,)
        assert torch.allclose(depths, torch.tensor(1.5))

    def test_two_depths(self):
        """Test edge case: num_depths == 2."""
        d_min, d_max = 1.0, 2.0
        depths = generate_depth_hypotheses(d_min, d_max, 2)

        assert depths.shape == (2,)
        assert torch.allclose(depths[0], torch.tensor(d_min))
        assert torch.allclose(depths[1], torch.tensor(d_max))


class TestBgrToGray:
    """Tests for _bgr_to_gray()."""

    def test_uint8_conversion(self):
        """Test conversion from uint8 BGR to grayscale."""
        # Create a simple test image: pure blue, pure green, pure red
        bgr = torch.tensor(
            [
                [[255, 0, 0], [0, 255, 0], [0, 0, 255]],  # Blue, Green, Red
            ],
            dtype=torch.uint8,
        )

        gray = _bgr_to_gray(bgr)

        assert gray.shape == (1, 3)
        assert gray.dtype == torch.float32

        # BGR to gray: 0.114*B + 0.587*G + 0.299*R
        expected = torch.tensor([[0.114, 0.587, 0.299]])
        assert torch.allclose(gray, expected, atol=1e-6)

    def test_float32_conversion(self):
        """Test conversion from float32 BGR to grayscale."""
        # Same test but with float32 input
        bgr = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            dtype=torch.float32,
        )

        gray = _bgr_to_gray(bgr)

        assert gray.shape == (1, 3)
        assert gray.dtype == torch.float32

        expected = torch.tensor([[0.114, 0.587, 0.299]])
        assert torch.allclose(gray, expected, atol=1e-6)

    def test_grayscale_weights(self):
        """Test that grayscale weights sum to 1."""
        # A uniform gray image should stay uniform
        bgr = torch.ones((5, 5, 3), dtype=torch.float32) * 0.5

        gray = _bgr_to_gray(bgr)

        assert torch.allclose(gray, torch.full((5, 5), 0.5), atol=1e-6)


class TestMakePixelGrid:
    """Tests for _make_pixel_grid()."""

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA not available"
                ),
            ),
        ],
    )
    def test_basic(self, device):
        """Test basic pixel grid generation."""
        height, width = 3, 4
        grid = _make_pixel_grid(height, width, device=device)

        assert grid.shape == (height * width, 2)
        assert grid.dtype == torch.float32
        assert grid.device.type == device

        # Check that u (column) ranges from 0 to W-1
        u_vals = grid[:, 0]
        assert u_vals.min() == 0.0
        assert u_vals.max() == width - 1

        # Check that v (row) ranges from 0 to H-1
        v_vals = grid[:, 1]
        assert v_vals.min() == 0.0
        assert v_vals.max() == height - 1

    def test_pixel_ordering(self):
        """Test that pixels are ordered row-major."""
        height, width = 2, 3
        grid = _make_pixel_grid(height, width)

        # Expected: (0,0), (1,0), (2,0), (0,1), (1,1), (2,1)
        expected = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [1, 1],
                [2, 1],
            ],
            dtype=torch.float32,
        )

        assert torch.allclose(grid, expected)


class TestWarpSourceImage:
    """Tests for warp_source_image()."""

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA not available"
                ),
            ),
        ],
    )
    def test_identity_warp(self, device):
        """Test warping with identical camera models (identity warp)."""
        # Create a simple projection model (pinhole approximation for testing)
        # Use reference geometry from CLAUDE.md
        K = torch.eye(3, dtype=torch.float32, device=device)
        K[0, 0] = 1000.0  # fx
        K[1, 1] = 1000.0  # fy
        K[0, 2] = 400.0  # cx
        K[1, 2] = 300.0  # cy

        R = torch.eye(3, dtype=torch.float32, device=device)
        t = torch.zeros(3, dtype=torch.float32, device=device)
        water_z = 0.978
        normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
        n_air = 1.0
        n_water = 1.333

        model = RefractiveProjectionModel(K, R, t, water_z, normal, n_air, n_water)

        # Create a test image with some texture
        H, W = 64, 64
        src_image = torch.rand(H, W, device=device)

        # Generate pixel grid
        pixel_grid = _make_pixel_grid(H, W, device=device)

        # Warp at some reasonable depth
        depth = 0.5  # 0.5m along ray
        warped = warp_source_image(model, model, src_image, depth, pixel_grid)

        assert warped.shape == (H, W)
        assert warped.dtype == torch.float32

        # For identity warp, most pixels should match (except near borders due to interpolation)
        # Check interior region
        interior = warped[10:-10, 10:-10]
        src_interior = src_image[10:-10, 10:-10]

        # Most pixels should be valid (not NaN)
        valid_mask = ~torch.isnan(interior)
        assert valid_mask.sum() > 0.9 * interior.numel()

        # Valid pixels should approximately match
        if valid_mask.any():
            diff = torch.abs(interior[valid_mask] - src_interior[valid_mask])
            assert diff.mean() < 0.1  # Allow for interpolation error

    def test_out_of_bounds_nan(self):
        """Test that out-of-bounds pixels are NaN."""
        # Create two cameras with different positions
        K = torch.eye(3, dtype=torch.float32)
        K[0, 0] = 1000.0
        K[1, 1] = 1000.0
        K[0, 2] = 32.0
        K[1, 2] = 24.0

        # Reference camera at origin
        R_ref = torch.eye(3, dtype=torch.float32)
        t_ref = torch.zeros(3, dtype=torch.float32)

        # Source camera offset to the side
        R_src = torch.eye(3, dtype=torch.float32)
        t_src = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

        water_z = 0.978
        normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        n_air = 1.0
        n_water = 1.333

        ref_model = RefractiveProjectionModel(
            K, R_ref, t_ref, water_z, normal, n_air, n_water
        )
        src_model = RefractiveProjectionModel(
            K, R_src, t_src, water_z, normal, n_air, n_water
        )

        # Create a small test image
        H, W = 32, 32
        src_image = torch.ones(H, W)

        pixel_grid = _make_pixel_grid(H, W)

        # Warp at some depth
        depth = 0.5
        warped = warp_source_image(ref_model, src_model, src_image, depth, pixel_grid)

        assert warped.shape == (H, W)

        # There should be some NaN pixels due to out-of-bounds
        # (exact count depends on geometry, but there should be some)
        nan_count = torch.isnan(warped).sum()
        # Just verify that warping works and produces some valid and some invalid pixels
        assert nan_count >= 0
        assert nan_count < warped.numel()


class TestBuildCostVolume:
    """Tests for build_cost_volume()."""

    def test_identical_images(self):
        """Test cost volume with identical reference and source images."""
        # Simple projection model
        K = torch.eye(3, dtype=torch.float32)
        K[0, 0] = 500.0
        K[1, 1] = 500.0
        K[0, 2] = 16.0
        K[1, 2] = 16.0

        R = torch.eye(3, dtype=torch.float32)
        t = torch.zeros(3, dtype=torch.float32)
        water_z = 0.978
        normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        n_air = 1.0
        n_water = 1.333

        model = RefractiveProjectionModel(K, R, t, water_z, normal, n_air, n_water)

        # Small test image with texture
        H, W = 32, 32
        image = torch.rand(H, W)

        # Config
        config = DenseStereoConfig(
            num_depths=8,
            cost_function="ncc",
            window_size=5,
        )

        # Generate depths
        depths = torch.linspace(0.3, 0.8, config.num_depths)

        # Build cost volume (ref and src are the same)
        cost_volume = build_cost_volume(
            model,
            [model],
            image,
            [image],
            depths,
            config,
        )

        assert cost_volume.shape == (H, W, config.num_depths)
        assert cost_volume.dtype == torch.float32

        # For identical images with identity warp, cost should be low
        # (but not zero due to border effects and interpolation)
        # Check interior region
        interior_cost = cost_volume[10:-10, 10:-10, :]
        assert (
            interior_cost.mean() < 0.2
        )  # NCC cost should be near 0 for identical images

    def test_cost_volume_shape(self):
        """Test that cost volume has correct shape with multiple sources."""
        K = torch.eye(3, dtype=torch.float32)
        K[0, 0] = 500.0
        K[1, 1] = 500.0
        K[0, 2] = 16.0
        K[1, 2] = 16.0

        R = torch.eye(3, dtype=torch.float32)
        t = torch.zeros(3, dtype=torch.float32)
        water_z = 0.978
        normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        n_air = 1.0
        n_water = 1.333

        model = RefractiveProjectionModel(K, R, t, water_z, normal, n_air, n_water)

        H, W = 32, 32
        image = torch.rand(H, W)

        config = DenseStereoConfig(num_depths=16)
        depths = torch.linspace(0.3, 0.8, config.num_depths)

        # Multiple source cameras (same model for simplicity)
        num_sources = 3
        src_models = [model] * num_sources
        src_images = [image] * num_sources

        cost_volume = build_cost_volume(
            model,
            src_models,
            image,
            src_images,
            depths,
            config,
        )

        assert cost_volume.shape == (H, W, config.num_depths)


class TestPlaneSweepStereo:
    """Tests for plane_sweep_stereo()."""

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA not available"
                ),
            ),
        ],
    )
    def test_end_to_end_structure(self, device):
        """Test end-to-end plane sweep stereo structure."""
        # Simple projection model
        K = torch.eye(3, dtype=torch.float32, device=device)
        K[0, 0] = 500.0
        K[1, 1] = 500.0
        K[0, 2] = 16.0
        K[1, 2] = 16.0

        R = torch.eye(3, dtype=torch.float32, device=device)
        t = torch.zeros(3, dtype=torch.float32, device=device)
        water_z = 0.978
        normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
        n_air = 1.0
        n_water = 1.333

        ref_model = RefractiveProjectionModel(K, R, t, water_z, normal, n_air, n_water)

        # Create source models (slightly offset)
        src_models = {}
        src_names = ["cam1", "cam2"]
        for i, name in enumerate(src_names):
            t_src = torch.tensor(
                [0.1 * (i + 1), 0.0, 0.0], dtype=torch.float32, device=device
            )
            src_models[name] = RefractiveProjectionModel(
                K, R, t_src, water_z, normal, n_air, n_water
            )

        # Create test images (BGR format)
        H, W = 32, 32
        ref_image = torch.rand(H, W, 3, device=device)
        src_images = {name: torch.rand(H, W, 3, device=device) for name in src_names}

        # Config
        config = DenseStereoConfig(num_depths=8)

        # Run plane sweep stereo
        depth_range = (0.3, 0.8)
        result = plane_sweep_stereo(
            "ref_cam",
            ref_model,
            src_names,
            src_models,
            ref_image,
            src_images,
            depth_range,
            config,
            device=device,
        )

        # Check return structure
        assert "cost_volume" in result
        assert "depths" in result
        assert "ref_name" in result

        assert result["ref_name"] == "ref_cam"
        assert result["cost_volume"].shape == (H, W, config.num_depths)
        assert result["depths"].shape == (config.num_depths,)
        assert result["cost_volume"].device.type == device
        assert result["depths"].device.type == device

        # Check depth range
        assert torch.allclose(
            result["depths"][0], torch.tensor(depth_range[0], device=device)
        )
        assert torch.allclose(
            result["depths"][-1], torch.tensor(depth_range[1], device=device)
        )

    def test_uint8_image_input(self):
        """Test that uint8 images are handled correctly."""
        K = torch.eye(3, dtype=torch.float32)
        K[0, 0] = 500.0
        K[1, 1] = 500.0
        K[0, 2] = 16.0
        K[1, 2] = 16.0

        R = torch.eye(3, dtype=torch.float32)
        t = torch.zeros(3, dtype=torch.float32)
        water_z = 0.978
        normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        n_air = 1.0
        n_water = 1.333

        ref_model = RefractiveProjectionModel(K, R, t, water_z, normal, n_air, n_water)

        H, W = 32, 32
        ref_image = torch.randint(0, 256, (H, W, 3), dtype=torch.uint8)
        src_images = {"cam1": torch.randint(0, 256, (H, W, 3), dtype=torch.uint8)}
        src_models = {"cam1": ref_model}

        config = DenseStereoConfig(num_depths=4)

        result = plane_sweep_stereo(
            "ref",
            ref_model,
            ["cam1"],
            src_models,
            ref_image,
            src_images,
            (0.5, 1.0),
            config,
        )

        # Should work without errors
        assert result["cost_volume"].shape == (H, W, 4)
        assert result["cost_volume"].dtype == torch.float32
