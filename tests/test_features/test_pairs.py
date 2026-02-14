"""Tests for camera pair selection."""

import math

import torch

from aquamvs.config import PairSelectionConfig
from aquamvs.features import select_pairs


class TestSelectPairs:
    """Tests for select_pairs() function."""

    def test_ring_of_12_cameras_4_neighbors(self):
        """Test pair selection with 12 cameras on a ring, num_neighbors=4.

        Each camera should select its 2 nearest neighbors on each side (4 total).
        """
        # Generate 12 cameras evenly spaced on a circle of radius 0.635m
        ring_cameras = [f"cam{i:02d}" for i in range(12)]
        positions = {}
        for i, name in enumerate(ring_cameras):
            angle = 2 * math.pi * i / 12
            positions[name] = torch.tensor(
                [0.635 * math.cos(angle), 0.635 * math.sin(angle), 0.0]
            )

        config = PairSelectionConfig(num_neighbors=4, include_center=False)
        pairs = select_pairs(positions, ring_cameras, [], config)

        # Verify all ring cameras are references
        assert set(pairs.keys()) == set(ring_cameras)

        # Verify each camera has exactly 4 sources
        for sources in pairs.values():
            assert len(sources) == 4

        # For cam00 at angle 0, nearest neighbors should be:
        # cam11 (angle -30°), cam01 (angle +30°), cam10 (angle -60°), cam02 (angle +60°)
        cam00_sources = pairs["cam00"]
        assert len(cam00_sources) == 4
        # Check that the nearest 4 are selected (by index modulo 12)
        expected_nearest = {"cam01", "cam02", "cam10", "cam11"}
        assert set(cam00_sources) == expected_nearest

    def test_include_center_true(self):
        """Test that auxiliary camera is included when include_center=True."""
        # Simple 4-camera ring
        ring_cameras = ["ring_a", "ring_b", "ring_c", "ring_d"]
        positions = {
            "ring_a": torch.tensor([1.0, 0.0, 0.0]),
            "ring_b": torch.tensor([0.0, 1.0, 0.0]),
            "ring_c": torch.tensor([-1.0, 0.0, 0.0]),
            "ring_d": torch.tensor([0.0, -1.0, 0.0]),
            "center": torch.tensor([0.0, 0.0, 0.0]),
        }
        auxiliary_cameras = ["center"]

        config = PairSelectionConfig(num_neighbors=2, include_center=True)
        pairs = select_pairs(positions, ring_cameras, auxiliary_cameras, config)

        # Verify center camera is in every source list (at the end)
        for sources in pairs.values():
            assert "center" in sources
            # Center should be at the end
            assert sources[-1] == "center"
            # Each ring camera should have 2 ring neighbors + 1 auxiliary = 3 total
            assert len(sources) == 3

    def test_include_center_false(self):
        """Test that auxiliary camera is excluded when include_center=False."""
        ring_cameras = ["ring_a", "ring_b", "ring_c"]
        positions = {
            "ring_a": torch.tensor([1.0, 0.0, 0.0]),
            "ring_b": torch.tensor([0.0, 1.0, 0.0]),
            "ring_c": torch.tensor([-1.0, 0.0, 0.0]),
            "center": torch.tensor([0.0, 0.0, 0.0]),
        }
        auxiliary_cameras = ["center"]

        config = PairSelectionConfig(num_neighbors=2, include_center=False)
        pairs = select_pairs(positions, ring_cameras, auxiliary_cameras, config)

        # Verify center camera is not in any source list
        for sources in pairs.values():
            assert "center" not in sources

    def test_auxiliary_never_reference(self):
        """Test that auxiliary cameras never appear as reference (keys)."""
        ring_cameras = ["ring_a", "ring_b"]
        positions = {
            "ring_a": torch.tensor([1.0, 0.0, 0.0]),
            "ring_b": torch.tensor([-1.0, 0.0, 0.0]),
            "center": torch.tensor([0.0, 0.0, 0.0]),
        }
        auxiliary_cameras = ["center"]

        config = PairSelectionConfig(num_neighbors=1, include_center=True)
        pairs = select_pairs(positions, ring_cameras, auxiliary_cameras, config)

        # Only ring cameras should be keys
        assert set(pairs.keys()) == set(ring_cameras)
        assert "center" not in pairs

    def test_self_exclusion(self):
        """Test that no camera appears in its own source list."""
        ring_cameras = ["cam_a", "cam_b", "cam_c", "cam_d"]
        positions = {
            "cam_a": torch.tensor([1.0, 0.0, 0.0]),
            "cam_b": torch.tensor([0.0, 1.0, 0.0]),
            "cam_c": torch.tensor([-1.0, 0.0, 0.0]),
            "cam_d": torch.tensor([0.0, -1.0, 0.0]),
        }

        config = PairSelectionConfig(num_neighbors=3, include_center=False)
        pairs = select_pairs(positions, ring_cameras, [], config)

        # Verify no camera is in its own source list
        for ref_cam, sources in pairs.items():
            assert ref_cam not in sources

    def test_source_list_ordering(self):
        """Test that source cameras are ordered by distance (nearest first)."""
        ring_cameras = ["cam_a", "cam_b", "cam_c", "cam_d"]
        # Place cameras so distances from cam_a are unambiguous
        positions = {
            "cam_a": torch.tensor([0.0, 0.0, 0.0]),
            "cam_b": torch.tensor([1.0, 0.0, 0.0]),  # distance 1.0
            "cam_c": torch.tensor([2.0, 0.0, 0.0]),  # distance 2.0
            "cam_d": torch.tensor([3.0, 0.0, 0.0]),  # distance 3.0
        }

        config = PairSelectionConfig(num_neighbors=3, include_center=False)
        pairs = select_pairs(positions, ring_cameras, [], config)

        sources = pairs["cam_a"]
        # Should be ordered: cam_b (1.0), cam_c (2.0), cam_d (3.0)
        assert sources == ["cam_b", "cam_c", "cam_d"]

    def test_num_neighbors_exceeds_available(self):
        """Test that num_neighbors is clamped to available camera count."""
        ring_cameras = ["cam_a", "cam_b", "cam_c", "cam_d"]
        positions = {
            "cam_a": torch.tensor([0.0, 0.0, 0.0]),
            "cam_b": torch.tensor([1.0, 0.0, 0.0]),
            "cam_c": torch.tensor([2.0, 0.0, 0.0]),
            "cam_d": torch.tensor([3.0, 0.0, 0.0]),
        }

        # Request 10 neighbors, but only 3 others are available
        config = PairSelectionConfig(num_neighbors=10, include_center=False)
        pairs = select_pairs(positions, ring_cameras, [], config)

        # Each camera should have exactly 3 sources (all other ring cameras)
        for sources in pairs.values():
            assert len(sources) == 3

    def test_num_neighbors_zero(self):
        """Test that num_neighbors=0 returns empty source lists (or just auxiliary)."""
        ring_cameras = ["cam_a", "cam_b", "cam_c"]
        positions = {
            "cam_a": torch.tensor([0.0, 0.0, 0.0]),
            "cam_b": torch.tensor([1.0, 0.0, 0.0]),
            "cam_c": torch.tensor([2.0, 0.0, 0.0]),
            "center": torch.tensor([0.0, 0.0, 1.0]),
        }
        auxiliary_cameras = ["center"]

        # Test with include_center=False: should be empty
        config = PairSelectionConfig(num_neighbors=0, include_center=False)
        pairs = select_pairs(positions, ring_cameras, [], config)
        for sources in pairs.values():
            assert len(sources) == 0

        # Test with include_center=True: should only have auxiliary
        config = PairSelectionConfig(num_neighbors=0, include_center=True)
        pairs = select_pairs(positions, ring_cameras, auxiliary_cameras, config)
        for sources in pairs.values():
            assert sources == ["center"]

    def test_determinism(self):
        """Test that repeated calls with same inputs produce identical results."""
        ring_cameras = ["cam_a", "cam_b", "cam_c", "cam_d", "cam_e"]
        positions = {
            "cam_a": torch.tensor([0.0, 0.0, 0.0]),
            "cam_b": torch.tensor([1.0, 0.0, 0.0]),
            "cam_c": torch.tensor([0.5, 0.5, 0.0]),
            "cam_d": torch.tensor([-1.0, 0.0, 0.0]),
            "cam_e": torch.tensor([0.0, 1.0, 0.0]),
        }

        config = PairSelectionConfig(num_neighbors=3, include_center=False)

        # Call twice
        pairs1 = select_pairs(positions, ring_cameras, [], config)
        pairs2 = select_pairs(positions, ring_cameras, [], config)

        # Results should be identical
        assert pairs1 == pairs2

    def test_asymmetric_geometry(self):
        """Test with non-uniform camera positions."""
        ring_cameras = ["cam_a", "cam_b", "cam_c", "cam_d"]
        positions = {
            "cam_a": torch.tensor([0.0, 0.0, 0.0]),
            "cam_b": torch.tensor([0.5, 0.0, 0.0]),  # distance 0.5
            "cam_c": torch.tensor([0.0, 2.0, 0.0]),  # distance 2.0
            "cam_d": torch.tensor([10.0, 0.0, 0.0]),  # distance 10.0
        }

        config = PairSelectionConfig(num_neighbors=2, include_center=False)
        pairs = select_pairs(positions, ring_cameras, [], config)

        # For cam_a, nearest 2 should be cam_b (0.5) and cam_c (2.0)
        sources = pairs["cam_a"]
        assert sources == ["cam_b", "cam_c"]

    def test_empty_auxiliary_list_with_include_center(self):
        """Test that empty auxiliary list doesn't cause error."""
        ring_cameras = ["cam_a", "cam_b", "cam_c"]
        positions = {
            "cam_a": torch.tensor([0.0, 0.0, 0.0]),
            "cam_b": torch.tensor([1.0, 0.0, 0.0]),
            "cam_c": torch.tensor([2.0, 0.0, 0.0]),
        }

        config = PairSelectionConfig(num_neighbors=2, include_center=True)
        pairs = select_pairs(positions, ring_cameras, [], config)

        # Should work without error, just no auxiliary cameras added
        for sources in pairs.values():
            assert len(sources) == 2  # Only ring neighbors

    def test_tiebreaking_by_name(self):
        """Test that equal distances are broken by alphabetical ordering."""
        ring_cameras = ["cam_b", "cam_a", "cam_c", "cam_d"]
        # Place cam_a, cam_b, cam_c at equal distance from cam_d
        positions = {
            "cam_d": torch.tensor([0.0, 0.0, 0.0]),
            "cam_a": torch.tensor([1.0, 0.0, 0.0]),  # distance 1.0
            "cam_b": torch.tensor([0.0, 1.0, 0.0]),  # distance 1.0
            "cam_c": torch.tensor([0.0, 0.0, 1.0]),  # distance 1.0
        }

        config = PairSelectionConfig(num_neighbors=2, include_center=False)
        pairs = select_pairs(positions, ring_cameras, [], config)

        sources = pairs["cam_d"]
        # With equal distances, should be sorted alphabetically: cam_a, cam_b
        assert sources == ["cam_a", "cam_b"]

    def test_multiple_auxiliary_cameras(self):
        """Test with multiple auxiliary cameras."""
        ring_cameras = ["ring_a", "ring_b"]
        positions = {
            "ring_a": torch.tensor([1.0, 0.0, 0.0]),
            "ring_b": torch.tensor([-1.0, 0.0, 0.0]),
            "aux1": torch.tensor([0.0, 0.0, 0.0]),
            "aux2": torch.tensor([0.0, 0.0, 1.0]),
        }
        auxiliary_cameras = ["aux1", "aux2"]

        config = PairSelectionConfig(num_neighbors=1, include_center=True)
        pairs = select_pairs(positions, ring_cameras, auxiliary_cameras, config)

        # Each ring camera should have 1 ring neighbor + 2 auxiliary = 3 total
        for sources in pairs.values():
            assert len(sources) == 3
            # Both auxiliary cameras should be included
            assert "aux1" in sources
            assert "aux2" in sources
