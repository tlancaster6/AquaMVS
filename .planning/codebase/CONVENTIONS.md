# Coding Conventions

**Analysis Date:** 2026-02-14

## Naming Patterns

**Files:**
- Snake case: `projection.py`, `refraction.py`, `plane_sweep.py`
- Pattern: Descriptive single-word or compound terms matching module responsibility

**Functions:**
- Snake case: `triangulate_rays()`, `cast_ray()`, `filter_depth_map()`
- Private helper functions prefixed with underscore: `_triangulate_two_rays_batch()`, `_apply_clahe()`, `_make_pixel_grid()`
- Imperative verbs for action functions: `compute_cost()`, `generate_depth_hypotheses()`, `save_sparse_cloud()`

**Variables:**
- Snake case throughout: `ray_depth`, `origin_z`, `direction_z`, `valid_pixels`
- Device variables: `device` (common), occasionally `device_` when disambiguating from parameters
- Coordinate variables follow domain conventions: `origin`, `direction`, `points`, `pixels`

**Classes:**
- PascalCase: `RefractiveProjectionModel`, `ProjectionModel`, `PipelineContext`
- Configuration classes: `PipelineConfig`, `FeatureExtractionConfig`, `DenseStereoConfig`

**Constants:**
- Uppercase snake case: `UPPER_SNAKE_CASE` (limited usage; most configs are dataclass fields)
- Example: Tolerance constants when hardcoded

## Code Style

**Formatter:** Black (enforced via `.claude/hooks/auto_format.py`)

**Line length:** Black default (88 characters)

**Docstrings:** Google style

Format:
```python
def project(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project 3D world points to 2D pixel coordinates.

    Args:
        points: 3D points in world frame, shape (N, 3), float32.

    Returns:
        pixels: 2D pixel coordinates (u, v), shape (N, 2), float32.
        valid: Boolean validity mask, shape (N,). False for points that
            cannot be projected (behind camera, total internal reflection, etc.).
    """
```

**Type Hints:**
- Full type hints on all public functions (parameters and return type)
- Include shapes and dtypes in docstring Returns/Args sections
- Example shapes: `shape (N, 3)`, `shape (H, W)`, `shape (M, 3, 3)`
- Use modern syntax: `torch.Tensor`, `tuple[Type1, Type2]`, `list[Type]`, not `typing.List`, `typing.Tuple`
- Union types: `str | Path`, `bool | torch.device` (Python 3.10+ syntax)
- Optional: `int | None`, not `Optional[int]`

**Imports:** Standard structure with three groups separated by blank lines

```python
import logging
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import PipelineConfig
from .projection.protocol import ProjectionModel
```

Order:
1. Standard library (logging, math, pathlib, etc.)
2. Third-party (torch, numpy, cv2, etc.)
3. Local package imports (relative dots)

## Error Handling

**Approach:** Explicit validation with descriptive exceptions, not silent failures

**Patterns:**

1. **Input validation:**
   ```python
   if len(rays) < 2:
       raise ValueError("Need at least 2 rays")

   if config.extractor_type not in ("superpoint", "aliked", "disk"):
       raise ValueError(
           f"Unknown extractor_type: {config.extractor_type!r}. "
           "Valid types: 'superpoint', 'aliked', 'disk'"
       )
   ```

2. **Algorithmic failures:**
   ```python
   try:
       P = torch.linalg.solve(A_sum, b_sum)
   except torch.linalg.LinAlgError:
       raise ValueError("Degenerate ray configuration")
   ```

3. **File/IO errors:** Let exceptions propagate with context
   ```python
   # No try-catch for standard file operations; let pathlib/cv2 raise
   mask_path = mask_dir / f"{camera_name}.png"
   if not mask_path.exists():
       logger.debug("Mask not found for %s: %s", camera_name, mask_path)
       return None
   ```

4. **Invalid projections:** Return validity masks instead of raising
   ```python
   # From ProjectionModel protocol:
   # Return (pixels, valid_mask) where valid_mask is False for failed projections
   # Invalid pixels have undefined coordinate values (may be NaN or arbitrary)
   ```

**Validity Masks:** Used for geometric operations that may fail per-sample
- Functions return `(result, valid)` tuple where `valid` is boolean shape `(N,)`
- Invalid entries in `result` are undefined (no NaN guarantee)
- Examples: `project()`, `cast_ray()`, `_triangulate_two_rays_batch()`

## Logging

**Framework:** Python's standard `logging` module

**Usage Pattern:**
```python
import logging

logger = logging.getLogger(__name__)

# In functions:
logger.debug("Mask not found for %s: %s", camera_name, mask_path)
logger.warning("Mask size mismatch for %s: expected %s, got %s", camera_name, expected, actual)
logger.info("Processing frame %d", frame_idx)
```

**Levels:**
- `debug()`: Detailed diagnostic info (missing optional files, internal state)
- `info()`: General progress messages (frame processing start/end)
- `warning()`: Degraded behavior but recovery available (missing mask, size mismatch)
- `error()`: Serious failure (do not use; raise exception instead)

## Comments

**When to Comment:**
- **Algorithm explanation:** Complex mathematics or non-obvious workarounds
- **Gotchas:** Tricky indexing, device movement, tensor shape assumptions
- **References:** Links to external references (e.g., AquaCal geometry files)
- **Avoid:** Obvious code that reads as English (e.g., "Initialize X" above `x = 0`)

**JSDoc/Docstrings:**
- All public functions and classes required
- Public = anything in a module's `__all__` list
- Private helpers (prefixed `_`) may have docstrings but not required

**Examples from codebase:**

```python
# Step 1: Pinhole back-projection (pixels to rays in camera frame)
# Homogeneous pixel coords: (N, 3)
ones = torch.ones(N, 1, device=pixels.device, dtype=pixels.dtype)
pixels_h = torch.cat([pixels, ones], dim=-1)  # (N, 3)

# Normalize pixel coords to [-1, 1] for grid_sample
# (Avoid grid_sample NaN behavior by replacing NaN with 0 before sampling)
depth_clean = torch.where(
    torch.isnan(depth_map), torch.zeros_like(depth_map), depth_map
)
```

## Function Design

**Size:** Keep functions focused and under 100 lines where practical
- Larger functions (e.g., `plane_sweep_stereo()` ~150 lines) are acceptable when tightly coupled steps
- Use helper functions for distinct sub-tasks (e.g., `_warp_source_at_depth()`)

**Parameters:**
- Batch-oriented: Functions accept batches of inputs (N samples) and return batches of outputs
- Device-agnostic: Accept input device implicitly from tensor device, no explicit `device` param in low-level math
- Pipeline functions receive `device` from config: `plane_sweep_stereo(..., device=config.device)`

**Return Values:**
- Tuple returns for multiple related outputs: `(origins, directions)` from `cast_ray()`
- Single return for single concept: `point` from `triangulate_rays()`
- Include validity information: `(values, valid_mask)` when operation can fail per-sample

**Device Convention:**
- High-level pipeline modules: Accept explicit `device` parameter, use it for `to(device)` calls
- Low-level geometry operations: Follow input tensor device, no explicit device param
- Never hardcode `.cuda()` or `.cpu()`

## Module Design

**Public API via `__init__.py`:**
All modules define a module-level docstring and explicit `__all__` list:

```python
"""Feature matching using LightGlue."""

# ... imports and functions ...

__all__ = [
    "create_matcher",
    "match_pair",
    "match_all_pairs",
    "save_matches",
    "load_matches",
]
```

Package-level `__init__.py` imports all public submodule APIs:
```python
from .features.extraction import extract_features, create_extractor
from .features.matching import match_pair, match_all_pairs
from .features.pairs import select_pairs

__all__ = [
    "extract_features",
    "create_extractor",
    "match_pair",
    ...
]
```

**Barrel Files:** Full re-export of submodule public APIs from package `__init__.py`
- Users import from top level: `from aquamvs import extract_features`
- Implementation details in submodules: `from aquamvs.features.extraction import ...`

**Layering:**
- `config.py`: Configuration dataclasses, no business logic
- `calibration.py`, `projection/`: Core geometric models
- `features/`, `dense/`: Feature and matching operations
- `fusion.py`, `triangulation.py`: Multi-view aggregation
- `pipeline.py`: Top-level orchestration
- `cli.py`: Command-line interface

## Coordinate System Conventions

(Documented in `CLAUDE.md` Domain Conventions, enforced in code)

**Tensor Shapes:**
- Pixels: `(u, v)` where u=column, v=row (OpenCV convention)
- Points: `(x, y, z)` in world frame
- Rays: `(origin, direction)` both shape `(3,)` or batches `(N, 3)`
- Ray depth: scalar or `(N,)`, distance along ray from origin

**Depth Parameterization:**
- Ray depth: distance from surface intersection along refracted ray
- 3D point: `point = origin + depth * direction`
- World Z: `Z = origin_z + depth * direction_z`

## Data Types

**Floating Point:** Float32 throughout (torch.float32 or float)
- No float64 except at AquaCal boundary (numpy)
- GPU-friendly and matches typical CV model expectations

**Validity:** Use boolean tensors (torch.bool or bool dtype)
- Not integer masks (0/1)
- Example: `valid = torch.ones(N, dtype=torch.bool)`

**Images:**
- Input: uint8 or float32 (float32 in [0, 1] or [0, 255] ranges)
- Internal processing: float32
- Output: uint8 for visualization/export

## Configuration

All pipeline configuration via dataclasses in `config.py`:
- Immutable once created
- Serializable to/from YAML
- Composed hierarchically: `PipelineConfig` contains `DenseStereoConfig`, `FeatureExtractionConfig`, etc.

Example usage:
```python
config = PipelineConfig(
    frames=FrameSamplingConfig(start=0, stop=100, step=5),
    features=FeatureExtractionConfig(extractor_type="superpoint", max_keypoints=2048),
    device=DeviceConfig(cuda=False),
)
```

---

*Convention analysis: 2026-02-14*
