# Testing Patterns

**Analysis Date:** 2026-02-14

## Test Framework

**Runner:** pytest

**Config:** `pyproject.toml` (no separate pytest.ini)
```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
```

**Assertion Library:** pytest's native `assert` statements + `torch.testing.assert_close()`

**Run Commands:**
```bash
pytest tests/                           # Run all tests
pytest tests/ -m "not slow"             # Run fast tests only
pytest tests/test_projection/           # Run specific module tests
pytest tests/test_dense/test_cost.py    # Run specific test file
pytest tests/ -v                        # Verbose output
pytest tests/ --tb=short                # Shorter traceback format
```

**Coverage (installed but not enforced):**
```bash
pytest tests/ --cov=aquamvs --cov-report=html
```

## Test File Organization

**Location:** Co-located with source code
- Source: `src/aquamvs/projection/refractive.py`
- Tests: `tests/test_projection/test_refractive.py`

**Naming:**
- Test files: `test_<module_name>.py` (exact correspondence to source module)
- Test classes: `Test<FunctionName>` (one class per function group, optional)
- Test functions: `test_<behavior_or_condition>` (descriptive, not numbered)

**Structure:**
```
tests/
├── conftest.py                    # Shared fixtures (device fixture)
├── test_calibration.py            # Top-level module tests
├── test_projection/
│   ├── __init__.py
│   ├── test_protocol.py          # Protocol compliance tests
│   ├── test_refractive.py        # RefractiveProjectionModel tests
│   └── test_cross_validation.py  # PyTorch vs AquaCal cross-checks
├── test_features/
│   ├── __init__.py
│   ├── test_extraction.py
│   ├── test_matching.py
│   └── test_roma.py
├── test_dense/
│   ├── __init__.py
│   ├── test_cost.py
│   ├── test_plane_sweep.py
│   └── test_roma_depth.py
└── test_fusion/
    ├── __init__.py
    └── test_consistency.py
```

## Test Structure

**Suite Organization:**

```python
class TestTriangulateRays:
    """Tests for triangulate_rays() function."""

    def test_two_rays_intersection(self, device):
        """Test triangulation with two rays that intersect at a known point."""
        # Setup: Create inputs
        target = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=torch.float32)
        origin1 = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)
        direction1 = (target - origin1) / torch.linalg.norm(target - origin1)

        # Execute: Call function
        rays = [(origin1, direction1), (origin2, direction2)]
        result = triangulate_rays(rays)

        # Assert: Check results
        torch.testing.assert_close(result, target, atol=1e-5, rtol=0)
```

**Patterns:**

1. **Setup-Execute-Assert:** Clear three-phase structure
   - Setup: Create fixtures, synthetic data, mock objects
   - Execute: Call the function under test
   - Assert: Verify outcomes

2. **Descriptive test names:** What behavior is tested, not how
   - Good: `test_parallel_rays_raises()`
   - Bad: `test_parallel_rays_1()`

3. **One assertion focus per test:** Test one behavior
   - Multiple assertions OK if testing same outcome
   - Separate tests for error cases, edge cases, normal cases

## Mocking

**Framework:** `unittest.mock` for limited mocking; prefer real objects

**Patterns:**

1. **Mock external services:**
   ```python
   # When testing CLI without actually running pipeline
   with patch("aquamvs.pipeline.run_pipeline") as mock_run:
       mock_run.return_value = None
       # Call CLI
       result = cli_run_command(config_path)
   ```

2. **Avoid mocking internals:**
   - Test integration: use real `RefractiveProjectionModel`, not mock
   - Test features: use real feature extractors or lightweight synthetic images

3. **Test data fixtures:** Prefer fixtures over mocks for data
   ```python
   @pytest.fixture
   def simple_camera(device):
       """Simple camera at world origin looking straight down."""
       R = torch.eye(3, dtype=torch.float32, device=device)
       t = torch.zeros(3, dtype=torch.float32, device=device)
       K = torch.tensor([...], dtype=torch.float32, device=device)
       return K, R, t
   ```

**What to Mock:**
- Heavy external services (file I/O for CLI testing)
- Entire sub-pipelines when testing one component
- Example: Mock `VideoSet` in CLI tests to avoid reading actual video files

**What NOT to Mock:**
- Geometric operations (project, cast_ray, triangulate)
- Core math (cost computation, filtering)
- PyTorch tensor operations
- Mock defeats the test; real data catches integration bugs

## Fixtures and Factories

**Test Data Factories:**

Synthetic data builders inline in test file or in `conftest.py`:

```python
def create_checkerboard_image(
    height: int = 480, width: int = 640, square_size: int = 40
) -> torch.Tensor:
    """Create a checkerboard pattern image with corners for feature detection."""
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    checker = ((x // square_size) + (y // square_size)) % 2
    image = (checker * 255).to(torch.uint8)
    return image
```

**Fixture Location:**

Shared fixtures in `tests/conftest.py`:
```python
@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Parametrized device fixture for CPU and CUDA testing."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)
```

Module-specific fixtures in test file (e.g., `test_refractive.py`):
```python
@pytest.fixture
def simple_camera(device):
    """Simple camera at world origin looking straight down."""
    R = torch.eye(3, dtype=torch.float32, device=device)
    t = torch.zeros(3, dtype=torch.float32, device=device)
    K = torch.tensor([...], dtype=torch.float32, device=device)
    return K, R, t
```

**Fixture Scope:**
- Default: `function` (new fixture per test)
- Larger scope (module, session) only for expensive setup (model loading)

## Coverage

**Requirements:** Not enforced

**Coverage Tool:** pytest-cov
```bash
pytest tests/ --cov=aquamvs --cov-report=html
```

**Gaps noted but not addressed:**
- Some utility functions have minimal tests (e.g., coloring utilities)
- CLI argument parsing partially mocked
- E2E tests are integration tests, not true end-to-end

## Test Types

**Unit Tests:**
- Scope: Single function or tightly coupled class
- Isolation: Use fixtures for dependencies, mock external services
- Examples: `test_projection/test_refractive.py`, `test_dense/test_cost.py`
- Coverage: Normal cases, edge cases (empty input, boundary values), error cases

**Integration Tests:**
- Scope: Multiple components working together
- Isolation: Real sub-components, mocked only at external boundaries
- Examples: `tests/test_pipeline.py`, `tests/test_fusion/test_fusion.py`
- Coverage: Full pipeline workflows, cross-module data flow

**Cross-Validation Tests:**
- Scope: Comparing PyTorch implementation against AquaCal NumPy reference
- Location: `tests/test_projection/test_cross_validation.py`
- Purpose: Ensure PyTorch projection model matches ground truth
- Tolerance: `atol=1e-5` pixels (documented in CLAUDE.md)

**E2E Tests:**
- Scope: Entire reconstruction pipeline end-to-end
- Location: `tests/test_integration.py`
- Setup: Synthetic scene with multiple cameras, ground truth point cloud
- Verification: Output point cloud is within expected accuracy

## Common Patterns

**Async Testing:**

Not used (PyTorch is synchronous, no async/await patterns in codebase).

**Error Testing:**

```python
def test_parallel_rays_raises(self, device):
    """Test that parallel rays raise ValueError."""
    direction = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
    origin1 = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)
    origin2 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)

    rays = [(origin1, direction), (origin2, direction)]

    with pytest.raises(ValueError, match="Degenerate"):
        triangulate_rays(rays)
```

**Device Parametrization:**

Parametrize tests over CPU and CUDA:

```python
@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Parametrize tests over CPU and CUDA devices."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)

class TestRefractiveProjectionModel:
    def test_cast_ray(self, device):
        """Test ray casting on both CPU and CUDA."""
        model = RefractiveProjectionModel(..., device=device)
        # Test uses fixture device automatically
```

**Numerical Assertion:**

Use `torch.testing.assert_close()` for tensor comparisons:

```python
# Projection roundtrips: project -> cast_ray -> reproject
torch.testing.assert_close(reprojected_pixels, original_pixels, atol=1e-4, rtol=0)

# 3D point positions
torch.testing.assert_close(result_point, expected_point, atol=1e-5, rtol=0)
```

**Tolerance Defaults** (from CLAUDE.md):
- Projection roundtrips: `atol=1e-4` pixels
- PyTorch vs AquaCal NumPy: `atol=1e-5` pixels
- 3D point positions: `atol=1e-5` meters (0.01 mm)

**Temporary Files:**

Use `tempfile.TemporaryDirectory()` for I/O tests:

```python
def test_save_load_roundtrip():
    """Test saving and loading matches dict."""
    matches = {"ref_keypoints": tensor(...), ...}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "matches.pt"

        save_matches(matches, path)
        assert path.exists()

        loaded = load_matches(path)
        assert torch.allclose(loaded["ref_keypoints"], matches["ref_keypoints"])
```

**Monkeypatching:**

Patch module-level functions for testing without side effects:

```python
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(...))])
def test_prepare_lightglue_input(device):
    """Test preparing features for LightGlue input format."""
    feats = {"keypoints": ..., "descriptors": ..., "scores": ...}
    result = _prepare_lightglue_input(feats, image_size, device)

    assert result["keypoints"].device.type == device
```

**Conditional Skipping:**

Skip tests when dependencies missing:

```python
try:
    result = extract_features(image, config, device="cpu")
    assert isinstance(result, dict)
except Exception as e:
    if "SuperPoint" in str(type(e).__name__):
        pytest.skip("SuperPoint model not available")
    raise
```

**Parametrized Tests:**

Using `@pytest.mark.parametrize`:

```python
@pytest.mark.parametrize("extractor_type", ["superpoint", "aliked", "disk"])
def test_create_extractor(extractor_type):
    """Test extractor creation for all supported backends."""
    config = FeatureExtractionConfig(extractor_type=extractor_type)
    extractor = create_extractor(config, device="cpu")
    assert isinstance(extractor, torch.nn.Module)
```

## Special Test Categories

**Protocol Compliance Tests:**

Verify interface implementations:

```python
def test_protocol_compliance_positive():
    """Verify that a class with both methods passes isinstance check."""
    dummy = _DummyProjectionModel()
    assert isinstance(dummy, ProjectionModel)

def test_protocol_compliance_missing_project():
    """Verify that a class missing project fails isinstance check."""
    obj = _MissingProject()
    assert not isinstance(obj, ProjectionModel)
```

**Configuration Tests:**

Test config loading, validation, serialization:

```python
class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_valid_config_from_yaml(self):
        """Load valid config from YAML."""
        config_str = "..."
        config = PipelineConfig.from_yaml(config_str)
        assert config.device.cuda is False
```

**Visualization Tests:**

Visual output tests (mostly smoke tests, minimal assertion):

```python
def test_visualize_features():
    """Test that feature visualization doesn't crash."""
    image = torch.randint(0, 256, (100, 100, 3), dtype=torch.uint8)
    features = {"keypoints": torch.tensor([[10.0, 20.0]]), ...}

    # Smoke test: just ensure it doesn't raise
    result = visualize_features(image, features)
    assert isinstance(result, np.ndarray)
```

---

*Testing analysis: 2026-02-14*
