# Phase 3: Pipeline Decomposition and Modularization - Research

**Researched:** 2026-02-14
**Domain:** Python pipeline architecture refactoring and modular design
**Confidence:** HIGH

## Summary

Phase 3 refactors the monolithic 1124-line `pipeline.py` into a modular `pipeline/` package with separate builder, runner, and stage modules. The research reveals that modern Python scientific computing libraries (PyTorch, scikit-learn, Kedro) use a combination of builder patterns, protocol-based interfaces, and config-driven stage composition to achieve maintainability and extensibility. The codebase already demonstrates mature patterns (Protocol interfaces in `projection/`, factory functions in `features/`, Pydantic config validation) that should guide the decomposition strategy.

**Primary recommendation:** Use Protocol-based interfaces for extension points (FrameSource, CalibrationProvider), factory functions for stage instantiation, and a Pipeline class with a builder-style API. Keep stages internal (not independently importable) initially, focusing on clean separation of concerns within the package. The clean break backward compatibility strategy is appropriate given single-user context.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Clean break** — no deprecation shims on old imports
- Only current user is the author; no external consumers to protect
- Primary entry points are CLI (`aquamvs run`) and `run_pipeline()` function — these continue to work
- Update ROADMAP success criterion #5 to remove backward-compat requirement
- Any breakage in tests or internal imports is fixed directly during refactoring

### Public API Surface
- **Pipeline class** as primary programmatic entry point: `pipeline = Pipeline(config); pipeline.run()`
- Top-level re-export: `from aquamvs import Pipeline` works (canonical location: `from aquamvs.pipeline import Pipeline`)
- **Claude's discretion:** Whether to expose intermediate results (depth maps, point clouds) as attributes vs. final output only
- **Claude's discretion:** Whether individual stages (matching, depth estimation, fusion) are independently importable or internal-only

### Extension Points
- Matchers, depth estimation, and fusion are all **potentially swappable** (not urgent, but don't paint into a corner)
- Extension is an **advanced/Python API concern**, not a CLI concern
- **Claude's discretion:** Whether extension happens through Protocol/ABC or config-driven registration — pick what fits the codebase
- Document the process for substituting custom modules (note for Phase 4 documentation)

### AquaCal Isolation
- **FrameSource interface** — abstract frame reading; `VideoSet` and `ImageDirectorySet` are implementations
- **CalibrationProvider interface** — separate from frame reading; provides camera params AND refractive geometry
- CalibrationProvider includes refractive parameters (water_z, n_water, interface_normal)
- **Refraction-naive fallback:** If refractive parameters are missing, print descriptive warning and set n_air=n_water=1.0 (equivalent to non-refractive model downstream)
- Refraction-naive mode is minimally tested for now — add thorough testing as a TODO/backlog item

### Claude's Discretion Areas
- Module boundary decisions (how to split stages into files)
- Protocol vs ABC vs duck typing for interfaces
- Intermediate result exposure on Pipeline class
- Stage independence (importable individually or internal-only)

### Deferred Ideas (OUT OF SCOPE)
- Support for non-AquaCal calibration formats (COLMAP, Metashape) — future phase, enabled by CalibrationProvider interface
- Thorough testing of refraction-naive mode — backlog item after Phase 3
- Custom module documentation — Phase 4 (Documentation and Examples)
</user_constraints>

## Standard Stack

### Core Libraries (Already in Use)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| **Pydantic** | 2.x | Config validation, data models | Industry standard for Python config validation with excellent error messages, already adopted in Phase 2 |
| **PyTorch** | 1.x | Tensor operations, device management | Project convention (Phase 2), all math uses PyTorch for GPU support |
| **typing.Protocol** | Python 3.8+ | Structural subtyping interfaces | Standard library, preferred over ABC for duck-typed interfaces in modern Python |

### Supporting Patterns (From Ecosystem Research)

| Pattern | Purpose | When to Use |
|---------|---------|-------------|
| **Builder pattern** | Pipeline construction | Composing stages with configuration |
| **Factory functions** | Stage instantiation | Creating extractors, matchers, estimators based on config |
| **Chain of Responsibility** | Sequential processing | Data flowing through independent stage modules |
| **Protocol interfaces** | Extension points | Defining behavior without inheritance (FrameSource, CalibrationProvider) |

### Installation

No new dependencies required. All patterns use Python 3.10+ standard library and existing project dependencies.

## Architecture Patterns

### Recommended Project Structure

```
src/aquamvs/pipeline/
├── __init__.py              # Public API: Pipeline class, run_pipeline()
├── builder.py               # Pipeline setup, context creation
├── runner.py                # Frame iteration, stage orchestration
├── stages/
│   ├── __init__.py          # Internal-only (no public exports)
│   ├── undistortion.py      # Stage 1: undistort + color norm
│   ├── sparse_matching.py   # Stages 2-4: features + matching + triangulation (LightGlue)
│   ├── dense_matching.py    # Dense matching via RoMa
│   ├── depth_estimation.py  # Stages 5-6: depth ranges + plane sweep (LightGlue path)
│   ├── fusion.py            # Stage 7-8: filtering + fusion
│   └── surface.py           # Stage 9: surface reconstruction
├── interfaces.py            # Protocol definitions: FrameSource, CalibrationProvider
└── context.py               # PipelineContext dataclass (unchanged)
```

**Key principles:**
- **stages/** is internal — not exported from `__init__.py`
- Each stage module is self-contained with clear inputs/outputs
- Stages import from sibling modules (`features/`, `dense/`, `fusion/`, etc.) but don't depend on each other
- `runner.py` orchestrates stage execution based on config (`matcher_type`, `pipeline_mode`)

### Pattern 1: Protocol-Based Interfaces (Structural Subtyping)

**What:** Define expected behavior via `typing.Protocol` without requiring inheritance.

**When to use:** For extension points where you want flexibility without forcing class hierarchies (FrameSource, CalibrationProvider).

**Why Protocol over ABC:**
- Already used successfully in `projection/protocol.py` (ProjectionModel)
- Allows third-party classes to "implement" interface without subclassing
- Better for duck-typed interfaces in scientific Python ecosystem
- Lighter weight than ABC for simple interfaces

**Example:**
```python
# src/aquamvs/pipeline/interfaces.py
from typing import Protocol, Iterator
import numpy as np

class FrameSource(Protocol):
    """Protocol for frame reading (video, image dirs, etc.)."""

    def iterate_frames(
        self,
        start: int = 0,
        stop: int | None = None,
        step: int = 1
    ) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
        """Iterate over frames.

        Yields:
            (frame_idx, {camera_name: image_bgr}) tuples.
        """
        ...

    def __enter__(self) -> "FrameSource":
        ...

    def __exit__(self, *args) -> None:
        ...

class CalibrationProvider(Protocol):
    """Protocol for calibration data provision."""

    @property
    def cameras(self) -> dict[str, "CameraData"]:
        """Per-camera calibration data."""
        ...

    @property
    def water_z(self) -> float:
        """Water surface Z-coordinate (meters)."""
        ...

    @property
    def n_water(self) -> float:
        """Refractive index of water."""
        ...

    # ... other refractive parameters
```

**Reference:** [Python Protocols: Leveraging Structural Subtyping – Real Python](https://realpython.com/python-protocol/)

**Confidence:** HIGH — Pattern already proven in this codebase (`ProjectionModel`), official Python typing spec, recommended by Real Python and Python typing docs.

### Pattern 2: Builder Pattern for Pipeline Construction

**What:** Separate pipeline configuration from execution via a builder that creates PipelineContext.

**When to use:** One-time setup that is reused across all frames (undistortion maps, projection models, camera pairs).

**Current code pattern:**
```python
# Current: setup_pipeline() returns PipelineContext
ctx = setup_pipeline(config)
# then: process_frame(frame_idx, images, ctx)
```

**Refactored pattern:**
```python
# pipeline/builder.py
def build_pipeline_context(config: PipelineConfig) -> PipelineContext:
    """One-time setup: load calibration, compute undistortion, create projections."""
    ...

# pipeline/__init__.py
class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.context = build_pipeline_context(config)

    def run(self) -> None:
        """Run full pipeline over all frames."""
        run_pipeline_impl(self.config, self.context)
```

**Benefit:** Clear separation of setup (builder) vs. execution (runner), matches scikit-learn pattern (fit/predict separation).

**Confidence:** HIGH — Builder pattern is well-established for complex object construction, already used implicitly in current `setup_pipeline()`.

### Pattern 3: Config-Driven Stage Routing

**What:** Use config fields (`matcher_type`, `pipeline_mode`) to route execution through different stage combinations.

**Current routing logic (lines 416-588 in pipeline.py):**
```python
if config.matcher_type == "roma":
    if config.pipeline_mode == "full":
        # roma+full: warps -> depth maps
    else:
        # roma+sparse: matches -> triangulation
elif config.matcher_type == "lightglue":
    # lightglue: features -> matches -> triangulation
```

**Refactored pattern:**
```python
# pipeline/runner.py
def run_frame_pipeline(frame_idx, images, ctx):
    # Stage 1: Undistortion (always)
    undistorted = run_undistortion_stage(images, ctx)

    # Stages 2-6: Matching + Depth (config-driven routing)
    if ctx.config.matcher_type == "roma" and ctx.config.pipeline_mode == "full":
        depth_maps = run_roma_full_path(undistorted, ctx)
        skip_fusion_filter = True
    elif ctx.config.matcher_type == "roma":
        matches = run_roma_sparse_path(undistorted, ctx)
        depth_maps = run_depth_estimation_path(undistorted, matches, ctx)
        skip_fusion_filter = False
    else:  # lightglue
        matches = run_lightglue_path(undistorted, ctx)
        depth_maps = run_depth_estimation_path(undistorted, matches, ctx)
        skip_fusion_filter = False

    # Stages 7-9: Fusion + Surface (always)
    fused_cloud = run_fusion_stage(depth_maps, undistorted, ctx, skip_filter=skip_fusion_filter)
    mesh = run_surface_stage(fused_cloud, undistorted, ctx)
```

**Benefit:** Explicit routing logic in one place, easier to test individual paths, clearer than deeply nested conditionals.

**Confidence:** MEDIUM — Pattern is logical but needs careful extraction to preserve existing behavior. Test coverage critical.

### Pattern 4: Factory Functions for Stage Components

**What:** Use factory functions (not classes) to instantiate stage components based on config.

**Current pattern (already in codebase):**
```python
# features/extraction.py
def create_extractor(extractor_type: str, **kwargs):
    if extractor_type == "superpoint":
        return SuperPoint(...)
    elif extractor_type == "aliked":
        return ALIKED(...)
    # ...

# features/matching.py
def create_matcher(extractor_type: str, **kwargs):
    return LightGlue(features=extractor_type, ...)
```

**Refactored stages use same pattern:**
```python
# pipeline/stages/sparse_matching.py
def run_sparse_matching_stage(images, ctx):
    """Extract features, match pairs, triangulate."""
    # Factory calls to features/ module
    features = extract_features_batch(images, ctx.config.sparse_matching, ctx.device)
    matches = match_all_pairs(features, ctx.pairs, ...)
    sparse_cloud = triangulate_all_pairs(ctx.projection_models, matches)
    return sparse_cloud
```

**Benefit:** Stages delegate to existing modular code (`features/`, `dense/`, `fusion/`), avoid duplication, maintain single responsibility.

**Confidence:** HIGH — Pattern already proven in codebase, matches scikit-learn and PyTorch conventions.

### Anti-Patterns to Avoid

- **Deep class hierarchies:** Don't create `BaseStage` → `MatchingStage` → `LightGlueMatchingStage`. Use functions and Protocols instead.
- **Stateful stages:** Stages should be pure functions (inputs → outputs), not classes with mutable state. Context is read-only.
- **Cross-stage dependencies:** Stage modules should not import each other. They import from `features/`, `dense/`, `fusion/`, etc.
- **Global singletons:** No module-level pipeline state. All state in PipelineContext or local variables.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| **Config validation** | Custom dict parsing with manual error checks | Pydantic models (already adopted) | Handles validation, defaults, type coercion, clear error messages |
| **Pipeline stage composition** | Custom stage registration/discovery system | Config-driven routing + factory functions | Simpler, explicit, easier to test |
| **Interface definitions** | Custom base classes or metaclasses | `typing.Protocol` with `@runtime_checkable` if needed | Standard library, static type checkers understand it, lighter weight |
| **Deprecation warnings** | Manual `warnings.warn()` calls | `warnings.deprecated()` decorator (Python 3.13+) or established pattern | Consistent messaging, easier to track |

**Key insight:** Python's scientific computing ecosystem (NumPy, SciPy, PyTorch, scikit-learn) favors composition over inheritance, protocols over ABCs, and configuration over code for defining workflows. AquaMVS should follow these conventions.

## Common Pitfalls

### Pitfall 1: Over-Abstracting Stage Interfaces

**What goes wrong:** Creating complex stage protocols or base classes that force every stage to implement the same interface, even when stages have fundamentally different inputs/outputs.

**Why it happens:** Applying enterprise design patterns (Strategy, Command) without considering domain constraints.

**How to avoid:**
- Keep stages as simple functions: `(inputs, context) -> outputs`
- Only extract common interface when genuinely needed (e.g., FrameSource, CalibrationProvider)
- If a Protocol has more than 3-4 methods, it's probably too complex

**Warning signs:**
- Stage Protocol with methods that only apply to some implementations
- Lots of `NotImplementedError` or `pass` in stage implementations
- Complex method signatures with many optional parameters

**Confidence:** HIGH — Based on Python scientific computing best practices, observed issues in over-engineered CV pipelines.

### Pitfall 2: Breaking Existing Tests During Refactoring

**What goes wrong:** Refactoring internal imports breaks test assertions that relied on specific module locations.

**Why it happens:** Tests import from `aquamvs.pipeline` but modules move to `aquamvs.pipeline.stages.X`.

**How to avoid:**
1. Update `__init__.py` to re-export public names (Pipeline, run_pipeline, PipelineContext)
2. Keep internal helpers (e.g., `_should_viz`) in the module where they're used
3. Update tests incrementally: one module → update tests → verify → next module
4. Run full test suite after each migration step

**Warning signs:**
- ImportError failures in test suite
- Tests passing but with different behavior (wrong function imported)
- Circular import errors (stages importing from each other)

**Confidence:** HIGH — Standard refactoring risk, well-documented in Python testing literature.

### Pitfall 3: Protocol Runtime Checks are Slow and Incomplete

**What goes wrong:** Using `@runtime_checkable` + `isinstance(obj, ProtocolClass)` for validation, expecting it to verify method signatures.

**Why it happens:** Misunderstanding Protocol's runtime checking limitations — it only checks attribute/method existence, not signatures or types.

**How to avoid:**
- Use Protocols primarily for static type checking (mypy, pyright)
- If runtime validation needed, check specific methods/attributes manually or use Pydantic
- Don't use `isinstance()` with Protocols in hot paths (it's slow)

**Reference:** [Python Protocols: Leveraging Structural Subtyping – Real Python](https://realpython.com/python-protocol/) warns: "isinstance() with protocols is not completely safe at runtime. Signatures of methods are not checked."

**Warning signs:**
- `isinstance(obj, ProtocolClass)` passes but method calls fail with type errors
- Performance degradation when Protocol checks are in tight loops
- False confidence that runtime isinstance() guarantees correctness

**Confidence:** HIGH — Documented in official typing spec and Real Python guides.

### Pitfall 4: Premature Exposure of Stage Internals

**What goes wrong:** Making individual stages (`sparse_matching.py`, `fusion.py`) importable from top-level API before understanding usage patterns.

**Why it happens:** Desire for "modularity" and "reusability" without concrete use cases.

**How to avoid:**
- Keep stages internal initially (`pipeline/stages/` not exported)
- Only expose when there's a concrete extension use case
- Pipeline class should be the main public API
- Document intent to expose stages later (Phase 4)

**Warning signs:**
- `__all__` in stage modules before any external usage
- Complex re-export chains (`__init__.py` importing from deeply nested modules)
- Stage APIs that feel "designed for reuse" but have no actual reusers

**Confidence:** MEDIUM — Based on YAGNI principle and Phase 3's clean-break strategy. Can always expose later.

### Pitfall 5: Ignoring Existing Test Coverage

**What goes wrong:** Refactoring without checking which internal functions are already tested, leading to broken tests or untested code paths.

**Why it happens:** Assuming tests only cover public API when they often test internal helpers.

**How to avoid:**
1. Run `pytest --collect-only tests/test_pipeline.py` to see what's tested
2. Grep for imports: `grep -r "from aquamvs.pipeline import" tests/`
3. Keep internal helpers that are directly tested in accessible locations
4. If moving a tested internal function, update test imports or re-export from new location

**Current test coverage to preserve:**
- `setup_pipeline()` — tested for structure, undistortion K usage
- `process_frame()` — tested for directory structure, stage execution
- `_should_viz()` — tested for stage filtering logic
- PipelineContext creation and validation

**Warning signs:**
- Test suite fails after "simple" module move
- Coverage drops significantly without corresponding code deletion
- Tests skip previously tested code paths

**Confidence:** HIGH — Existing test suite (test_pipeline.py) has 60+ tests covering setup, frame processing, and viz stages.

## Code Examples

Verified patterns from current codebase:

### Protocol Interface (Existing Pattern)

```python
# Source: src/aquamvs/projection/protocol.py (lines 8-54)
from typing import Protocol, runtime_checkable
import torch

@runtime_checkable
class ProjectionModel(Protocol):
    """Protocol for geometric projection models."""

    def project(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project 3D points to 2D pixels.

        Returns:
            pixels: (N, 2) pixel coordinates
            valid: (N,) boolean mask
        """
        ...

    def cast_ray(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cast rays from pixels.

        Returns:
            origins: (N, 3) ray origins
            directions: (N, 3) unit direction vectors
        """
        ...
```

**Usage:** Apply same pattern to FrameSource and CalibrationProvider.

### Factory Function (Existing Pattern)

```python
# Source: src/aquamvs/features/extraction.py (lines 40-71)
def create_extractor(
    extractor_type: Literal["superpoint", "aliked", "disk"],
    max_num_keypoints: int = 2048,
    detection_threshold: float = 0.005,
) -> torch.nn.Module:
    """Create feature extractor based on type.

    Handles config -> implementation mapping.
    """
    if extractor_type == "superpoint":
        from lightglue import SuperPoint
        return SuperPoint(
            max_num_keypoints=max_num_keypoints,
            detection_threshold=detection_threshold,
        ).eval()
    elif extractor_type == "aliked":
        from lightglue import ALIKED
        return ALIKED(
            max_num_keypoints=max_num_keypoints,
            detection_threshold=detection_threshold,
        ).eval()
    # ... etc
```

**Usage:** Apply same pattern to stage instantiation if stages become classes (currently prefer functions).

### Pydantic Config with Validation (Existing Pattern)

```python
# Source: src/aquamvs/config.py (lines 26-58)
from pydantic import BaseModel, model_validator
from typing import Literal

class PreprocessingConfig(BaseModel):
    """Preprocessing config with validation."""

    model_config = ConfigDict(extra="allow")  # Allow unknown keys

    color_norm_enabled: bool = False
    color_norm_method: Literal["gain", "histogram"] = "gain"
    frame_start: int = 0
    frame_stop: int | None = None
    frame_step: int = 1

    @model_validator(mode="after")
    def warn_extra_fields(self) -> "PreprocessingConfig":
        """Warn about unknown config keys."""
        if self.__pydantic_extra__:
            logger.warning(
                "Unknown config keys in PreprocessingConfig (ignored): %s",
                list(self.__pydantic_extra__.keys()),
            )
        return self
```

**Usage:** Config validation is already robust. No changes needed.

### Context Manager Pattern (Existing in AquaCal)

```python
# Source: Current pipeline.py usage (lines 1082-1104)
from aquacal.io.video import VideoSet

with VideoSet(camera_video_map) as videos:
    for frame_idx, raw_images in videos.iterate_frames(start=..., stop=..., step=...):
        process_frame(frame_idx, raw_images, ctx)
```

**Usage:** FrameSource Protocol should support same context manager interface.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Monolithic pipeline scripts | Modular pipeline packages with stage separation | Ongoing (2024-2025 ML/CV projects) | Better testability, reusability, maintainability |
| ABC for interfaces | `typing.Protocol` for structural subtyping | Python 3.8+ (2019), widespread adoption 2023+ | Lighter weight, better for duck typing, works with third-party code |
| Manual config validation | Pydantic v2 for typed config models | Pydantic 2.0 (2023) | Faster validation, better error messages, JSON schema generation |
| Inheritance-based extension | Composition + factory pattern | Long-standing Python idiom, emphasized in modern ML libraries | Simpler code, easier testing, less coupling |

**Current (2025-2026) CV/ML Pipeline Patterns:**
- **PyTorch Lightning:** Modular training pipeline with stages (training_step, validation_step) as methods
- **Hugging Face Transformers:** Factory functions (`AutoModel.from_pretrained`) + config-driven model selection
- **scikit-learn:** Pipeline composition via `Pipeline([("step1", obj1), ("step2", obj2)])` + fit/transform protocol
- **Kedro:** DAG-based pipelines with nodes as functions, catalog for I/O abstraction

**AquaMVS context:** Not building a training pipeline (Lightning), not building a generic ML framework (Kedro). Closest analogy is **scikit-learn's Pipeline** — sequential stages with config-driven composition.

**Deprecated/outdated:**
- Zope interfaces (`zope.interface`) — replaced by typing.Protocol
- ABC with `@abstractmethod` for simple interfaces — overkill when Protocol suffices
- Manual `warnings.warn()` for deprecations — Python 3.13+ has `warnings.deprecated()` decorator

## Open Questions

1. **Intermediate Result Exposure**
   - What we know: Pipeline needs to save depth maps, point clouds, meshes to disk (already implemented)
   - What's unclear: Should Pipeline class expose these as attributes (e.g., `pipeline.results.depth_maps`) or only via files?
   - Recommendation: **File-only for now.** Phase 3 focuses on decomposition, not API expansion. Exposing intermediate results as attributes adds complexity without clear use case. Can add in Phase 4 if documentation reveals user need.

2. **Sparse vs Full Mode Abstraction**
   - What we know: Four execution paths (lightglue+sparse, lightglue+full, roma+sparse, roma+full)
   - What's unclear: Should these be separate stage module files or conditionals within shared stage files?
   - Recommendation: **Separate stage modules for clarity.** Create `stages/sparse_matching.py` (LightGlue), `stages/dense_matching.py` (RoMa), `stages/depth_estimation.py` (plane sweep for LightGlue path), `stages/fusion.py` (always runs in full mode). Runner routes based on config. Easier to understand than deeply nested conditionals.

3. **AquaCal CalibrationData as CalibrationProvider?**
   - What we know: CalibrationData (from calibration.py) already provides cameras, water_z, n_water, etc.
   - What's unclear: Is a new Protocol needed or does existing CalibrationData satisfy it?
   - Recommendation: **CalibrationData already satisfies the interface.** Create CalibrationProvider Protocol that CalibrationData structurally conforms to. No code changes to CalibrationData needed. If future calibration sources (COLMAP) are added, they implement the same Protocol.

## Sources

### Primary (HIGH confidence)
- [PEP 544 – Protocols: Structural subtyping](https://peps.python.org/pep-0544/) - Official Python typing spec
- [Python Protocols: Leveraging Structural Subtyping – Real Python](https://realpython.com/python-protocol/) - Comprehensive guide with examples
- [Protocols and structural subtyping — typing documentation](https://typing.python.org/en/latest/reference/protocols.html) - Official typing documentation
- AquaMVS codebase (`src/aquamvs/projection/protocol.py`, `src/aquamvs/features/`, `src/aquamvs/config.py`) - Existing patterns

### Secondary (MEDIUM confidence)
- [The Elegance of Modular Data Processing with Python's Pipeline Approach | Medium](https://medium.com/@dkraczkowski/the-elegance-of-modular-data-processing-with-pythons-pipeline-approach-e63bec11d34f) - Pipeline pattern examples
- [Data Pipeline Design Patterns in Python – Start Data Engineering](https://www.startdataengineering.com/post/code-patterns/) - Factory + Strategy patterns
- [Python interfaces: abandon ABC and switch to Protocols | Medium](https://levelup.gitconnected.com/python-interfaces-choose-protocols-over-abc-3982e112342e) - Protocol vs ABC comparison
- [When Code Changes Break the World: Python Backward Compatibility in 2025](https://an4t.com/python-backward-compatibility/) - Deprecation strategies
- [pyDeprecate documentation](https://borda.github.io/pyDeprecate/) - Deprecation tooling (optional, may not need for clean break)

### Tertiary (LOW confidence, for context only)
- [Scikit-learn Pipelines Explained | Medium](https://medium.com/@sahin.samia/scikit-learn-pipelines-explained-streamline-and-optimize-your-machine-learning-processes-f17b1beb86a4) - scikit-learn pipeline patterns
- [Top Computer Vision Libraries OpenCV PyTorch](https://www.rapidinnovation.io/post/top-10-open-source-computer-vision-libraries-you-need-to-know) - Ecosystem overview

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Using existing project dependencies (Pydantic, PyTorch, typing.Protocol)
- Architecture patterns: HIGH - Patterns already proven in this codebase + standard Python practices
- Pitfalls: HIGH - Well-documented in Python refactoring literature + observed in scientific Python projects
- Code examples: HIGH - Extracted directly from current codebase
- Open questions: MEDIUM - Recommendations based on best practices but need validation during planning

**Research date:** 2026-02-14
**Valid until:** 2026-03-14 (30 days — stable domain, Python 3.10-3.13 features)

**Current codebase state:**
- 38 Python files in `src/aquamvs/`
- `pipeline.py`: 1124 lines (target for decomposition)
- Existing modular structure: `features/`, `dense/`, `projection/`, `visualization/` packages
- Pydantic config: 6 consolidated models (Phase 2 complete)
- Protocol example: `ProjectionModel` in `projection/protocol.py`
- Factory examples: `create_extractor()`, `create_matcher()` in `features/`
