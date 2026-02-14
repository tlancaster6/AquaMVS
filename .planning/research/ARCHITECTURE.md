# Architecture Research: Production-Ready Scientific Python Libraries

**Domain:** Scientific Python package (computer vision / 3D reconstruction)
**Researched:** 2026-02-14
**Confidence:** HIGH

## Standard Architecture

### Package Organization: The src/ Layout

Modern scientific Python packages (2026) strongly favor the **src/ layout** over flat layouts:

```
project-root/
├── src/
│   └── aquamvs/              # Package code
│       ├── __init__.py       # Public API definition with __all__
│       ├── cli.py            # CLI entry point
│       ├── config.py         # Configuration dataclasses
│       ├── core/             # Core algorithms (modular)
│       │   ├── __init__.py
│       │   ├── projection.py
│       │   ├── triangulation.py
│       │   └── fusion.py
│       ├── features/         # Feature extraction
│       │   ├── __init__.py
│       │   ├── lightglue.py
│       │   └── roma.py
│       ├── dense/            # Dense stereo
│       │   ├── __init__.py
│       │   ├── plane_sweep.py
│       │   └── roma_depth.py
│       ├── pipeline/         # Pipeline components (NOT monolith)
│       │   ├── __init__.py
│       │   ├── builder.py    # Constructs pipelines from config
│       │   ├── stages.py     # Reusable stage definitions
│       │   └── runner.py     # Execution orchestration
│       ├── io/               # Input/output boundaries
│       │   ├── __init__.py
│       │   ├── calibration.py
│       │   └── video.py      # VideoSet adapter (isolates AquaCal)
│       └── utils/            # Shared utilities
│           ├── __init__.py
│           ├── masks.py
│           └── visualization.py
├── tests/                    # Outside package (not distributed)
├── docs/                     # Sphinx/MkDocs documentation
├── pyproject.toml            # Modern build config
├── README.md
└── LICENSE

```

**Why src/ layout?**
- Tests run against installed package, not working directory files
- Catches import issues users will encounter
- Keeps package distribution clean (tests not in wheel)
- Industry standard for production packages since 2020

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| `cli.py` | Command-line interface, argument parsing | Typer (modern) or Click (mature) with type hints |
| `config.py` | Configuration dataclasses, validation | Pydantic dataclasses + YAML serialization |
| `pipeline/` | Pipeline construction and execution | Builder pattern + Strategy pattern for stages |
| `core/` | Domain algorithms (projection, fusion) | Pure functions or stateless classes |
| `features/` | Feature extraction backends | Strategy pattern with common interface |
| `dense/` | Dense stereo methods | Strategy pattern with common interface |
| `io/` | External dependencies boundary | Adapters to isolate third-party APIs |
| `utils/` | Shared functionality | Domain-agnostic helpers |

## Recommended Project Structure for AquaMVS

Based on ecosystem patterns from Kornia, Open3D, scikit-learn, and PyTorch:

```
src/aquamvs/
├── __init__.py               # Public API: expose key functions, not internals
├── cli.py                    # Typer-based CLI (migrate from argparse)
├── config.py                 # Pydantic models (migrate from plain dataclasses)
│
├── core/                     # Core algorithms (stable, tested)
│   ├── __init__.py
│   ├── projection/
│   │   ├── __init__.py
│   │   ├── protocol.py       # ProjectionModel interface
│   │   └── refractive.py     # RefractiveProjectionModel
│   ├── triangulation.py      # Ray triangulation
│   ├── fusion.py             # Depth map fusion
│   └── surface.py            # Surface reconstruction (Poisson)
│
├── features/                 # Feature extraction strategies
│   ├── __init__.py
│   ├── base.py               # FeatureExtractor protocol
│   ├── superpoint.py         # SuperPoint + LightGlue
│   ├── aliked.py             # ALIKED + LightGlue
│   ├── disk.py               # DISK + LightGlue
│   └── roma.py               # RoMa v2 dense matcher
│
├── dense/                    # Dense stereo strategies
│   ├── __init__.py
│   ├── base.py               # DenseStereo protocol
│   ├── plane_sweep.py        # Sparse-to-dense plane sweep
│   └── roma_depth.py         # RoMa warps to depth maps
│
├── pipeline/                 # Pipeline orchestration (REFACTORED)
│   ├── __init__.py
│   ├── stages.py             # Stage definitions (Calibration, Features, etc.)
│   ├── builder.py            # PipelineBuilder: config → stage graph
│   ├── runner.py             # PipelineRunner: executes stage graph
│   └── context.py            # PipelineContext: shared state between stages
│
├── io/                       # I/O boundary (isolates external deps)
│   ├── __init__.py
│   ├── calibration.py        # AquaCal calibration loading
│   ├── video.py              # VideoAdapter: wraps AquaCal VideoSet
│   ├── depth.py              # save_depth_map, load_depth_map
│   └── pointcloud.py         # save_point_cloud (Open3D wrapper)
│
├── visualization/            # Visualization utilities
│   ├── __init__.py
│   ├── depth.py              # Depth map viz
│   ├── matches.py            # Match viz
│   └── gallery.py            # Time-series gallery
│
└── utils/                    # Shared utilities
    ├── __init__.py
    ├── masks.py              # ROI masking
    ├── coloring.py           # Color normalization
    └── device.py             # Device management helpers

```

### Structure Rationale

- **Modular by concern:** `features/`, `dense/`, `core/` are independent. Can test/benchmark/replace each without touching others.
- **Isolated boundaries:** `io/` wraps AquaCal, Open3D. Easier to mock, test, swap implementations.
- **Pipeline decomposition:** 995-line `pipeline.py` → `pipeline/` package with separated concerns (building, running, stages).
- **Testability:** Clear interfaces (`base.py` protocols) enable unit testing without integration overhead.

## Architectural Patterns for AquaMVS Refactoring

### Pattern 1: Strategy Pattern for Feature Extractors

**What:** Define a common interface (Protocol) for feature extraction, with multiple implementations (SuperPoint, ALIKED, DISK, RoMa).

**When to use:** Multiple algorithms solve the same problem with the same inputs/outputs.

**Trade-offs:**
- **Pros:** Easy to benchmark, swap, extend. Clear contracts.
- **Cons:** Requires upfront interface design.

**Example:**
```python
from typing import Protocol
import torch

class FeatureExtractor(Protocol):
    """Protocol for feature extraction backends."""

    def extract(
        self,
        image: torch.Tensor,  # (B, C, H, W)
        device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract keypoints and descriptors.

        Returns:
            keypoints: (B, N, 2) in (u, v) pixel coords
            descriptors: (B, N, D) feature descriptors
        """
        ...

class SuperPointExtractor:
    def __init__(self, max_keypoints: int = 2048, threshold: float = 0.005):
        self.max_keypoints = max_keypoints
        self.threshold = threshold
        self._model = None  # Lazy load

    def extract(self, image: torch.Tensor, device: torch.device) -> tuple:
        # Implementation
        ...

# Config-driven selection
def get_extractor(config: FeatureExtractionConfig) -> FeatureExtractor:
    if config.extractor_type == "superpoint":
        return SuperPointExtractor(config.max_keypoints, config.detection_threshold)
    elif config.extractor_type == "aliked":
        return AlikedExtractor(config.max_keypoints)
    # ...
```

### Pattern 2: Builder Pattern for Pipeline Construction

**What:** Separate pipeline construction (config → stages) from execution (run stages). Builder reads config and assembles the pipeline; runner executes it.

**When to use:** Complex object construction with many configuration options. Need to support multiple execution modes (lightglue+sparse, roma+full, etc.).

**Trade-offs:**
- **Pros:** Flexible, testable, supports validation before execution. Clear separation of concerns.
- **Cons:** More classes than monolithic approach.

**Example:**
```python
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class PipelineStage:
    """Represents a single stage in the pipeline."""
    name: str
    function: Callable[[PipelineContext], None]
    enabled: bool = True

@dataclass
class PipelineContext:
    """Shared state between pipeline stages."""
    config: PipelineConfig
    calibration: CalibrationData
    video_adapter: VideoAdapter
    frame_idx: int
    # Stage outputs
    features: dict[str, Any] | None = None
    depth_maps: dict[str, Any] | None = None
    fused_cloud: o3d.geometry.PointCloud | None = None

class PipelineBuilder:
    """Constructs pipeline from configuration."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stages: list[PipelineStage] = []

    def build(self) -> list[PipelineStage]:
        """Construct the pipeline based on config."""
        self.stages = []

        # Core stages (always enabled)
        self.add_stage("load_calibration", self._load_calibration)
        self.add_stage("setup_video", self._setup_video)
        self.add_stage("extract_features", self._extract_features)

        # Conditional stages based on config
        if self.config.mode in ["lightglue+full", "roma+full"]:
            self.add_stage("dense_stereo", self._dense_stereo)
            self.add_stage("fuse_depth", self._fuse_depth)

        if self.config.surface.enabled:
            self.add_stage("reconstruct_surface", self._reconstruct_surface)

        if self.config.visualization.enabled:
            self.add_stage("visualize", self._visualize)

        return self.stages

    def add_stage(self, name: str, function: Callable):
        self.stages.append(PipelineStage(name, function))

    def _load_calibration(self, ctx: PipelineContext):
        ctx.calibration = load_calibration_data(self.config.calibration_path)

    # ... other stage functions

class PipelineRunner:
    """Executes a constructed pipeline."""

    def __init__(self, stages: list[PipelineStage]):
        self.stages = stages

    def run(self, context: PipelineContext):
        """Execute all enabled stages."""
        for stage in self.stages:
            if stage.enabled:
                logger.info(f"Running stage: {stage.name}")
                stage.function(context)

# Usage
config = PipelineConfig.from_yaml("config.yaml")
builder = PipelineBuilder(config)
stages = builder.build()
runner = PipelineRunner(stages)
context = PipelineContext(config=config, ...)
runner.run(context)
```

### Pattern 3: Adapter Pattern for External Dependencies

**What:** Wrap external APIs (AquaCal VideoSet, Open3D) behind internal interfaces. Isolates third-party dependencies to specific modules.

**When to use:** External API is complex, unstable, or might be swapped. Want to test without external dependency.

**Trade-offs:**
- **Pros:** Decouples codebase from external changes. Easy to mock for testing.
- **Cons:** Extra layer of indirection.

**Example:**
```python
# io/video.py
from aquacal.io.video import VideoSet as AquaCalVideoSet

class VideoAdapter:
    """Adapter for AquaCal VideoSet. Isolates external dependency."""

    def __init__(self, video_paths: dict[str, Path]):
        self._videoset = AquaCalVideoSet(video_paths)

    def read_frame(self, camera_name: str, frame_idx: int) -> np.ndarray:
        """Read a single frame from a camera.

        Returns:
            image: (H, W, 3) RGB image [0, 255] uint8
        """
        return self._videoset.read(camera_name, frame_idx)

    def get_frame_count(self, camera_name: str) -> int:
        return self._videoset.frame_count(camera_name)

    def close(self):
        self._videoset.close()

# Now pipeline.py doesn't import from aquacal directly
# video_adapter = VideoAdapter(video_paths)
# img = video_adapter.read_frame("e3v82e0", 0)
```

### Pattern 4: Facade Pattern for Complex Subsystems

**What:** Provide a simplified interface to a complex subsystem. Hides internal complexity behind a clean API.

**When to use:** Subsystem has many moving parts (undistortion, color normalization, ROI masking). Users need simple "just do it" function.

**Trade-offs:**
- **Pros:** Simple API for common cases. Internal complexity hidden.
- **Cons:** Advanced users might need to bypass facade.

**Example:**
```python
# core/preprocessing.py
class ImagePreprocessor:
    """Facade for image preprocessing pipeline."""

    def __init__(
        self,
        calibration: CalibrationData,
        color_norm_config: ColorNormConfig | None = None,
        masks: dict[str, np.ndarray] | None = None
    ):
        self.calibration = calibration
        self.color_norm_config = color_norm_config
        self.masks = masks
        self._undistortion_maps = compute_undistortion_maps(calibration)

    def preprocess(
        self,
        image: np.ndarray,
        camera_name: str
    ) -> torch.Tensor:
        """One-stop preprocessing: undistort → normalize → mask → tensor.

        Args:
            image: (H, W, 3) RGB [0, 255] uint8
            camera_name: Camera identifier

        Returns:
            (1, 3, H', W') float32 tensor [0, 1], preprocessed
        """
        # Undistort
        img = undistort_image(image, self._undistortion_maps[camera_name])

        # Color normalization
        if self.color_norm_config and self.color_norm_config.enabled:
            img = normalize_color(img, self.color_norm_config.method)

        # Apply mask
        if self.masks and camera_name in self.masks:
            img = apply_mask(img, self.masks[camera_name])

        # To tensor
        return image_to_tensor(img)

# Usage (simple!)
preprocessor = ImagePreprocessor(calibration, color_config, masks)
tensor = preprocessor.preprocess(raw_image, "e3v82e0")
```

## Data Flow

### Current (Monolithic Pipeline)

```
config.yaml → PipelineConfig
                ↓
            pipeline.py (995 lines)
    ┌───────────┴───────────────────────┐
    │  1. Load calibration              │
    │  2. Setup video                    │
    │  3. For each frame:               │
    │     - Undistort all images        │
    │     - Extract features            │
    │     - Match pairs                 │
    │     - Triangulate                 │
    │     - (If full) Dense stereo      │
    │     - (If full) Fuse depth        │
    │     - (If enabled) Surface        │
    │     - (If enabled) Visualize      │
    │  4. (If enabled) Gallery          │
    └───────────────────────────────────┘
                ↓
            output_dir/
```

**Problems:**
- Single 995-line function with deep nesting
- Hard to test individual stages
- Difficult to add/remove/reorder stages
- Config proliferation (9+ dataclasses)
- External dependency (VideoSet) leaked throughout

### Proposed (Modular Pipeline)

```
config.yaml → Pydantic Validation → PipelineConfig
                                          ↓
                                   PipelineBuilder
                                          ↓
                            [Stage1, Stage2, ...]  ← Stage definitions
                                          ↓
                                   PipelineRunner
                                          ↓
                                   PipelineContext (shared state)
                ┌────────────────────────┴────────────────────┐
                │                                              │
          Core Algorithms                            I/O Boundaries
    ┌──────────┴──────────┐                   ┌────────┴────────┐
    │ • Projection        │                   │ • VideoAdapter  │
    │ • Triangulation     │                   │ • save_depth    │
    │ • Fusion            │                   │ • save_cloud    │
    └─────────────────────┘                   └─────────────────┘
                │                                      │
          Feature/Dense                           Visualization
          Strategies                                  Utils
    ┌──────────┴──────────┐                   ┌────────┴────────┐
    │ • SuperPoint        │                   │ • depth viz     │
    │ • RoMa              │                   │ • gallery       │
    │ • Plane sweep       │                   │ • matches       │
    └─────────────────────┘                   └─────────────────┘
                                   ↓
                            output_dir/
```

**Benefits:**
- Each stage independently testable
- Clear data dependencies via PipelineContext
- Easy to add/remove/reorder stages
- External deps isolated to io/
- Strategy pattern for swappable algorithms

## Refactoring Roadmap: Build Order

Based on dependency analysis, refactor in this order:

### Phase 1: Isolate Boundaries (No Breaking Changes)
**Goal:** Extract I/O and external dependencies without changing existing API.

1. **Create `io/` package**
   - Move calibration loading → `io/calibration.py`
   - Create `VideoAdapter` → `io/video.py` (wraps AquaCal)
   - Move depth I/O → `io/depth.py`
   - Move point cloud I/O → `io/pointcloud.py`

2. **Create `visualization/` package**
   - Extract viz functions from `pipeline.py` → `visualization/`

**Dependencies:** None. Pure extraction.
**Risk:** Low. No API changes.

### Phase 2: Introduce Protocols (Backward Compatible)
**Goal:** Define interfaces for strategies without breaking existing code.

3. **Create strategy protocols**
   - `features/base.py`: FeatureExtractor protocol
   - `dense/base.py`: DenseStereo protocol
   - Existing implementations conform to protocols

4. **Add factory functions**
   - `get_feature_extractor(config) -> FeatureExtractor`
   - `get_dense_stereo(config) -> DenseStereo`

**Dependencies:** Phase 1 complete.
**Risk:** Low. Existing code still works.

### Phase 3: Decompose Pipeline (Breaking Change)
**Goal:** Replace monolithic `pipeline.py` with modular `pipeline/` package.

5. **Create `pipeline/` package**
   - `stages.py`: Extract stage functions from `pipeline.py`
   - `context.py`: PipelineContext dataclass
   - `builder.py`: PipelineBuilder class
   - `runner.py`: PipelineRunner class

6. **Migrate `pipeline.py` to use new architecture**
   - Keep old `run_pipeline()` as facade calling new architecture
   - Mark as deprecated

**Dependencies:** Phases 1-2 complete.
**Risk:** Medium. Changes orchestration but preserves facade.

### Phase 4: Migrate Configuration (Breaking Change)
**Goal:** Consolidate config, add validation, improve UX.

7. **Migrate to Pydantic**
   - Convert dataclasses → Pydantic models
   - Add field validators
   - Support environment variables

8. **Simplify config structure**
   - Merge related configs (e.g., FeatureExtractionConfig + MatchingConfig → FeatureConfig)
   - Reduce from 9 dataclasses → ~4-5

**Dependencies:** Phase 3 complete (pipeline needs config).
**Risk:** High. Breaks existing config files. Requires migration guide.

### Phase 5: Modernize CLI (Breaking Change)
**Goal:** Improve CLI UX, add type hints, better error messages.

9. **Migrate CLI to Typer**
   - Replace argparse → Typer
   - Add type hints for auto-completion
   - Rich output formatting

**Dependencies:** Phase 4 complete (CLI needs new config).
**Risk:** Medium. CLI API changes but usage similar.

### Phase 6: Polish Public API
**Goal:** Clean up `__init__.py`, ensure stable API for users.

10. **Define public API**
    - `src/aquamvs/__init__.py`: Explicit `__all__`
    - Prefix internals with `_`
    - Version stability guarantees

**Dependencies:** Phases 1-5 complete.
**Risk:** Low. Clarifies, doesn't change behavior.

## Configuration Management: Best Practices

### Current State (Plain Dataclasses + YAML)
```python
@dataclass
class FeatureExtractionConfig:
    extractor_type: str = "superpoint"
    max_keypoints: int = 2048
    detection_threshold: float = 0.005
    # No validation!
```

**Problems:**
- No validation (e.g., `max_keypoints = -1` accepted)
- No type coercion (YAML strings → expected types)
- Manual YAML serialization
- No environment variable support

### Recommended: Pydantic + YAML

```python
from pydantic import BaseModel, Field, field_validator
from pathlib import Path

class FeatureConfig(BaseModel):
    """Feature extraction and matching configuration."""

    extractor_type: Literal["superpoint", "aliked", "disk", "roma"] = Field(
        default="superpoint",
        description="Feature extraction backend"
    )
    max_keypoints: int = Field(
        default=2048,
        ge=64,
        le=8192,
        description="Maximum keypoints to extract"
    )
    detection_threshold: float = Field(
        default=0.005,
        ge=0.0,
        le=1.0,
        description="Detection confidence threshold"
    )

    @field_validator('extractor_type')
    def validate_extractor(cls, v):
        if v not in ["superpoint", "aliked", "disk", "roma"]:
            raise ValueError(f"Unknown extractor: {v}")
        return v

    model_config = ConfigDict(
        extra='forbid',  # Reject unknown fields
        frozen=False,    # Allow mutation after creation
    )

class PipelineConfig(BaseModel):
    """Root configuration for AquaMVS pipeline."""

    calibration_path: Path = Field(description="Path to AquaCal JSON")
    output_dir: Path = Field(description="Output directory")
    video_paths: dict[str, Path] = Field(description="Camera name → video path")

    features: FeatureConfig = Field(default_factory=FeatureConfig)
    dense: DenseConfig = Field(default_factory=DenseConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load and validate config from YAML."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: Path):
        """Save config to YAML."""
        with open(path, 'w') as f:
            yaml.safe_dump(self.model_dump(), f)
```

**Benefits:**
- Automatic validation (types, ranges, constraints)
- Clear error messages ("max_keypoints must be >= 64")
- YAML serialization via `model_dump()`
- Environment variable support (`Field(..., env="AQUAMVS_CALIBRATION")`)
- JSON schema generation for editors/docs

### Alternative: Hydra + OmegaConf (For Complex Projects)

For projects with many config combinations (hyperparameter sweeps, multi-run experiments):

```python
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Hydra handles YAML loading, CLI overrides, logging
    print(OmegaConf.to_yaml(cfg))
    # Run pipeline
```

**When to use Hydra:**
- Running hyperparameter sweeps
- Complex config composition (base + overrides)
- Multi-run experiments
- Need structured logging

**When NOT to use Hydra:**
- Simple config (AquaMVS current state)
- Library (not application) — Hydra is app-focused
- Users want simple YAML, not Hydra DSL

**Recommendation for AquaMVS:** Pydantic. Hydra is overkill for current needs, but consider if adding experiment tracking later.

## CLI Design: Best Practices

### Current State (argparse)
```python
parser = argparse.ArgumentParser()
parser.add_argument("--video-dir", type=Path, required=True)
parser.add_argument("--pattern", type=str, required=True)
# ... many arguments
args = parser.parse_args()
```

**Problems:**
- Verbose (lots of boilerplate)
- No type hints → no IDE autocompletion
- Manual type conversion
- Error messages not user-friendly

### Recommended: Typer (Modern, Type-Hinted)

```python
import typer
from typing_extensions import Annotated
from pathlib import Path

app = typer.Typer(help="AquaMVS: Refractive multi-view stereo reconstruction")

@app.command()
def init(
    video_dir: Annotated[Path, typer.Option(help="Directory containing video files")],
    pattern: Annotated[str, typer.Option(help="Regex pattern to extract camera names")],
    calibration: Annotated[Path, typer.Option(help="Path to AquaCal calibration JSON")],
    output_dir: Annotated[Path, typer.Option(help="Output directory for results")],
    config_path: Annotated[Path, typer.Option(help="Path to save generated config")] = Path("config.yaml"),
):
    """Generate a pipeline config from video directory and calibration."""
    # Implementation (same as current)
    ...
    typer.echo(f"Config written to {config_path}")

@app.command()
def run(
    config_path: Annotated[Path, typer.Argument(help="Path to pipeline config YAML")],
    frame: Annotated[int | None, typer.Option(help="Process single frame (default: all)")] = None,
):
    """Run the reconstruction pipeline."""
    config = PipelineConfig.from_yaml(config_path)
    # Build and run pipeline
    ...

if __name__ == "__main__":
    app()
```

**Benefits:**
- Type hints → automatic validation and IDE support
- Less boilerplate (50% fewer lines vs argparse)
- Beautiful help messages (auto-generated from docstrings)
- Subcommands naturally map to functions
- Rich formatting support (colors, progress bars)

**Migration cost:** Low. 1-2 hours to migrate existing argparse → Typer.

### Alternative: Click (Mature, Flexible)

```python
import click

@click.group()
def cli():
    """AquaMVS: Refractive multi-view stereo reconstruction"""
    pass

@cli.command()
@click.option("--video-dir", type=click.Path(exists=True), required=True)
@click.option("--pattern", type=str, required=True)
def init(video_dir, pattern):
    """Generate a pipeline config."""
    ...
```

**Comparison:**

| Feature | argparse | Click | Typer |
|---------|----------|-------|-------|
| Type hints | No | No | Yes |
| Boilerplate | High | Medium | Low |
| Subcommands | Manual | Decorators | Decorators |
| Validation | Manual | Built-in | Automatic (via types) |
| IDE support | No | Limited | Excellent |
| Dependencies | Stdlib | Click | Typer (+ Click underneath) |

**Recommendation for AquaMVS:** Typer. Modern, type-safe, minimal code. Click is fine if you prefer decorator style over type hints.

## Documentation Structure: Best Practices

### Structure for Computer Vision Libraries

Based on OpenCV, Kornia, Open3D:

```
docs/
├── index.md                    # Landing page
├── getting-started/
│   ├── installation.md         # pip install, dependencies
│   ├── quickstart.md           # 5-minute example
│   └── concepts.md             # Core concepts (rays, depth, refraction)
├── tutorials/
│   ├── basic-pipeline.md       # End-to-end example
│   ├── custom-features.md      # Using different extractors
│   ├── config-guide.md         # Config YAML reference
│   └── benchmark.md            # Benchmarking extractors
├── how-to/
│   ├── draw-roi-masks.md       # Practical task guides
│   ├── choose-depth-range.md
│   └── tune-fusion.md
├── api-reference/
│   ├── core.md                 # Auto-generated from docstrings
│   ├── features.md
│   ├── dense.md
│   ├── pipeline.md
│   └── config.md
├── explanation/
│   ├── refractive-geometry.md  # Deep dives
│   ├── plane-sweep.md
│   └── fusion-algorithm.md
└── development/
    ├── contributing.md
    ├── testing.md
    └── architecture.md          # This research doc!
```

**Documentation pyramid (Divio framework):**
1. **Tutorials** (learning-oriented): Step-by-step lessons for beginners
2. **How-to guides** (task-oriented): Solve specific problems
3. **Explanation** (understanding-oriented): Clarify concepts, discuss design
4. **Reference** (information-oriented): API docs, complete coverage

### Tools

| Tool | Use Case | Pros | Cons |
|------|----------|------|------|
| **Sphinx** | Scientific Python standard | NumPy/SciPy use it, autodoc, math | Complex config, ReStructuredText |
| **MkDocs** | Modern, Markdown-first | Simple, beautiful themes, fast | Less autodoc than Sphinx |
| **MkDocs + mkdocstrings** | Best of both worlds | Markdown + autodoc from docstrings | Needs plugin |

**Recommendation for AquaMVS:** MkDocs + mkdocstrings (Material theme). Markdown is easier than RST, autodoc plugin provides API reference, Material theme is gorgeous.

Example `mkdocs.yml`:
```yaml
site_name: AquaMVS
theme:
  name: material
  features:
    - navigation.tabs
    - toc.integrate
    - search.highlight

nav:
  - Home: index.md
  - Getting Started:
      - Installation: getting-started/installation.md
      - Quickstart: getting-started/quickstart.md
  - Tutorials:
      - Basic Pipeline: tutorials/basic-pipeline.md
  - API Reference:
      - Core: api-reference/core.md

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: google
            show_source: true
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: The 1000-Line Orchestrator

**What people do:** All pipeline logic in one giant function (`pipeline.py`: 995 lines).

**Why it's wrong:**
- Impossible to unit test individual stages
- Hard to debug (which of 50 steps failed?)
- Can't reuse/reorder/skip stages
- Every feature adds nesting depth

**Do this instead:**
- Builder pattern: construct pipeline from config
- Stage pattern: each stage is a function with clear inputs/outputs
- Context pattern: shared state between stages
- **Target:** No function > 50 lines, no file > 300 lines

### Anti-Pattern 2: Config Sprawl Without Validation

**What people do:** 9+ dataclasses, no validation, manual YAML parsing.

**Why it's wrong:**
- Typos cause runtime errors ("extractor_typ" → AttributeError)
- Invalid values accepted (`max_keypoints: -1`)
- No documentation of constraints
- Config evolution breaks backward compatibility

**Do this instead:**
- Pydantic models with validators
- Field descriptions for auto-docs
- Semantic versioning for config schema
- **Target:** 4-5 config classes max, all validated

### Anti-Pattern 3: Leaking External Dependencies

**What people do:** Import `aquacal.io.video.VideoSet` throughout codebase.

**Why it's wrong:**
- Tight coupling → hard to swap/mock
- External API changes break many files
- Testing requires full external dependency

**Do this instead:**
- Adapter pattern: `io/video.py` wraps VideoSet
- Only `io/video.py` imports from aquacal
- Rest of codebase uses adapter interface
- **Target:** External imports confined to `io/` package

### Anti-Pattern 4: God Class Pipeline

**What people do:** Pipeline class with 20+ methods, 500+ lines.

**Why it's wrong:**
- Violates Single Responsibility Principle
- Hard to test (need to mock entire class)
- Every method has access to all state

**Do this instead:**
- Composition over inheritance
- Small, focused classes (FeatureExtractor, DenseStereo, etc.)
- Pipeline assembles components, doesn't implement them
- **Target:** Classes < 200 lines, < 10 public methods

### Anti-Pattern 5: Implicit Public API

**What people do:** No `__all__`, everything imported, no `_` prefixes.

**Why it's wrong:**
- Users import internal functions
- Refactoring breaks user code
- No distinction between stable/unstable API

**Do this instead:**
- Explicit `__all__` in `__init__.py`
- Prefix internals with `_`
- Document public API stability
- **Target:** `from aquamvs import *` imports only stable API

### Anti-Pattern 6: Flat Package (Everything at Top Level)

**What people do:**
```
aquamvs/
├── projection_refractive.py
├── projection_protocol.py
├── features_superpoint.py
├── features_aliked.py
├── dense_plane_sweep.py
├── ...  # 36 files at top level
```

**Why it's wrong:**
- Hard to navigate
- No logical grouping
- `from aquamvs import ...` imports 36 modules

**Do this instead:**
- Group by concern: `core/`, `features/`, `dense/`, `io/`
- Each subpackage has `__init__.py` with `__all__`
- Top-level `__init__.py` re-exports key items
- **Target:** < 10 files at package root, rest in subpackages

## Module Organization: Advanced Patterns

### Pattern: Private Submodules for Implementation Details

```
aquamvs/
├── __init__.py          # Public API
├── projection/
│   ├── __init__.py      # Public: ProjectionModel, RefractiveProjectionModel
│   ├── protocol.py      # Public: ProjectionModel interface
│   ├── refractive.py    # Public: RefractiveProjectionModel
│   └── _snells.py       # Private: low-level Snell's law ops
```

`projection/__init__.py`:
```python
"""Projection models for camera geometry."""

from .protocol import ProjectionModel
from .refractive import RefractiveProjectionModel

__all__ = ["ProjectionModel", "RefractiveProjectionModel"]
# Note: _snells is NOT exported (private implementation detail)
```

**Rationale:** `_snells.py` is internal. Users shouldn't import it directly. If we refactor (e.g., merge into `refractive.py`), no breaking change.

### Pattern: Facade in Top-Level `__init__.py`

```python
# src/aquamvs/__init__.py
"""AquaMVS: Refractive multi-view stereo reconstruction."""

from .config import PipelineConfig
from .pipeline.builder import PipelineBuilder
from .pipeline.runner import PipelineRunner
from .core.projection import RefractiveProjectionModel
from .io.calibration import load_calibration_data

__all__ = [
    # Config
    "PipelineConfig",
    # Pipeline
    "PipelineBuilder",
    "PipelineRunner",
    # Core (selected, not all)
    "RefractiveProjectionModel",
    # I/O (selected)
    "load_calibration_data",
]

__version__ = "0.2.0"
```

**Rationale:** Users can do `from aquamvs import PipelineBuilder` (simple imports). Advanced users can do `from aquamvs.core.projection import ...` (submodules). Best of both worlds.

## Scaling Considerations

AquaMVS is a scientific library, not a web service, so "scaling" means:
- **Data size:** Handle large videos (10K+ frames)
- **Camera count:** Support 12+ cameras
- **Performance:** GPU acceleration, batching
- **Extensibility:** Easy to add new extractors/methods

| Scale | Architecture Adjustments |
|-------|--------------------------|
| **0-100 frames, 1-5 cameras** | Current architecture works. Single-frame processing, CPU is fine. |
| **100-1000 frames, 5-12 cameras** | Add batching (process multiple frames in parallel). GPU for feature extraction. Checkpoint intermediate outputs. |
| **1000+ frames, 12+ cameras** | Distributed processing (Dask, Ray). Incremental fusion (don't hold all depth maps in memory). Consider video compression for storage. |

### Scaling Priorities

1. **First bottleneck:** Feature extraction (CPU-bound). **Fix:** GPU acceleration (PyTorch/Kornia).
2. **Second bottleneck:** Depth map fusion (memory-bound). **Fix:** Incremental fusion, chunked processing.
3. **Third bottleneck:** I/O (disk-bound). **Fix:** Parallel video reading, SSD storage.

**Current AquaMVS:** Optimize for 100-1000 frames, 12 cameras. Don't prematurely optimize for 10K+ frames.

## Integration Points

### External Services (AquaMVS Context)

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| **AquaCal** | Adapter (`io/calibration.py`, `io/video.py`) | Import only in `io/`. Rest of codebase uses adapters. |
| **Open3D** | Adapter (`io/pointcloud.py`, `core/surface.py`) | Wrap save/load functions. Avoids direct imports elsewhere. |
| **LightGlue** | Strategy (`features/superpoint.py`) | Encapsulated in FeatureExtractor. Easy to swap. |
| **RoMa v2** | Strategy (`features/roma.py`, `dense/roma_depth.py`) | Isolated. Can disable via config. |
| **PyTorch/Kornia** | Direct use (core algorithms) | Acceptable — these are foundational, unlikely to change. |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| **pipeline/ ↔ core/** | Function calls via PipelineContext | Pipeline calls core functions. Core has no knowledge of pipeline. |
| **pipeline/ ↔ features/** | Strategy interface (FeatureExtractor) | Pipeline gets extractor from factory, calls `.extract()`. |
| **pipeline/ ↔ io/** | Function calls (load/save) | Pipeline calls I/O functions. I/O has no knowledge of pipeline. |
| **core/ ↔ features/** | No direct communication | Independent. Features produce data consumed by core via pipeline. |

**Dependency direction:** Always toward core. Pipeline depends on core, not vice versa.

```
        CLI
         ↓
      pipeline/  ← config
         ↓
    ┌────┴─────┬─────────┐
    ↓          ↓         ↓
 features/   dense/     core/
    ↓          ↓         ↓
    └──────────┴────> io/
```

## Sources

### Package Structure & Best Practices
- [Python Package Structure & Layout - pyOpenSci](https://www.pyopensci.org/python-package-guide/package-structure-code/python-package-structure.html)
- [Scientific Python Development Guide](https://learn.scientific-python.org/development/)
- [Best practices for configurations in Python-based pipelines - Micropole](https://belux.micropole.com/blog/python/blog-best-practices-for-configurations-in-python-based-pipelines/)
- [Best Practices for Working with Configuration in Python Applications - Preferred Networks](https://tech.preferred.jp/en/blog/working-with-configuration-in-python/)

### CLI Design
- [Comparing Python Command Line Interface Tools: Argparse, Click, and Typer - CodeCut](https://codecut.ai/comparing-python-command-line-interface-tools-argparse-click-and-typer/)
- [Navigating the CLI Landscape in Python - Medium](https://medium.com/@mohd_nass/navigating-the-cli-landscape-in-python-a-comparative-study-of-argparse-click-and-typer-480ebbb7172f)
- [Typer Documentation - Alternatives and Comparisons](https://typer.tiangolo.com/alternatives/)

### Configuration Management
- [Pydantic And Hydra - Omniverse](https://www.gaohongnan.com/software_engineering/config_management/01-pydra.html)
- [Configuration management for model training experiments using Pydantic and Hydra - Towards Data Science](https://towardsdatascience.com/configuration-management-for-model-training-experiments-using-pydantic-and-hydra-d14a6ae84c13/)
- [yamldataclassconfig - PyPI](https://pypi.org/project/yamldataclassconfig/)

### Pipeline Patterns
- [Data Pipeline Design Patterns - Start Data Engineering](https://www.startdataengineering.com/post/code-patterns/)
- [A Practical Example Of The Pipeline Pattern In Python - Pybites](https://pybit.es/articles/a-practical-example-of-the-pipeline-pattern-in-python/)
- [Strategy Design Pattern for Effective ML Pipeline - Medium](https://medium.com/mlearning-ai/strategy-design-pattern-for-effective-ml-pipeline-1099c5131553)
- [Coding Data Pipeline Design Patterns in Python - Medium](https://amsayed.medium.com/coding-data-pipeline-design-patterns-in-python-44a705f0af9e)

### Package Organization
- [How to Structure Python Projects - Dagster](https://dagster.io/blog/python-project-best-practices)
- [scikit-learn Project Structure - DeepWiki](https://deepwiki.com/scikit-learn/scikit-learn/1.2-project-structure)
- [Exploring Scikit-Learn Sub-Packages and Modules - Medium](https://mohamed-stifi.medium.com/exploring-scikit-learn-a-comprehensive-overview-of-its-sub-packages-and-modules-032bea32a65f)

### API Design
- [Python Private Function Coding Conventions - Py4u](https://www.py4u.org/blog/python-private-function-coding-convention/)
- [Public API Surface - Real Python](https://realpython.com/ref/best-practices/public-api-surface/)
- [Designing Pythonic library APIs - Ben Hoyt](https://benhoyt.com/writings/python-api-design/)
- [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/)

### Library Examples
- [Kornia Documentation](https://kornia.readthedocs.io/)
- [Kornia: Differentiable Computer Vision Library for PyTorch - ArXiv](https://ar5iv.labs.arxiv.org/html/1910.02190)
- [Open3D GitHub Repository](https://github.com/isl-org/Open3D)
- [Open3D Python Interface Documentation](https://www.open3d.org/docs/release/tutorial/geometry/python_interface.html)
- [PyTorch Modules Documentation](https://docs.pytorch.org/docs/stable/notes/modules.html)

### Anti-Patterns
- [The Little Book of Python Anti-Patterns](https://docs.quantifiedcode.com/python-anti-patterns/)
- [Python Anti-Patterns - charlax/antipatterns](https://github.com/charlax/antipatterns/blob/master/python-antipatterns.md)
- [Python Deployment Anti-Patterns - Hynek Schlawack](https://hynek.me/articles/python-deployment-anti-patterns/)

---
*Architecture research for: Production-ready scientific Python package (AquaMVS)*
*Researched: 2026-02-14*
