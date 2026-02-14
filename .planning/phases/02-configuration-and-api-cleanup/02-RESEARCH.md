# Phase 02: Configuration and API Cleanup - Research

**Researched:** 2026-02-14
**Domain:** Configuration validation and schema management with Pydantic v2
**Confidence:** HIGH

## Summary

Phase 2 consolidates AquaMVS's 14-dataclass configuration system into ~5 validated Pydantic models grouped by pipeline stage. The research confirms that Pydantic v2 (latest stable: 2.12.5) is the industry-standard solution for this use case, providing automatic error collection, field path reporting, YAML integration, and cross-field validation through `@model_validator` decorators. The current system uses standard Python dataclasses with manual validation in a single `validate()` method, which fails on first error and provides poor error messages.

Key findings: (1) Pydantic v2 automatically collects all validation errors before raising a single exception with structured error details including field paths; (2) `extra='allow'` with manual inspection provides forward-compatible unknown-key warnings; (3) `@model_validator(mode='after')` enables cross-stage constraint validation (e.g., matcher_type=roma requires dense_matching config); (4) tqdm provides automatic TTY detection and Jupyter support with zero configuration; (5) Field(default_factory=...) provides sensible defaults with logging of applied values.

**Primary recommendation:** Migrate to Pydantic v2 BaseModel with ConfigDict(extra='allow'), use @model_validator for cross-field validation, add tqdm to existing dependencies, implement dotted-path error formatting for user-friendly messages.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Config grouping:**
- Group by pipeline stage, not by user concern
- Dual-pathway architecture reflected in config: separate SparseMatchingConfig (feature extraction, pair selection, LightGlue matching) and DenseMatchingConfig (RoMa)
- Cross-cutting concerns (device, output, visualization, benchmark) grouped into a single RuntimeConfig section
- Preprocessing concerns (color normalization, frame sampling) grouped into PreprocessingConfig
- Resulting top-level structure: ~5 groups (Preprocessing, SparseMatching, DenseMatching, Reconstruction [stereo+fusion+surface], Runtime)

**Validation behavior:**
- Collect all validation errors and report them together (not fail-on-first)
- Unknown/extra YAML keys produce a warning, not an error (forwards-compatible)
- Cross-stage constraints (e.g., matcher_type=roma requires dense_matching settings) validated at load time, before any processing starts
- Error messages use YAML paths (e.g., `dense_stereo.num_depths: must be > 0`) so user knows exactly which field to fix

**Minimal config UX:**
- Minimum required fields: paths (video_dir, calibration_file, output_dir) + matcher_type (lightglue vs roma)
- `aquamvs init` generates a full annotated config with all fields and comments showing defaults
- When loading a config with missing optional sections, log applied defaults at INFO level (e.g., "Using default: dense_stereo.num_depths=128")

**Progress reporting:**
- Progress bars on slow operations only: plane sweep stereo, depth fusion, pair matching
- Use tqdm (lightweight, works in terminals and Jupyter)
- Progress bars on by default in CLI; suppressible with --quiet
- On by default with INFO-level log showing which defaults were applied

### Claude's Discretion

- CLI override mechanism for config values (dotted overrides vs YAML-only) — decide based on implementation complexity vs value
- Progress bar suppression in non-TTY/library contexts — decide between auto-detection and config control
- Exact Pydantic model field names and nesting depth within each group
- Evaluation config placement (could be Reconstruction or Runtime)

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope

</user_constraints>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pydantic | >=2.12.0 | Config validation and schema definition | Industry standard for Python data validation; 100M+ downloads/month; automatic error collection, JSON schema generation, and type coercion |
| tqdm | >=4.66.0 | Progress bars | Ubiquitous progress bar library; 300M+ downloads/month; zero-config TTY detection and Jupyter support |
| pyyaml | >=6.0 (existing) | YAML parsing | Already in dependencies; standard YAML library for Python |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pydantic-settings | >=2.5.0 (optional) | Environment variable and multi-source config loading | Only if CLI override mechanism via env vars is desired (Claude's discretion) |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Pydantic v2 | attrs + cattrs | attrs is faster but requires manual validation logic; no built-in error collection; not standard for config validation |
| Pydantic v2 | Marshmallow | Older paradigm (serialize/deserialize); less type-safe; declining popularity vs Pydantic |
| tqdm | rich.progress | rich is more feature-rich but heavier dependency (terminal rendering library); tqdm is minimal and focused |

**Installation:**
```bash
# Add to pyproject.toml dependencies
pydantic>=2.12.0
tqdm>=4.66.0
# pyyaml already present
```

## Architecture Patterns

### Recommended Project Structure

Current structure (14 dataclasses in single file):
```
src/aquamvs/
└── config.py  # 492 lines, 14 dataclasses
```

Consolidated structure (~5 Pydantic models in single file is acceptable):
```
src/aquamvs/
└── config.py  # ~400-500 lines, 5-6 BaseModel classes
    ├── PreprocessingConfig
    ├── SparseMatchingConfig
    ├── DenseMatchingConfig
    ├── ReconstructionConfig (stereo + fusion + surface)
    ├── RuntimeConfig (device, output, viz, benchmark)
    └── PipelineConfig (top-level, contains all above)
```

**Rationale:** Single-file config keeps it cohesive; splitting into separate modules only justified if file exceeds 1000 lines. Current 14 dataclasses @ 492 lines → ~5 Pydantic models @ 400-500 lines is maintainable in single file.

### Pattern 1: Pydantic BaseModel with Default Factories

**What:** Replace dataclass with Pydantic BaseModel; use Field(default_factory=...) for sub-configs
**When to use:** For all config classes
**Example:**
```python
from pydantic import BaseModel, Field, ConfigDict

class PreprocessingConfig(BaseModel):
    """Configuration for preprocessing stage."""
    model_config = ConfigDict(extra='allow')  # Forward-compatible

    color_norm_enabled: bool = False
    color_norm_method: str = "gain"
    frame_start: int = 0
    frame_stop: int | None = None
    frame_step: int = 1

class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""
    model_config = ConfigDict(extra='allow')

    # Required fields (no defaults)
    calibration_path: str
    output_dir: str
    camera_video_map: dict[str, str]
    matcher_type: str  # "lightglue" or "roma"

    # Optional fields with sub-config defaults
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    sparse_matching: SparseMatchingConfig = Field(default_factory=SparseMatchingConfig)
```
**Source:** [Pydantic Fields Documentation](https://docs.pydantic.dev/latest/concepts/fields/)

### Pattern 2: Cross-Field Validation with @model_validator

**What:** Validate constraints across multiple fields using `@model_validator(mode='after')`
**When to use:** For cross-stage constraints (e.g., matcher_type=roma requires DenseMatchingConfig)
**Example:**
```python
from typing_extensions import Self
from pydantic import model_validator

class PipelineConfig(BaseModel):
    matcher_type: str
    dense_matching: DenseMatchingConfig = Field(default_factory=DenseMatchingConfig)

    @model_validator(mode='after')
    def validate_matcher_config(self) -> Self:
        """Ensure matcher_type=roma has appropriate dense_matching settings."""
        if self.matcher_type == "roma":
            # RoMa requires certainty threshold >= 0.5 for quality
            if self.dense_matching.certainty_threshold < 0.5:
                raise ValueError(
                    "dense_matching.certainty_threshold must be >= 0.5 for matcher_type='roma'"
                )
        return self
```
**Source:** [Pydantic Validators Documentation](https://docs.pydantic.dev/latest/concepts/validators/)

### Pattern 3: User-Friendly Error Messages with Field Paths

**What:** Transform ValidationError into dotted-path format for YAML navigation
**When to use:** In CLI error handling (cli.py) when catching ValidationError
**Example:**
```python
from pydantic import ValidationError

def format_validation_errors(e: ValidationError) -> str:
    """Format validation errors with YAML paths."""
    errors = []
    for error in e.errors():
        # Convert ('dense_stereo', 'num_depths') -> 'dense_stereo.num_depths'
        path = '.'.join(str(x) for x in error['loc'] if isinstance(x, str))
        msg = error['msg']
        errors.append(f"  {path}: {msg}")
    return "Configuration validation failed:\n" + "\n".join(errors)

# Usage in CLI:
try:
    config = PipelineConfig.from_yaml(config_path)
except ValidationError as e:
    print(format_validation_errors(e), file=sys.stderr)
    sys.exit(1)
```
**Source:** [Pydantic Error Handling Documentation](https://docs.pydantic.dev/latest/errors/errors/)

### Pattern 4: Progress Bars with tqdm

**What:** Wrap long-running loops with tqdm for automatic progress display
**When to use:** Plane sweep stereo, depth fusion, pair matching
**Example:**
```python
from tqdm import tqdm

# Automatic TTY detection and Jupyter support
for ref_name in tqdm(reference_cameras, desc="Computing depth maps"):
    depth_map = plane_sweep_stereo(...)

# Suppression via disable parameter (auto-detect non-TTY)
import sys
for ref_name in tqdm(reference_cameras, disable=not sys.stderr.isatty()):
    ...
```
**Source:** [tqdm Documentation](https://github.com/tqdm/tqdm)

### Pattern 5: Logging Applied Defaults

**What:** Log INFO messages when optional config sections use default values
**When to use:** During PipelineConfig.from_yaml() when sub-config is missing from YAML
**Example:**
```python
import logging
logger = logging.getLogger(__name__)

@classmethod
def from_yaml(cls, path: Path) -> "PipelineConfig":
    """Load config from YAML, logging applied defaults."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    # Log defaults for missing sections
    if "preprocessing" not in data:
        logger.info("Using default: preprocessing (all defaults)")
    if "dense_stereo" not in data:
        logger.info("Using default: dense_stereo.num_depths=128")

    return cls.model_validate(data)
```
**Source:** [Python Logging Documentation](https://docs.python.org/3/howto/logging.html)

### Pattern 6: Extra Fields Warning (Forward-Compatible)

**What:** Detect unknown YAML keys and log warnings without failing validation
**When to use:** To support future config versions while alerting users to typos
**Example:**
```python
class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra='allow')

    @model_validator(mode='after')
    def warn_extra_fields(self) -> Self:
        """Warn about unknown config keys (potential typos or future fields)."""
        if hasattr(self, '__pydantic_extra__') and self.__pydantic_extra__:
            unknown = list(self.__pydantic_extra__.keys())
            logger.warning(
                "Unknown config keys (ignored): %s. "
                "Check for typos or consult latest documentation.",
                unknown
            )
        return self
```
**Source:** [Pydantic Configuration Documentation](https://docs.pydantic.dev/latest/api/config/)

### Anti-Patterns to Avoid

- **Manual validation in validate() method:** Pydantic performs validation automatically; custom validate() methods are redundant and prevent error collection
- **Fail-on-first validation:** Breaks UX; always collect all errors (Pydantic does this automatically)
- **Hard-coded `.cuda()` calls:** Current code properly follows input tensor device; maintain this pattern
- **Using `extra='forbid'`:** Breaks forward compatibility; users upgrading config files would hit errors on new fields

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Config validation | Manual `validate()` methods with if/else chains | Pydantic `@field_validator` and `@model_validator` | Pydantic handles error collection, type coercion, nested validation, and generates clear error messages with field paths automatically |
| Error message formatting | String concatenation for error paths | Pydantic's ValidationError.errors() | ValidationError provides structured error details with 'loc' tuples, 'msg', 'type', and 'ctx' for rendering |
| YAML schema validation | Custom YAML parsers with schema checking | Pydantic BaseModel with yaml.safe_load() | Pydantic validates Python dicts from YAML; handles type coercion, nested objects, and defaults automatically |
| Progress bar rendering | Manual print() with percentage calculations | tqdm | tqdm handles TTY detection, terminal width, Jupyter widgets, ETA calculation, and thread-safe updates |
| Nested config defaults | Deep dict.get() chains or custom merge logic | Pydantic Field(default_factory=...) | Pydantic instantiates sub-models automatically; handles None vs missing distinctions correctly |
| CLI config overrides | Manual string parsing for nested paths | jsonargparse or pydantic-settings (if implemented) | These libraries handle dotted-path parsing, type coercion, and merging with config files |

**Key insight:** Configuration validation is a solved problem with established patterns. Hand-rolling validation logic leads to incomplete error handling (fail-on-first), poor error messages (no field paths), and missed edge cases (type coercion, nested validation). Pydantic is the Python ecosystem standard for this problem domain.

## Common Pitfalls

### Pitfall 1: Using extra='forbid' for Strict Validation

**What goes wrong:** Users cannot load config files with unknown keys, breaking forward compatibility when new fields are added
**Why it happens:** Developers confuse "strict validation" with "reject unknown fields"; assumes YAML typos are more common than version mismatches
**How to avoid:** Use `extra='allow'` and manually inspect `__pydantic_extra__` in a model_validator to log warnings
**Warning signs:** User reports "validation error: extra fields not permitted" when loading config generated by `aquamvs init` from newer version
**Example:**
```python
# BAD: Breaks forward compatibility
model_config = ConfigDict(extra='forbid')

# GOOD: Warns about unknown keys but doesn't fail
model_config = ConfigDict(extra='allow')
@model_validator(mode='after')
def warn_extra_fields(self) -> Self:
    if self.__pydantic_extra__:
        logger.warning("Unknown config keys: %s", list(self.__pydantic_extra__.keys()))
    return self
```

### Pitfall 2: Raising ValidationError Directly in Custom Validators

**What goes wrong:** ValidationError raised in validator has wrong field path, confusing users about which field failed
**Why it happens:** Developers see ValidationError as the "validation exception" and raise it directly instead of ValueError
**How to avoid:** Always raise ValueError or AssertionError from custom validators; Pydantic wraps these automatically with correct field path
**Warning signs:** Error message shows `loc=()` (empty tuple) instead of field path
**Example:**
```python
# BAD: Loses field path context
@field_validator('num_depths')
def check_positive(cls, v):
    if v <= 0:
        raise ValidationError("must be positive")  # Wrong!
    return v

# GOOD: Pydantic wraps ValueError with correct field path
@field_validator('num_depths')
def check_positive(cls, v):
    if v <= 0:
        raise ValueError("must be positive")  # Pydantic adds field path
    return v
```
**Source:** [Pydantic Validators Documentation](https://docs.pydantic.dev/latest/concepts/validators/)

### Pitfall 3: Not Logging Applied Defaults

**What goes wrong:** Users don't know which defaults were applied, leading to confusion when behavior differs from expectations
**Why it happens:** Developers assume users will read documentation or remember defaults from `aquamvs init`
**How to avoid:** Log INFO message for each missing config section with defaults applied
**Warning signs:** User reports "unexpected behavior" that traces to a default value they didn't explicitly set
**Example:**
```python
# BAD: Silent defaults
config = PipelineConfig.model_validate(data)

# GOOD: Explicit logging
if "dense_stereo" not in data:
    logger.info("Using default: dense_stereo.num_depths=128")
config = PipelineConfig.model_validate(data)
```

### Pitfall 4: tqdm in Non-TTY Contexts Polluting Logs

**What goes wrong:** Progress bars render as multiple lines in log files or CI environments, creating unreadable output
**Why it happens:** tqdm defaults to enabled; developers test in interactive terminals only
**How to avoid:** Use `disable=not sys.stderr.isatty()` for auto-detection, or provide --quiet flag
**Warning signs:** Log files contain thousands of lines of progress bar updates
**Example:**
```python
import sys
from tqdm import tqdm

# Auto-detect TTY and disable in non-interactive contexts
for item in tqdm(items, disable=not sys.stderr.isatty()):
    process(item)
```
**Source:** [tqdm GitHub Repository](https://github.com/tqdm/tqdm)

### Pitfall 5: Incorrect Field Path Formatting for Arrays

**What goes wrong:** Error messages show `items.0.value` instead of `items[0].value` for list elements
**Why it happens:** Naive `.join()` on location tuple treats integers as strings
**How to avoid:** Format integer indices with brackets: `[{x}]` for int, `.{x}` for str
**Warning signs:** User confusion when error says "field items.0.value" (looks like nested dict, not list index)
**Example:**
```python
# BAD: Incorrect array formatting
def loc_to_path(loc: tuple) -> str:
    return '.'.join(str(x) for x in loc)  # items.0.value

# GOOD: Proper array formatting
def loc_to_path(loc: tuple) -> str:
    path = ''
    for i, x in enumerate(loc):
        if isinstance(x, str):
            path += '.' if i > 0 else ''
            path += x
        elif isinstance(x, int):
            path += f'[{x}]'
    return path  # items[0].value
```

## Code Examples

Verified patterns from official sources:

### Cross-Stage Validation Pattern

```python
from pydantic import BaseModel, model_validator
from typing_extensions import Self

class PipelineConfig(BaseModel):
    matcher_type: str
    pipeline_mode: str
    dense_matching: DenseMatchingConfig = Field(default_factory=DenseMatchingConfig)
    dense_stereo: DenseStereoConfig = Field(default_factory=DenseStereoConfig)

    @model_validator(mode='after')
    def validate_cross_stage_constraints(self) -> Self:
        """Validate constraints across pipeline stages."""
        # Constraint: matcher_type=roma requires dense_matching config
        if self.matcher_type == "roma":
            if self.dense_matching.certainty_threshold < 0.5:
                raise ValueError(
                    "dense_matching.certainty_threshold must be >= 0.5 when matcher_type='roma'"
                )

        # Constraint: pipeline_mode=full requires dense_stereo config
        if self.pipeline_mode == "full":
            if self.dense_stereo.num_depths < 16:
                raise ValueError(
                    "dense_stereo.num_depths must be >= 16 for pipeline_mode='full'"
                )

        return self
```
**Source:** [Pydantic Validators Documentation](https://docs.pydantic.dev/latest/concepts/validators/)

### YAML Loading with Applied Defaults Logging

```python
import logging
from pathlib import Path
import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class PipelineConfig(BaseModel):
    # Required fields
    calibration_path: str
    output_dir: str
    matcher_type: str

    # Optional with defaults
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    sparse_matching: SparseMatchingConfig = Field(default_factory=SparseMatchingConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load config from YAML with default logging."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # Log missing sections that will use defaults
        sections = {
            "preprocessing": "color_norm_enabled=False, frame_step=1",
            "sparse_matching": "max_keypoints=2048, num_neighbors=4",
            "dense_matching": "certainty_threshold=0.5",
            "dense_stereo": "num_depths=128, cost_function='ncc'",
            "runtime": "device='cpu', save_depth_maps=True",
        }

        for section, defaults in sections.items():
            if section not in data:
                logger.info(f"Using default: {section} ({defaults})")

        # Pydantic validates and fills defaults
        return cls.model_validate(data)
```
**Source:** [Python Logging HOWTO](https://docs.python.org/3/howto/logging.html)

### Error Formatting for User Messages

```python
from pydantic import ValidationError

def format_validation_errors(e: ValidationError) -> str:
    """Format Pydantic validation errors for user-friendly CLI output.

    Converts error locations to YAML paths and groups errors by type.
    """
    lines = ["Configuration validation failed:"]

    for error in e.errors():
        # Convert location tuple to dotted path
        path_parts = []
        for item in error['loc']:
            if isinstance(item, str):
                path_parts.append(item)
            elif isinstance(item, int):
                path_parts[-1] += f'[{item}]'

        path = '.'.join(path_parts)
        msg = error['msg']

        # Add context if available (e.g., constraint values)
        ctx = error.get('ctx')
        if ctx:
            ctx_str = ', '.join(f"{k}={v}" for k, v in ctx.items())
            lines.append(f"  {path}: {msg} ({ctx_str})")
        else:
            lines.append(f"  {path}: {msg}")

    return '\n'.join(lines)

# CLI usage
try:
    config = PipelineConfig.from_yaml(config_path)
except ValidationError as e:
    print(format_validation_errors(e), file=sys.stderr)
    sys.exit(1)
```
**Source:** [Pydantic Error Handling Documentation](https://docs.pydantic.dev/latest/errors/errors/)

### Progress Bar Integration

```python
import sys
from tqdm import tqdm

def process_frame_batch(
    config: PipelineConfig,
    frames: list[int],
    quiet: bool = False,
) -> None:
    """Process multiple frames with progress reporting.

    Args:
        config: Pipeline configuration.
        frames: List of frame indices to process.
        quiet: Suppress progress bars (for non-interactive contexts).
    """
    # Auto-detect TTY if not explicitly suppressed
    disable_progress = quiet or not sys.stderr.isatty()

    # Plane sweep stereo (slow operation)
    for ref_camera in tqdm(
        reference_cameras,
        desc="Computing depth maps",
        disable=disable_progress,
        unit="camera",
    ):
        depth_map = plane_sweep_stereo(ref_camera, config)

    # Depth fusion (slow operation)
    for frame_idx in tqdm(
        frames,
        desc="Fusing depth maps",
        disable=disable_progress,
        unit="frame",
    ):
        fused_cloud = fuse_depth_maps(frame_idx, config)
```
**Source:** [tqdm GitHub Repository](https://github.com/tqdm/tqdm)

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Dataclasses with manual validate() | Pydantic BaseModel with validators | Pydantic v2 (Jun 2023) | Automatic error collection, type coercion, JSON schema generation |
| Config class for settings | model_config = ConfigDict(...) | Pydantic v2.0 | Cleaner syntax, better IDE support, ConfigDict is typed |
| @validator decorator | @field_validator and @model_validator | Pydantic v2.0 | Clearer distinction between field-level and model-level validation |
| default_factory=lambda: Type() | Field(default_factory=Type) | Pydantic v1.8+ | Explicit Field() usage is more discoverable and typed |
| ValidationError from validators | ValueError/AssertionError | Pydantic v2.0 | Correct field path attribution, better error messages |

**Deprecated/outdated:**
- **Config class pattern:** Nested `class Config:` inside model is deprecated in Pydantic v2; use `model_config = ConfigDict(...)` instead
- **@validator decorator:** Deprecated in v2; replaced by `@field_validator` (single field) and `@model_validator` (cross-field)
- **parse_obj():** Deprecated in v2; replaced by `model_validate()` for dict input
- **dict():** Deprecated in v2; replaced by `model_dump()` for serialization

## Open Questions

1. **CLI Override Mechanism**
   - What we know: User wants option for CLI overrides (marked Claude's discretion)
   - What's unclear: Whether to use dotted-path args (--dense-stereo.num-depths=256) or YAML-only
   - Recommendation: Implement YAML-only for MVP; add dotted-path overrides in later phase if requested. Rationale: Current CLI already has `--device` override as precedent; extending to arbitrary paths requires library (jsonargparse) or custom parsing; complexity vs. value unclear without user feedback.

2. **Progress Bar Suppression**
   - What we know: User wants progress bars suppressible with --quiet flag
   - What's unclear: Whether to auto-detect non-TTY or require explicit flag
   - Recommendation: Use `disable=quiet or not sys.stderr.isatty()` for best-of-both-worlds. Auto-detects CI/log contexts while allowing explicit --quiet override.

3. **Evaluation Config Placement**
   - What we know: User left placement as Claude's discretion (Reconstruction vs Runtime)
   - What's unclear: Whether evaluation metrics belong with reconstruction (domain) or runtime (cross-cutting)
   - Recommendation: Place in RuntimeConfig. Rationale: Evaluation is optional post-processing, not core reconstruction; groups naturally with output and visualization settings; keeps ReconstructionConfig focused on surface generation.

## Sources

### Primary (HIGH confidence)

- [Pydantic v2 Configuration API](https://docs.pydantic.dev/latest/api/config/) - ConfigDict, extra field handling
- [Pydantic v2 Validators](https://docs.pydantic.dev/latest/concepts/validators/) - @field_validator, @model_validator patterns
- [Pydantic v2 Error Handling](https://docs.pydantic.dev/latest/errors/errors/) - ValidationError formatting, error extraction
- [Pydantic v2 Fields](https://docs.pydantic.dev/latest/concepts/fields/) - Field() usage, default_factory
- [tqdm GitHub Repository](https://github.com/tqdm/tqdm) - TTY detection, Jupyter support
- [Python Logging Documentation](https://docs.python.org/3/howto/logging.html) - INFO level logging, basicConfig()
- [Python warnings module](https://docs.python.org/3/library/warnings.html) - DeprecationWarning, @deprecated decorator

### Secondary (MEDIUM confidence)

- [Pydantic Migration Guide](https://docs.pydantic.dev/latest/migration/) - v1 to v2 migration, deprecated features
- [How to Validate YAML Configs Using Pydantic](https://medium.com/better-programming/validating-yaml-configs-made-easy-with-pydantic-594522612db5) - YAML integration patterns
- [jsonargparse Documentation](https://jsonargparse.readthedocs.io/) - Dotted-path CLI parsing (if implemented)
- [Pydantic Complete Guide 2026](https://devtoolbox.dedyn.io/blog/pydantic-complete-guide) - Recent best practices

### Tertiary (LOW confidence)

- Various GitHub discussions on Pydantic extra field handling, custom error messages - implementation details vary, verify against official docs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Pydantic v2 and tqdm are undisputed standards with official documentation
- Architecture: HIGH - Patterns verified against official Pydantic v2 docs and current codebase structure
- Pitfalls: HIGH - Derived from official docs, known issues in Pydantic GitHub, and common config validation mistakes

**Research date:** 2026-02-14
**Valid until:** 2026-03-14 (30 days - Pydantic v2 is stable, minimal API churn expected)
