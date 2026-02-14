# Phase 2: Configuration and API Cleanup - Context

**Gathered:** 2026-02-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Consolidate the 14-dataclass configuration system into ~5 validated Pydantic models grouped by pipeline stage. Add load-time validation with clear error messages, support minimal config files with sensible defaults, and add progress bars to long-running operations. Public API typing and deprecation patterns established.

</domain>

<decisions>
## Implementation Decisions

### Config grouping
- Group by pipeline stage, not by user concern
- Dual-pathway architecture reflected in config: separate SparseMatchingConfig (feature extraction, pair selection, LightGlue matching) and DenseMatchingConfig (RoMa)
- Cross-cutting concerns (device, output, visualization, benchmark) grouped into a single RuntimeConfig section
- Preprocessing concerns (color normalization, frame sampling) grouped into PreprocessingConfig
- Resulting top-level structure: ~5 groups (Preprocessing, SparseMatching, DenseMatching, Reconstruction [stereo+fusion+surface], Runtime)

### Validation behavior
- Collect all validation errors and report them together (not fail-on-first)
- Unknown/extra YAML keys produce a warning, not an error (forwards-compatible)
- Cross-stage constraints (e.g., matcher_type=roma requires dense_matching settings) validated at load time, before any processing starts
- Error messages use YAML paths (e.g., `dense_stereo.num_depths: must be > 0`) so user knows exactly which field to fix

### Minimal config UX
- Minimum required fields: paths (video_dir, calibration_file, output_dir) + matcher_type (lightglue vs roma)
- `aquamvs init` generates a full annotated config with all fields and comments showing defaults
- When loading a config with missing optional sections, log applied defaults at INFO level (e.g., "Using default: dense_stereo.num_depths=128")

### Progress reporting
- Progress bars on slow operations only: plane sweep stereo, depth fusion, pair matching
- Use tqdm (lightweight, works in terminals and Jupyter)
- Progress bars on by default in CLI; suppressible with --quiet
- On by default with INFO-level log showing which defaults were applied

### Claude's Discretion
- CLI override mechanism for config values (dotted overrides vs YAML-only) — decide based on implementation complexity vs value
- Progress bar suppression in non-TTY/library contexts — decide between auto-detection and config control
- Exact Pydantic model field names and nesting depth within each group
- Evaluation config placement (could be Reconstruction or Runtime)

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches. The key constraint is maintaining the dual-pathway architecture (LightGlue sparse vs RoMa dense) in the config structure.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-configuration-and-api-cleanup*
*Context gathered: 2026-02-14*
