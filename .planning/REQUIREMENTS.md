# Requirements: AquaMVS

**Defined:** 2026-02-14
**Core Value:** Accurate refractive multi-view stereo reconstruction — cameras in air, geometry underwater, Snell's law bridging the two.

## v1 Requirements

Requirements for production-ready release. Each maps to roadmap phases.

### Packaging

- [ ] **PKG-01**: User can install AquaMVS via `pip install aquamvs` from PyPI
- [ ] **PKG-02**: LightGlue dependency pinned to specific commit hash for reproducible builds
- [ ] **PKG-03**: RoMa v2 installable without `--no-deps` workaround (or workaround automated)
- [ ] **PKG-04**: AquaCal dependency resolved for distribution (published to PyPI, vendored, or documented prerequisite)
- [ ] **PKG-05**: Package uses semantic versioning with CHANGELOG.md tracking releases
- [ ] **PKG-06**: Wheel excludes tests, docs, and dev files from distribution

### Configuration

- [ ] **CFG-01**: Invalid config values produce clear error messages at load time (not silent failures downstream)
- [ ] **CFG-02**: Config dataclasses consolidated from 9+ to 4-5 logical groups
- [ ] **CFG-03**: YAML config has sensible defaults so minimal config files work out of the box
- [ ] **CFG-04**: Cross-stage constraints validated (e.g., matcher_type=roma implies DenseMatchingConfig applies)

### Pipeline Refactoring

- [ ] **REF-01**: pipeline.py decomposed into modular pipeline/ package (builder, runner, stages)
- [ ] **REF-02**: Each execution path (lightglue+sparse, lightglue+full, roma+sparse, roma+full) is a distinct stage module
- [ ] **REF-03**: AquaCal VideoSet usage isolated behind adapter interface (not directly in pipeline)
- [ ] **REF-04**: Feature extractors accessed via protocol/factory pattern for swappability

### Documentation

- [ ] **DOC-01**: README includes project description, installation instructions, and quickstart example
- [ ] **DOC-02**: Platform-specific installation guide (Windows, Linux, CPU-only, GPU)
- [ ] **DOC-03**: At least one Jupyter tutorial demonstrating end-to-end reconstruction workflow
- [ ] **DOC-04**: Example dataset available (bundled or downloadable) for users to test pipeline immediately

### CLI & UX

- [ ] **UX-01**: Long-running operations (matching, plane sweep, fusion) display progress bars

### Benchmarking & Performance

- [ ] **BEN-01**: Internal benchmark comparing RoMa vs LightGlue pathway accuracy on same dataset
- [ ] **BEN-02**: Runtime profiling identifies top 3 bottlenecks with measurements
- [ ] **BEN-03**: At least one optimization implemented based on profiling results (RoMa or plane sweep)

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Code Quality

- **QA-01**: Type hints throughout codebase with mypy passing
- **QA-02**: CI/CD via GitHub Actions (Windows + Linux matrix)
- **QA-03**: Ruff migration from Black for linting/formatting

### Documentation (Advanced)

- **DOC-05**: Sphinx API documentation auto-generated from docstrings
- **DOC-06**: ReadTheDocs hosting with versioned documentation
- **DOC-07**: Configuration reference documenting all parameters and defaults

### CLI (Advanced)

- **UX-02**: Better error messages with contextual diagnostics
- **UX-03**: Typer migration from argparse for type-safe CLI

### Config (Advanced)

- **CFG-05**: Pydantic migration replacing dataclasses (runtime validation, JSON schema)

### Benchmarking (Advanced)

- **BEN-04**: Ground truth evaluation against charuco board calibration targets
- **BEN-05**: Historical performance tracking via asv benchmark suite

### Packaging (Advanced)

- **PKG-07**: Docker/Singularity container for reproducible research environments

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| External MVS comparison (COLMAP, OpenMVS) | Nice-to-have but not required for v1 release |
| Temporal median preprocessing (fish removal) | Designed but deferred; separate milestone |
| Pinhole mode for comparison | Future work; refractive mode is the product |
| Multi-GPU support | Post-v1 optimization; single GPU sufficient |
| GUI application | Maintenance burden; Jupyter + CLI covers use cases |
| Real-time reconstruction | Domain mismatch; post-processing is standard |
| Plugin architecture | Premature; clean Python API is already extensible |
| Automated overlap-aware pair selection | Future improvement; current N-nearest works |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| PKG-01 | TBD | Pending |
| PKG-02 | TBD | Pending |
| PKG-03 | TBD | Pending |
| PKG-04 | TBD | Pending |
| PKG-05 | TBD | Pending |
| PKG-06 | TBD | Pending |
| CFG-01 | TBD | Pending |
| CFG-02 | TBD | Pending |
| CFG-03 | TBD | Pending |
| CFG-04 | TBD | Pending |
| REF-01 | TBD | Pending |
| REF-02 | TBD | Pending |
| REF-03 | TBD | Pending |
| REF-04 | TBD | Pending |
| DOC-01 | TBD | Pending |
| DOC-02 | TBD | Pending |
| DOC-03 | TBD | Pending |
| DOC-04 | TBD | Pending |
| UX-01 | TBD | Pending |
| BEN-01 | TBD | Pending |
| BEN-02 | TBD | Pending |
| BEN-03 | TBD | Pending |

**Coverage:**
- v1 requirements: 22 total
- Mapped to phases: 0
- Unmapped: 22 ⚠️

---
*Requirements defined: 2026-02-14*
*Last updated: 2026-02-14 after initial definition*
