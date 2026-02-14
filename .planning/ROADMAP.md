# Roadmap: AquaMVS Production Readiness

## Overview

AquaMVS already delivers working refractive multi-view stereo reconstruction. This roadmap transforms it from research code into a production-ready library — resolving dependency blockers for PyPI publication, refactoring the monolithic pipeline for maintainability, establishing comprehensive documentation for users, and optimizing performance bottlenecks. Each phase builds on the previous: packaging foundations enable safe refactoring, stable architecture enables thorough documentation, and working infrastructure enables targeted optimization.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Dependency Resolution and Packaging Foundations** - Resolve PyPI blockers, establish clean wheel, set up CI/CD
- [ ] **Phase 2: Configuration and API Cleanup** - Consolidate config, validate inputs, establish type hints and deprecation patterns
- [ ] **Phase 3: Pipeline Decomposition and Modularization** - Refactor 995-line pipeline.py into modular package with protocols
- [ ] **Phase 4: Documentation and Examples** - Sphinx docs, Jupyter tutorials, example datasets, ReadTheDocs hosting
- [ ] **Phase 5: Performance and Optimization** - Benchmark suite, profiling, optimization based on measurements

## Phase Details

### Phase 1: Dependency Resolution and Packaging Foundations
**Goal**: Package is publishable to PyPI with clean dependencies and tested on multiple platforms
**Depends on**: Nothing (first phase)
**Requirements**: PKG-01, PKG-02, PKG-03, PKG-04, PKG-05, PKG-06
**Success Criteria** (what must be TRUE):
  1. User can run `pip install aquamvs` from PyPI and successfully import the package
  2. Package builds produce wheels that exclude tests, docs, and dev files
  3. LightGlue dependency is pinned to specific commit hash (or PyPI version if available)
  4. RoMa v2 installs without requiring manual `--no-deps` workaround
  5. AquaCal dependency is resolved for distribution (published to PyPI, vendored, or documented)
  6. CI pipeline runs tests successfully on both Windows and Linux platforms
**Plans:** 2 plans

Plans:
- [x] 01-01-PLAN.md — Resolve dependency blockers (LightGlue, RoMa, AquaCal), clean pyproject.toml, versioning setup
- [x] 01-02-PLAN.md — Create CI/CD workflows (test matrix + PyPI publish)

### Phase 2: Configuration and API Cleanup
**Goal**: Configuration is validated at load time with clear error messages, and public APIs are typed and stable
**Depends on**: Phase 1
**Requirements**: CFG-01, CFG-02, CFG-03, CFG-04, UX-01
**Success Criteria** (what must be TRUE):
  1. Invalid config files produce clear error messages identifying the problem and location (not silent failures)
  2. Config structure consolidated from 9+ dataclasses to 4-5 logical Pydantic models
  3. User can create minimal config file with only essential parameters (sensible defaults fill the rest)
  4. Long-running operations (matching, plane sweep, fusion) display progress bars showing completion percentage
  5. Cross-stage configuration constraints are validated (e.g., matcher_type=roma requires DenseMatchingConfig)
**Plans**: TBD

Plans:
- TBD during planning

### Phase 3: Pipeline Decomposition and Modularization
**Goal**: Monolithic pipeline.py is decomposed into maintainable modular package while preserving backward compatibility
**Depends on**: Phase 2
**Requirements**: REF-01, REF-02, REF-03, REF-04
**Success Criteria** (what must be TRUE):
  1. Pipeline code is organized into pipeline/ package with separate builder, runner, and stage modules
  2. Each execution path (lightglue+sparse, lightglue+full, roma+sparse, roma+full) is implemented as distinct stage module
  3. AquaCal VideoSet usage is isolated behind adapter interface (not directly imported throughout codebase)
  4. Feature extractors (SuperPoint, ALIKED, DISK, RoMa) are accessed via protocol interface enabling swappability
  5. Existing user code that imports from old pipeline.py continues to work with deprecation warnings
**Plans**: TBD

Plans:
- TBD during planning

### Phase 4: Documentation and Examples
**Goal**: Users can learn, install, and use AquaMVS through comprehensive documentation and working examples
**Depends on**: Phase 3
**Requirements**: DOC-01, DOC-02, DOC-03, DOC-04
**Success Criteria** (what must be TRUE):
  1. README includes project description, installation instructions, and quickstart example that new users can run
  2. Platform-specific installation guide covers Windows, Linux, CPU-only, and GPU configurations
  3. At least one Jupyter notebook tutorial demonstrates complete reconstruction workflow from videos to mesh
  4. Example dataset is available (bundled or downloadable) for users to test pipeline without preparing their own data
  5. Sphinx documentation is hosted on ReadTheDocs with auto-generated API reference from docstrings
**Plans**: TBD

Plans:
- TBD during planning

### Phase 5: Performance and Optimization
**Goal**: Performance bottlenecks identified and optimized based on measurements, with benchmarking infrastructure for tracking
**Depends on**: Phase 4
**Requirements**: BEN-01, BEN-02, BEN-03
**Success Criteria** (what must be TRUE):
  1. Internal benchmark comparing RoMa vs LightGlue pathway accuracy is implemented and results are documented
  2. Runtime profiling identifies and documents the top 3 performance bottlenecks with specific measurements
  3. At least one optimization targeting a measured bottleneck (RoMa or plane sweep) is implemented and verified
  4. Benchmark suite (asv or pytest-benchmark) tracks performance across code changes to detect regressions
**Plans**: TBD

Plans:
- TBD during planning

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Dependency Resolution and Packaging Foundations | 2/2 | ✓ Complete | 2026-02-14 |
| 2. Configuration and API Cleanup | 0/TBD | Not started | - |
| 3. Pipeline Decomposition and Modularization | 0/TBD | Not started | - |
| 4. Documentation and Examples | 0/TBD | Not started | - |
| 5. Performance and Optimization | 0/TBD | Not started | - |
