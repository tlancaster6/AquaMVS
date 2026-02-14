# Project Research Summary

**Project:** AquaMVS Production Readiness
**Domain:** Scientific Python Reconstruction Library (Computer Vision / Underwater MVS)
**Researched:** 2026-02-14
**Confidence:** HIGH

## Executive Summary

AquaMVS is a scientific Python library for multi-view stereo reconstruction with refractive geometry modeling — a niche capability with unique technical requirements. Research shows that production-ready scientific Python packages follow well-established patterns: modern build systems (pyproject.toml + hatchling), comprehensive documentation (Sphinx on ReadTheDocs), quality tooling (Ruff + mypy + pytest), and CI/CD with trusted publishing. The current codebase has functional reconstruction algorithms but lacks production packaging, creates dependency time bombs, and has a monolithic 995-line pipeline that violates maintainability best practices.

The recommended approach is a phased modernization: first, resolve critical dependency blockers (unpinned git dependencies, local path dependencies, --no-deps workarounds) that prevent PyPI publication; second, refactor the monolithic pipeline using builder and strategy patterns while maintaining backward compatibility; third, establish production packaging with proper wheel configuration and platform testing; fourth, add comprehensive documentation and examples. This order reflects the dependency chain: cannot publish to PyPI without resolving dependencies, cannot refactor safely without tests, cannot document properly without stable architecture.

Key risks are breaking existing users during refactoring (mitigate with deprecation warnings and compatibility shims), creating unresolvable dependency conflicts (mitigate by following lower-bound-only versioning), and platform-specific failures (mitigate with multi-platform CI testing). The refractive geometry core is already solid; productionization is about packaging and developer experience, not algorithmic development.

## Key Findings

### Recommended Stack

The 2026 scientific Python ecosystem has converged on a modern toolchain that replaces older approaches. Research confirms Ruff has replaced Black+isort+Flake8 with 100x speedup and identical output, uv has replaced pip with 10-100x faster resolution, and trusted publishing has replaced manual API tokens for PyPI uploads. Sphinx remains the standard for scientific documentation despite MkDocs' popularity, primarily because autodoc from docstrings is critical for API-heavy libraries and the Scientific Python community has standardized on it.

**Core technologies:**
- **pyproject.toml + hatchling**: Modern build system — industry standard (PEP 621), replaces deprecated setup.py
- **Ruff (>=0.15.1)**: Unified linter+formatter — replaces Black/isort/Flake8 with 100x speedup
- **Sphinx + ReadTheDocs**: Documentation generation — Scientific Python standard, superior autodoc vs MkDocs
- **pytest-benchmark + asv**: Benchmarking — quick feedback (pytest-benchmark) + historical tracking (asv)
- **GitHub Actions + Trusted Publishing**: CI/CD — OIDC authentication, no API token management
- **Pydantic + Typer**: Config validation + modern CLI — runtime validation + type-safe CLI vs plain dataclasses + argparse

### Expected Features

Production-ready scientific Python libraries require comprehensive documentation, PyPI packaging, CI/CD, type hints, and cross-platform support as baseline expectations. Users assume these exist; missing them makes the product feel incomplete or unmaintained. Differentiators for AquaMVS are its refractive geometry handling (unique capability), built-in benchmarking tools (rare in reconstruction libraries), and config-driven workflows (vs API-only approaches).

**Must have (table stakes):**
- PyPI packaging with `pip install aquamvs` — standard Python distribution
- Comprehensive API documentation — users need to understand the library
- GitHub Actions CI with tests — users trust tested code
- Type hints throughout — modern Python expectation, enables IDE autocomplete
- Installation instructions for platforms — especially critical for PyTorch CPU/GPU variants
- Error messages with context — users need to debug failures
- Versioning and changelog — users track changes between releases

**Should have (competitive):**
- Jupyter notebook tutorials — research library adoption pattern (PyTorch3D model)
- Pre-computed example datasets — lowers barrier to entry
- Progress bars with logging — long operations need feedback
- Docker container — reproducibility for scientific research
- Better error diagnostics — based on user bug reports after initial release

**Defer (v2+):**
- Interactive visualization tools — high complexity, unclear value vs static viz
- Advanced config validation with schema — wait for config patterns to stabilize
- Plugin architecture for custom extractors — wait for external contributor demand

### Architecture Approach

Scientific Python packages use src/ layout with modular subpackages organized by concern (core/, features/, dense/, io/, utils/). The current monolithic 995-line pipeline.py should decompose into a pipeline/ package using builder pattern (construct from config), strategy pattern (swappable feature extractors and dense stereo methods), and adapter pattern (isolate external dependencies like AquaCal VideoSet and Open3D). This follows established patterns from Kornia, Open3D, and PyTorch3D.

**Major components:**
1. **pipeline/** — Builder constructs stages from config, runner executes with shared context, replaces monolithic orchestrator
2. **features/** + **dense/** — Strategy pattern with protocol interfaces enables swappable extractors (SuperPoint/ALIKED/DISK/RoMa) and benchmarking
3. **io/** — Adapter pattern isolates AquaCal and Open3D dependencies to specific modules for easier testing and future changes
4. **config.py** — Migrate from plain dataclasses to Pydantic for runtime validation and clear error messages
5. **cli.py** — Migrate from argparse to Typer for type-safe commands and better UX

### Critical Pitfalls

Production packaging failures from research reveal common time bombs in scientific libraries. Most critical for AquaMVS are dependency issues that block PyPI publication and refactoring mistakes that break existing users.

1. **Git dependencies without commit pins** — LightGlue via unpinned git+ URL creates unreproducible builds and PyPI rejection; must pin to commit hash, fork and vendor, or wait for PyPI release before publishing
2. **Local editable dependencies in production** — AquaCal via path="../AquaCal" cannot be distributed; must publish AquaCal to PyPI first, extract and vendor needed interfaces, or document as prerequisite installation step
3. **--no-deps workaround becomes requirement** — RoMa installation with --no-deps bypasses resolver and creates support nightmare; must investigate actual runtime requirements, test if newer numpy works despite metadata, or switch to alternative
4. **Breaking API changes without migration** — Refactoring 995-line pipeline will break existing imports; must add deprecation warnings, provide compatibility shims, document migration with examples, and follow semantic versioning
5. **Configuration sprawl without validation** — 9+ dataclasses with no validation silently accept invalid values; must consolidate to 4-5 Pydantic models with field validators and fail fast at initialization

## Implications for Roadmap

Based on research, suggested phase structure follows dependency order: must resolve packaging blockers before refactoring, must stabilize architecture before documentation.

### Phase 1: Dependency Resolution and Packaging Foundations
**Rationale:** Cannot publish to PyPI with current dependency configuration. Git dependencies without commit pins, local path dependencies, and --no-deps workarounds are critical blockers. Must resolve before any refactoring or documentation work because these prevent distribution.

**Delivers:**
- Publishable to PyPI (resolves git/path dependencies)
- Clean wheel configuration (tests excluded, platform markers set)
- Basic CI/CD pipeline (tests on Windows/Linux)

**Addresses:**
- PyPI packaging (table stakes from FEATURES.md)
- GitHub Actions CI (table stakes from FEATURES.md)
- Python 3.10+ support (table stakes from FEATURES.md)

**Avoids:**
- Git dependencies without pins (Pitfall #1)
- Local editable dependencies (Pitfall #2)
- --no-deps workaround (Pitfall #5)
- Test directories in wheel (Pitfall #4)
- GPU dependencies break CPU (Pitfall #6)

### Phase 2: Configuration and API Cleanup
**Rationale:** Current 9+ dataclass sprawl and monolithic pipeline make refactoring risky. Consolidate and validate configuration first, then establish deprecation pattern before touching public APIs. This phase sets foundation for safe refactoring.

**Delivers:**
- Consolidated Pydantic config (4-5 models with validation)
- Typer-based CLI with type hints
- Deprecation warnings on old APIs
- Type hints throughout codebase

**Addresses:**
- Type hints (table stakes from FEATURES.md)
- Error messages with context (table stakes from FEATURES.md)
- Configuration validation (should-have from FEATURES.md)

**Uses:**
- Pydantic for runtime validation (STACK.md recommendation)
- Typer for modern CLI (STACK.md recommendation)

**Avoids:**
- Configuration sprawl without validation (Pitfall #9)
- Breaking API changes without migration (Pitfall #8)
- Platform-specific hacks (Pitfall #3)

### Phase 3: Pipeline Decomposition and Modularization
**Rationale:** With dependencies resolved and config validated, refactor 995-line pipeline.py into modular pipeline/ package. Builder pattern separates construction from execution, strategy pattern enables swappable algorithms, adapter pattern isolates external dependencies. Maintain backward compatibility via facade.

**Delivers:**
- Modular pipeline/ package (builder, runner, stages, context)
- Strategy protocols for features/ and dense/
- Adapter wrappers for io/ (AquaCal, Open3D)
- Compatibility facade maintains old API

**Implements:**
- Builder pattern for pipeline construction (ARCHITECTURE.md)
- Strategy pattern for feature extractors (ARCHITECTURE.md)
- Adapter pattern for external dependencies (ARCHITECTURE.md)

**Addresses:**
- Code organization for maintainability
- Multiple feature extractors (differentiator from FEATURES.md)
- Extensibility for future algorithms

**Avoids:**
- 1000-line orchestrator anti-pattern (ARCHITECTURE.md)
- Leaking external dependencies (ARCHITECTURE.md)
- God class pipeline (ARCHITECTURE.md)

### Phase 4: Documentation and Examples
**Rationale:** With stable architecture and clean APIs, create comprehensive documentation. Sphinx autodoc generates API reference from docstrings, Jupyter tutorials demonstrate workflows, ReadTheDocs hosts versioned docs. Example datasets lower barrier to entry.

**Delivers:**
- Sphinx documentation with autodoc
- ReadTheDocs hosting with auto-build
- Jupyter tutorial notebooks
- Example dataset or download scripts
- CHANGELOG.md with versioning

**Uses:**
- Sphinx + nbsphinx + ReadTheDocs (STACK.md recommendation)
- pytest-benchmark for performance docs (STACK.md)

**Addresses:**
- Comprehensive API documentation (table stakes from FEATURES.md)
- Basic usage examples (table stakes from FEATURES.md)
- Jupyter tutorials (should-have from FEATURES.md)
- Example datasets (should-have from FEATURES.md)
- Versioning and changelog (table stakes from FEATURES.md)

**Avoids:**
- Missing type hints in signatures (PITFALLS.md documentation section)
- No installation examples for platforms (PITFALLS.md documentation section)
- Configuration reference without defaults (PITFALLS.md documentation section)

### Phase 5: Performance and Optimization
**Rationale:** After core functionality is production-ready, profile and optimize bottlenecks. Add benchmarking infrastructure for continuous performance tracking. This comes last because premature optimization wastes effort.

**Delivers:**
- asv benchmark suite for historical tracking
- Performance profiling with scalene
- torch.compile integration for GPU speedup
- Batch processing capabilities

**Uses:**
- pytest-benchmark + asv (STACK.md recommendation)
- scalene for profiling (STACK.md recommendation)
- torch.compile for JIT (STACK.md recommendation)

**Addresses:**
- Performance benchmarks (should-have from FEATURES.md)
- GPU acceleration optimization
- Batch processing (differentiator from FEATURES.md)

### Phase Ordering Rationale

- **Dependency resolution first:** Cannot publish, refactor, or document without resolving packaging blockers. Git dependencies, path dependencies, and --no-deps workarounds prevent PyPI distribution.
- **Config before refactoring:** Consolidating configuration and adding validation creates stable foundation. Trying to refactor pipeline with 9+ sprawling dataclasses invites errors.
- **Refactoring before documentation:** Documenting unstable APIs wastes effort. Architecture must stabilize before investing in comprehensive docs.
- **Documentation before optimization:** Users need docs more than speed. Optimize after establishing adoption, based on actual bottlenecks.
- **Backward compatibility throughout:** Deprecation warnings in Phase 2, compatibility facade in Phase 3, migration docs in Phase 4. Breaking changes deferred to major version bump.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1 (Dependency Resolution):** LightGlue and RoMa dependency status unclear — need to investigate current PyPI availability, commit stability, and version compatibility. May require contacting maintainers or exploring alternative libraries.
- **Phase 3 (Pipeline Decomposition):** Complexity of maintaining backward compatibility while refactoring — need detailed plan for deprecation warnings, compatibility shims, and testing old vs new APIs simultaneously.

Phases with standard patterns (skip research-phase):
- **Phase 2 (Config Cleanup):** Pydantic migration well-documented with clear examples from ecosystem
- **Phase 4 (Documentation):** Sphinx setup is standard practice with extensive guides for scientific Python
- **Phase 5 (Performance):** PyTorch optimization patterns well-established, asv documentation comprehensive

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Official Python Packaging Guide, verified PyPI versions, Astral/PyPA official docs. Ruff 0.15.1 and uv 0.9.27+ are current releases. |
| Features | HIGH | Scientific Python Development Guide, competitor analysis (Open3D, PyTorch3D, COLMAP), packaging standards from pyOpenSci. |
| Architecture | HIGH | Established patterns from Kornia, Open3D, scikit-learn. Builder/Strategy/Adapter patterns well-documented in ecosystem. |
| Pitfalls | HIGH | Python Packaging User Guide, PEP specifications, real-world examples from jwodder knowledge bits and community best practices. |

**Overall confidence:** HIGH

Research is based on official sources (Python Packaging Guide, PyPA, Astral docs), verified current tool versions (PyPI), and established patterns from major scientific libraries. The technical recommendations are not speculative — they reflect 2026 ecosystem standards.

### Gaps to Address

- **LightGlue/RoMa PyPI status:** Research found these require git installation. Need to verify if PyPI releases exist now (early 2026), check commit stability for pinning, and decide between waiting for official release vs vendoring vs switching alternatives. This affects Phase 1 timeline.
- **AquaCal publication timeline:** Local dependency on ../AquaCal blocks distribution. Need decision: publish AquaCal to PyPI first (requires coordination with AquaCal project), vendor minimal required interfaces (if license permits), or document as prerequisite manual install (poor UX). This is a critical blocker for Phase 1.
- **Backward compatibility scope:** Research recommends deprecation warnings and compatibility shims, but need to define exactly which APIs are "public" vs "internal". The current codebase may have users importing from various locations. Phase 2 needs audit of actual usage patterns if possible.
- **Example dataset licensing/size:** Research recommends pre-computed examples to lower barrier, but underwater multi-camera datasets are large. Need to determine appropriate sample size, licensing for distribution, and whether to bundle vs download-on-demand. Affects Phase 4 implementation.

## Sources

### Primary (HIGH confidence)
- [Python Packaging User Guide](https://packaging.python.org/) — pyproject.toml standards, trusted publishing, dependency specification (PEPs 508, 621, 660, 740)
- [Ruff Documentation](https://docs.astral.sh/ruff/) — linter/formatter migration from Black, version 0.15.1 verified
- [uv Documentation](https://docs.astral.sh/uv/) — package manager, version 0.9.27+ verified
- [Scientific Python Development Guide](https://learn.scientific-python.org/development/) — packaging, docs, CI patterns for scientific libraries
- [pyOpenSci Python Package Guide](https://www.pyopensci.org/python-package-guide/) — structure, testing, versioning best practices
- [PyTorch Documentation](https://docs.pytorch.org/) — torch.compile, GPU packaging considerations
- [Sphinx Documentation](https://www.sphinx-doc.org/) — scientific documentation standard
- [Pydantic Documentation](https://docs.pydantic.dev/) — configuration validation patterns

### Secondary (MEDIUM confidence)
- [Kornia GitHub](https://github.com/kornia/kornia) + [Open3D GitHub](https://github.com/isl-org/Open3D) — architecture reference examples
- [Typer Documentation](https://typer.tiangolo.com/) — CLI alternatives comparison
- [pytest-benchmark](https://pytest-benchmark.readthedocs.io/) + [asv](https://asv.readthedocs.io/) — benchmarking tools
- [Knowledge Bits - Common Python Packaging Mistakes](https://jwodder.github.io/kbits/posts/pypkg-mistakes/) — pitfall collection
- Community guides (Medium, blogs) — packaging best practices 2026, build backend comparison, configuration patterns

### Tertiary (LOW confidence, needs validation)
- RoMa v2 PyPI availability — mentioned in research but not verified with live check
- LightGlue PyPI status — research found git dependency but did not check recent releases
- MkDocs deprecation rumors — research mentioned Zensical replacement but timeline unclear

---
*Research completed: 2026-02-14*
*Ready for roadmap: yes*
