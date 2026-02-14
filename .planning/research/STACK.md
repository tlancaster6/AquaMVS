# Stack Research: Production Readiness for AquaMVS

**Domain:** Scientific Python Library Production Readiness
**Researched:** 2026-02-14
**Confidence:** HIGH

## Recommended Stack

### Packaging & Build System

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| pyproject.toml | PEP 621 | Unified project configuration | Industry standard (2026); replaces setup.py/setup.cfg. All modern tools read from [project] table. |
| hatchling | >=1.26 | Build backend | Fast, reproducible wheels, zero config for pure Python, plugin support for extensions. Recommended by Python Packaging Guide 2026. Alternative: setuptools >=77 (more complex) or flit-core >=3.12 (minimalist, no C extensions). |
| uv | >=0.9.27 | Package manager & resolver | 10-100x faster than pip, manages Python versions, global cache. Written in Rust by Astral (creators of Ruff). uv_build backend available but hatchling more mature. |

**Confidence:** HIGH - Official Python Packaging User Guide, verified against packaging.python.org (2026)

### Documentation Generation

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Sphinx | >=9.1.0 | Documentation generator | Industry standard for scientific Python (NumPy, SciPy, PyTorch). Autodoc from docstrings, LaTeX output, maturity. MyST parser adds Markdown support. |
| sphinx-autodoc-typehints | latest | Type hint rendering | Automatically renders type hints in API docs from annotations |
| nbsphinx | latest | Jupyter notebook integration | Executes notebooks during build, creates example galleries. Better for tutorials than MyST-NB (has gallery support). |
| sphinx-gallery | >=0.15 | Example gallery generation | Alternative to nbsphinx; generates galleries from .py files with structured comments |
| sphinx-book-theme | latest | Modern Sphinx theme | Clean, responsive, used by Scientific Python community |
| Read the Docs | cloud | Documentation hosting | Free for open source, auto-builds on git push, version management, search. Standard for scientific Python. |

**Confidence:** HIGH - Scientific Python Development Guide, official docs verified

**Rationale:** Sphinx over MkDocs because (1) autodoc from docstrings is critical for API-heavy scientific libraries, (2) Scientific Python ecosystem standardizes on Sphinx, (3) MkDocs future uncertain (creators building Zensical replacement). MkDocs only if prioritizing Markdown over features.

### Code Quality & Linting

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| ruff | >=0.15.1 | Linter + formatter | Replaces Black, isort, Flake8, pyupgrade, autoflake in one tool. 10-1000x faster (Rust-based). Format + lint in <1s on large codebases. |
| mypy | >=1.0 | Static type checker | Industry standard type checker. Ruff handles linting/formatting; mypy handles type analysis. Use together. |
| pre-commit | >=4.5.1 | Git hook manager | Runs ruff/mypy before commit. Prevents bad code from entering repo. |

**Confidence:** HIGH - Astral official docs, verified PyPI versions, multiple 2026 sources confirm Ruff dominance

**Rationale:** Do NOT use Black (replaced by `ruff format`), isort (replaced by `ruff check --select I`), or Flake8 (replaced by `ruff check`). Ruff provides identical output to Black (tested compatibility) with 100x speedup.

### Testing & Coverage

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|---|
| pytest | >=8.0 | Test framework | Standard for Python testing. Fixtures, parametrization, plugin ecosystem. |
| pytest-cov | >=7.0.0 | Coverage measurement | Integrates coverage.py with pytest. Automatic .coverage merging, xdist support. Version 7.0+ removes .pth approach (cleaner). |
| pytest-xdist | latest | Parallel testing | Run tests across CPUs/machines. Essential for large test suites. |

**Confidence:** HIGH - pytest official docs, pytest-cov 7.0 release notes

### Benchmarking

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| pytest-benchmark | >=5.2.3 | Micro-benchmarking | Integrates with pytest, automatic calibration, statistical analysis, regression detection. For quick "is this function fast?" checks. |
| asv (airspeed velocity) | >=0.6 | Historical performance tracking | Tracks performance over git history. Persistent storage, regression detection across releases. For "how has performance changed?" analysis. |

**Confidence:** HIGH - pytest-benchmark official docs, asv documentation

**Rationale:** Use both. pytest-benchmark for development (fast feedback), asv for release validation (historical trends). pytest-benchmark excellent for existing pytest suite; asv requires separate benchmark/ directory but provides lifetime tracking.

### Profiling & Optimization

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| scalene | latest | CPU/GPU/memory profiler | AI-powered optimization suggestions, line-level profiling, separates Python from native code time. Faster than cProfile. |
| py-spy | latest | Production profiler | Sampling profiler, zero overhead (runs in separate process), safe for production. Written in Rust. No code changes required. |
| torch.compile | PyTorch 2.x | JIT compilation | Modern PyTorch optimization (replaces torch.jit). 2-3x speedup on Conv2d/LSTM with minimal code changes. Use TorchInductor backend. |

**Confidence:** MEDIUM-HIGH - Official PyTorch docs (HIGH for torch.compile), GitHub repos and multiple 2026 sources (MEDIUM for scalene/py-spy versions)

**Rationale:** Scalene for development (find bottlenecks), py-spy for production (safe live profiling). torch.compile is PyTorch 2.x standard, superior to older torch.jit.script/trace.

### CI/CD & Publishing

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| GitHub Actions | cloud | CI/CD pipeline | Native GitHub integration, free for public repos, OIDC trusted publishing to PyPI. |
| pypa/gh-action-pypi-publish | >=v1.11.0 | PyPI upload action | Official PyPA action. Auto-generates PEP 740 attestations (v1.11.0+). Supports trusted publishing (no API tokens). |
| Trusted Publishing (OIDC) | protocol | PyPI authentication | Secure, tokenless publishing. Uses GitHub's OIDC identity. No secrets to manage. Standard practice 2026. |
| TestPyPI | cloud | Pre-production testing | Practice uploads before production PyPI. Same workflow, safe environment. |

**Confidence:** HIGH - Official Python Packaging Guide, PyPA official action, verified 2026 best practices

**Rationale:** Trusted Publishing eliminates API token management (security risk). Digital attestations (PEP 740) auto-generated by v1.11.0+. Do NOT manually create PyPI tokens.

## Supporting Libraries

| Library | Purpose | When to Use |
|---------|---------|-------------|
| sphinx-copybutton | Add copy buttons to code blocks | All Sphinx projects with code examples |
| sphinx-design | Responsive web components (grids, cards, tabs) | Tutorial-heavy documentation |
| myst-parser | Markdown support in Sphinx | If team prefers Markdown over reStructuredText |
| sphinx-autosummary-accessors | Generate API docs for class attributes | Libraries with attribute-heavy APIs |
| pytest-timeout | Timeout long-running tests | Prevent CI hangs from infinite loops |
| pytest-mock | Mocking utilities | Tests requiring mocks/stubs |

**Confidence:** MEDIUM - Common in scientific Python projects, but not universally required

## Installation

```bash
# Build system (required at build time only)
# Specified in pyproject.toml [build-system]
# Users won't install these manually

# Documentation (dev dependency)
pip install sphinx>=9.1.0 sphinx-autodoc-typehints nbsphinx \
    sphinx-book-theme sphinx-copybutton myst-parser

# Code quality (dev dependency)
pip install ruff>=0.15.1 mypy pre-commit>=4.5.1

# Testing (dev dependency)
pip install pytest>=8.0 pytest-cov>=7.0.0 pytest-xdist

# Benchmarking (dev dependency)
pip install pytest-benchmark>=5.2.3 asv

# Profiling (dev dependency)
pip install scalene py-spy

# Package management (global install, not project dependency)
curl -LsSf https://astral.sh/uv/install.sh | sh  # Unix/macOS
# OR: pip install uv  (if pip already available)
```

## Alternatives Considered

| Category | Recommended | Alternative | When to Use Alternative |
|----------|-------------|-------------|-------------------------|
| Build backend | hatchling | setuptools | Complex C extensions, existing setup.py with custom build logic |
| Build backend | hatchling | flit-core | Minimalist pure-Python projects, no plugin needs |
| Doc generator | Sphinx | MkDocs + mkdocstrings | Markdown-only team, prose-heavy docs (books/guides), willing to accept future uncertainty |
| Doc generator | Sphinx | JupyterBook | Notebook-centric projects (courses, interactive books) |
| Linter/formatter | ruff | black + flake8 + isort | Legacy tooling, team refuses to migrate (not recommended) |
| Package manager | uv | pip + pip-tools | Conservative environments, uv not approved |
| Package manager | uv | Poetry | Need lock files + dependency groups + complex version constraints. Poetry 2.0+ supports [project] table. |
| Benchmarking | pytest-benchmark + asv | timeit module | One-off scripts (not production benchmarking) |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| setup.py for config | Deprecated for pure-Python projects. Security risk (code execution). | pyproject.toml with [project] table |
| setup.cfg | Superseded by pyproject.toml (PEP 621). | pyproject.toml |
| Black as standalone tool | Ruff provides identical formatting 100x faster. | ruff format |
| isort | Ruff includes import sorting. | ruff check --select I |
| Flake8 + plugins | Ruff replaces 50+ Flake8 plugins. | ruff check |
| Manual API tokens for PyPI | Security risk. Trusted Publishing is standard. | GitHub OIDC trusted publishing |
| torch.jit.script/trace | Older JIT approach. torch.compile is PyTorch 2.x standard. | torch.compile |
| cProfile for first-pass profiling | Scalene provides more detail with less overhead. | scalene (dev), py-spy (prod) |

## Stack Patterns by Project Type

**Pure-Python scientific library (like AquaMVS):**
- Build: hatchling + uv
- Docs: Sphinx + nbsphinx + Read the Docs
- Quality: ruff + mypy + pre-commit
- Test: pytest + pytest-cov + pytest-benchmark
- CI/CD: GitHub Actions + trusted publishing

**Library with C extensions:**
- Build: setuptools (if complex) or hatchling with plugin
- Rest: same as pure-Python

**Notebook-heavy educational project:**
- Docs: JupyterBook instead of Sphinx
- Rest: same as pure-Python

**Research prototype (not for PyPI):**
- Skip: packaging, PyPI publishing workflow
- Keep: ruff, mypy, pytest, documentation (for reproducibility)

## Version Compatibility

| Package | Requires | Notes |
|---------|----------|-------|
| Sphinx >=9.1.0 | Python >=3.12 | Older Sphinx versions support Python 3.9+ |
| pytest-cov >=7.0.0 | Python >=3.9 | Dropped Python 3.8 support |
| ruff | No Python version constraint | Works with any Python version it's linting |
| uv | System tool | Manages Python versions itself |
| hatchling | Python >=3.8 | Build backend, not runtime dependency |

**Compatibility with existing AquaMVS stack:**
- PyTorch: torch.compile available in PyTorch 2.x (AquaMVS should verify current version)
- All tools compatible with existing kornia, Open3D, OpenCV dependencies
- No conflicts with scientific stack

## Modern Workflow (2026 Best Practices)

1. **Project initialization:**
   ```bash
   uv init --lib aquamvs
   cd aquamvs
   ```

2. **Configure pyproject.toml:**
   - Use [project] table for metadata
   - [build-system] with hatchling
   - [tool.ruff], [tool.mypy], [tool.pytest.ini_options]

3. **Set up pre-commit:**
   ```bash
   uv pip install pre-commit
   # Create .pre-commit-config.yaml with ruff, mypy
   pre-commit install
   ```

4. **Documentation:**
   - Sphinx with nbsphinx for tutorials
   - Host on Read the Docs (auto-build from git)

5. **CI/CD (GitHub Actions):**
   - Run: ruff check, mypy, pytest with coverage
   - Publish to TestPyPI on tag (test run)
   - Publish to PyPI on release (trusted publishing)

6. **Release workflow:**
   - Run benchmarks with asv (compare to previous release)
   - Profile with scalene if performance-critical changes
   - Update changelog, tag release
   - CI auto-publishes to PyPI with attestations

## Key Configuration Files

**pyproject.toml structure:**
```toml
[build-system]
requires = ["hatchling>=1.26"]
build-backend = "hatchling.build"

[project]
name = "aquamvs"
version = "0.1.0"
description = "Underwater multi-view stereo with refraction"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "kornia",
    # ... existing deps
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=7.0",
    "ruff>=0.15.1",
    "mypy",
]
docs = [
    "sphinx>=9.1.0",
    "nbsphinx",
    "sphinx-book-theme",
]

[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "A", "C4", "PT"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
```

**.pre-commit-config.yaml:**
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.1
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

## Migration from Current Setup (AquaMVS)

AquaMVS currently uses Black. Migration path:

1. **Replace Black with Ruff:**
   - Remove: black from dev dependencies
   - Add: ruff>=0.15.1
   - Update pre-commit config (replace black hook with ruff-format)
   - Run: `ruff format .` (identical output to Black)

2. **Consolidate linting:**
   - If using flake8/isort: remove, use `ruff check`
   - Configure ruff rules to match current style

3. **Add benchmarking (new capability):**
   - Install pytest-benchmark
   - Create tests/test_benchmark/ directory
   - Add benchmark fixtures

4. **Documentation setup (if not exists):**
   - Install Sphinx + nbsphinx
   - Create docs/ directory
   - Configure Read the Docs

5. **CI/CD modernization:**
   - Add trusted publishing to PyPI
   - Add benchmark runs on PR (pytest-benchmark)
   - Add coverage reporting

No breaking changes required. Incremental adoption possible.

## Sources

**Official Documentation (HIGH confidence):**
- [Writing pyproject.toml - Python Packaging User Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [Trusted Publishing - PyPI Docs](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
- [Publishing with GitHub Actions - Python Packaging User Guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [uv Documentation](https://docs.astral.sh/uv/)
- [pytest-benchmark Documentation](https://pytest-benchmark.readthedocs.io/)
- [torch.compile Tutorial - PyTorch](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [Sphinx Documentation](https://www.sphinx-doc.org/)

**Scientific Python Community (HIGH confidence):**
- [Writing documentation - Scientific Python Development Guide](https://learn.scientific-python.org/development/guides/docs/)
- [Python Packaging Tools - Scientific Python Guide](https://www.pyopensci.org/python-package-guide/package-structure-code/python-package-build-tools.html)
- [Python Package Documentation Guide](https://www.pyopensci.org/python-package-guide/documentation/hosting-tools/publish-documentation-online.html)

**PyPI Verified Versions (HIGH confidence):**
- [ruff 0.15.1](https://pypi.org/project/ruff/) - Released 2026-02-12
- [pytest-benchmark 5.2.3](https://pypi.org/project/pytest-benchmark/) - Released 2025-11-09
- [Sphinx 9.1.0](https://pypi.org/project/sphinx/) - Released 2025-12-31
- [pre-commit 4.5.1](https://pypi.org/project/pre-commit/) - Released 2025-12-16
- [MkDocs 1.6.1](https://pypi.org/project/mkdocs/) - Released 2024-08-30

**Community Best Practices 2026 (MEDIUM-HIGH confidence):**
- [Python Packaging Best Practices 2026](https://dasroot.net/posts/2026/01/python-packaging-best-practices-setuptools-poetry-hatch/)
- [Modern Python Code Quality Setup](https://simone-carolini.medium.com/modern-python-code-quality-setup-uv-ruff-and-mypy-8038c6549dcc)
- [Python Build Backends in 2025](https://medium.com/@dynamicy/python-build-backends-in-2025-what-to-use-and-why-uv-build-vs-hatchling-vs-poetry-core-94dd6b92248f)
- [GitHub Actions CI/CD Complete Guide 2026](https://devtoolbox.dedyn.io/blog/github-actions-cicd-complete-guide)

**Performance & Optimization (MEDIUM-HIGH confidence):**
- [Python Profiling: Scalene and py-spy](https://github.com/plasma-umass/scalene)
- [py-spy GitHub](https://github.com/benfred/py-spy)
- [JIT PyTorch Training Performance Guide](https://residentmario.github.io/pytorch-training-performance-guide/jit.html)

---

*Stack research for: AquaMVS Production Readiness*
*Researched: 2026-02-14*
*Overall confidence: HIGH*
