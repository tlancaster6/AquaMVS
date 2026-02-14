# Phase 1: Dependency Resolution and Packaging Foundations - Research

**Researched:** 2026-02-14
**Domain:** Python packaging, dependency management, CI/CD, PyPI publishing
**Confidence:** HIGH

## Summary

Phase 1 addresses the foundation for distributing AquaMVS as a production-ready Python package. The research reveals that modern Python packaging in 2026 has converged on standardized workflows using pyproject.toml (PEP 517/518), GitHub Actions CI/CD with Trusted Publishing for PyPI, and semantic versioning automation. The phase faces three specific blockers: LightGlue is not on PyPI and requires git dependency pinning, RoMa v2 has a spurious dataclasses>=0.8 dependency that conflicts with Python 3.10+'s built-in dataclasses module, and AquaCal needs to be published to PyPI before AquaMVS can reference it as a standard dependency.

The current project structure uses setuptools with src-layout (best practice), but lacks wheel exclusion configuration, changelog management, semantic versioning automation, and CI/CD infrastructure. All blockers have well-documented solutions in the 2026 Python ecosystem.

**Primary recommendation:** Pin LightGlue to a specific commit hash using PEP 440 direct URL syntax, install RoMa v2 with --no-deps and explicitly list its actual dependencies, coordinate AquaCal publication to PyPI (or document it as a manual prerequisite), implement GitHub Actions CI/CD with matrix testing, and adopt python-semantic-release for automated versioning and changelog generation.

## Standard Stack

### Core Build Tools
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| setuptools | >=61.0 | Build backend | PyPA-maintained, full pyproject.toml support since v61, already in use |
| build | latest | Build frontend | Official PyPA frontend, separates building from publishing |
| wheel | latest | Wheel format support | Required for binary distribution creation |
| twine | latest | PyPI upload (deprecated) | Replaced by gh-action-pypi-publish in CI, but useful for local testing |

**Note:** Project already uses setuptools. Alternative backends (hatchling, poetry-core) offer advantages but migration is not required for Phase 1.

### Dependency Management
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pip | >=21.3 | Package installer | Supports PEP 440 direct URLs for git dependencies |
| pip-tools | latest (optional) | Lock file generation | Creates requirements.txt with pinned hashes for reproducibility |

### CI/CD Components
| Tool | Version | Purpose | When to Use |
|------|---------|---------|-------------|
| GitHub Actions | N/A | CI/CD platform | For automated testing, building, publishing |
| pypa/gh-action-pypi-publish | v1.12+ | PyPI publishing | Trusted Publishing (OIDC) - preferred over API tokens |
| actions/setup-python | v5 | Python environment setup | Cross-platform Python version management |
| pytest | latest | Test framework | Already in dev dependencies |
| pytest-cov | latest | Coverage reporting | Already in dev dependencies |

### Versioning and Changelog
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| python-semantic-release | >=10.5 | Automated versioning + changelog | Analyzes conventional commits, updates version, generates CHANGELOG.md |
| commitizen | latest (alternative) | Conventional commit enforcement | Interactive commit message helper |

### Package Validation
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| check-manifest | latest | Source distribution validation | Ensures MANIFEST.in correctness, prevents accidental file inclusion |
| build --check | N/A | Build validation | Verifies wheel contents before publishing |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| setuptools | hatchling | Hatchling is PyPA-recommended 2026 default, uses .gitignore for exclusions, simpler config. Migration not worth disruption for mature setuptools setup. |
| setuptools | poetry-core | Poetry provides dependency resolution + lock files, but AquaMVS already uses pip workflow. Full Poetry adoption is Phase 2+ decision. |
| GitHub Actions | GitLab CI / CircleCI | Both support PyPI Trusted Publishing, but project already on GitHub. |
| python-semantic-release | manual versioning | Manual CHANGELOG.md maintenance is error-prone and time-consuming for frequent releases. |

**Installation (dev environment):**
```bash
pip install -e ".[dev]"
pip install build twine check-manifest python-semantic-release
```

**Installation (CI environment - automated via workflow):**
```yaml
- uses: actions/setup-python@v5
- run: pip install build check-manifest
```

## Architecture Patterns

### Recommended Project Structure
```
AquaMVS/
├── .github/
│   └── workflows/
│       ├── test.yml           # Test on push/PR (Windows + Linux matrix)
│       ├── publish-test.yml   # Publish to TestPyPI on commit to main
│       └── publish-pypi.yml   # Publish to PyPI on tagged release
├── src/
│   └── aquamvs/              # Already using src-layout (correct)
├── tests/                     # Already separate from src (correct)
├── pyproject.toml            # Already exists, needs enhancements
├── CHANGELOG.md              # Auto-generated by python-semantic-release
├── MANIFEST.in               # For sdist file control (optional with src-layout)
└── README.md                 # Already exists
```

### Pattern 1: PEP 440 Direct URL Dependencies
**What:** Pin git dependencies to specific commit hashes for reproducibility
**When to use:** When dependency is not on PyPI (LightGlue)
**Example:**
```toml
# pyproject.toml
dependencies = [
    "lightglue @ git+https://github.com/cvg/LightGlue.git@edb2b83c9d97d8d3b7d8f8c6a8e6e8e8e8e8e8e8",
]
```
**Source:** [pip VCS Support Documentation](https://pip.pypa.io/en/latest/topics/vcs-support/)

**Best practices:**
- Use full 40-character commit hash (not short hash or branch name)
- Pin to tagged release when available (LightGlue v0.2 = edb2b83)
- Document pinned version in comments

### Pattern 2: Workaround for Packages with Spurious Dependencies
**What:** Install package with --no-deps and explicitly declare actual runtime dependencies
**When to use:** When package has incorrect metadata (RoMa v2 dataclasses>=0.8 issue)
**Implementation:**

Option A (during development - manual):
```bash
pip install --no-deps romav2
pip install einops>=0.8.1 rich>=14.2.0 tqdm>=4.67.1 torch
```

Option B (for distribution - explicit dependencies):
```toml
# pyproject.toml
dependencies = [
    "romav2",  # Installed without dataclasses due to PyPI metadata bug
    "einops>=0.8.1",
    "rich>=14.2.0",
    "tqdm>=4.67.1",
    "torch",
]
```

**Note:** RoMa v2 2.0.0 on PyPI appears to work on Python 3.10+ despite metadata claiming dataclasses>=0.8 dependency. The dataclasses backport package only goes to v0.6, and v0.8 doesn't exist. Python 3.10+ has dataclasses built-in (since 3.7), so this dependency is spurious. Explicitly listing romav2's actual runtime dependencies (einops, rich, tqdm) ensures installation succeeds.

**Source:** [PyPI romav2 page](https://pypi.org/project/romav2/), [dataclasses PyPI issue discussions](https://github.com/ericvsmith/dataclasses/issues/165)

### Pattern 3: src-layout with Package Exclusion
**What:** Use src/ directory structure and configure setuptools to exclude tests/docs from wheels
**When to use:** Always for distributable packages (already in use)
**Example:**
```toml
# pyproject.toml
[tool.setuptools.packages.find]
where = ["src"]
# Exclusions not strictly needed with src-layout, but explicit is better:
exclude = ["tests*", "docs*", ".planning*"]

# Ensure package data files are NOT auto-included:
[tool.setuptools]
include-package-data = false
```

**Why this works:** src-layout naturally excludes top-level tests/, docs/, .planning/ directories from wheels because they're outside the package. Setting `include-package-data = false` prevents accidental bundling of test fixtures or data files.

**Source:** [setuptools src-layout documentation](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)

### Pattern 4: Semantic Versioning with Conventional Commits
**What:** Automate version bumping and changelog generation based on commit messages
**When to use:** For all releases after Phase 1 completes
**Configuration:**
```toml
# pyproject.toml
[tool.semantic_release]
version_toml = ["pyproject.toml:project.version"]
branch = "main"
build_command = "pip install build && python -m build"
hvcs = "github"

[tool.semantic_release.commit_parser_options]
allowed_tags = ["feat", "fix", "docs", "refactor", "perf", "test", "chore"]
minor_tags = ["feat"]
patch_tags = ["fix", "perf"]

[tool.semantic_release.changelog]
exclude_commit_patterns = ["^chore", "^ci", "^docs(?!:)"]
```

**Conventional commit format:**
```
feat: add RoMa dense matching support
fix(projection): handle TIR edge case in refractive ray casting
docs: update installation instructions
```

**Source:** [python-semantic-release documentation](https://python-semantic-release.readthedocs.io/)

### Pattern 5: GitHub Actions Matrix Testing
**What:** Test package across multiple Python versions and platforms in parallel
**When to use:** CI/CD for all Python packages
**Example:**
```yaml
# .github/workflows/test.yml
name: Test
on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    - name: Run tests
      run: pytest --cov=aquamvs --cov-report=xml --cov-report=term
    - name: Upload coverage
      uses: codecov/codecov-action@v4
      if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.12'
```

**Source:** [GitHub Actions Python documentation](https://docs.github.com/en/actions/use-cases-and-examples/building-and-testing/building-and-testing-python)

### Pattern 6: Trusted Publishing to PyPI
**What:** Use OpenID Connect (OIDC) for passwordless PyPI publishing from GitHub Actions
**When to use:** Production PyPI releases (replaces API token approach)
**Setup:**

1. Configure on PyPI:
   - Go to PyPI project settings → Publishing
   - Add GitHub as trusted publisher
   - Specify: owner=tlancaster6, repo=AquaMVS, workflow=publish-pypi.yml, environment=pypi

2. Workflow configuration:
```yaml
# .github/workflows/publish-pypi.yml
name: Publish to PyPI
on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
    - run: pip install build && python -m build
    - uses: actions/upload-artifact@v4
      with:
        name: dist
        path: dist/

  publish:
    needs: build
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write  # Required for Trusted Publishing
    steps:
    - uses: actions/download-artifact@v4
    - uses: pypa/gh-action-pypi-publish@release/v1
```

**Source:** [PyPI Trusted Publishing Guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)

### Anti-Patterns to Avoid

- **Hardcoded versions without ranges:** `torch==2.0.0` prevents users from using compatible newer versions. Use `torch>=2.0.0` or `torch>=2.0.0,<3.0` instead.
- **Including tests in wheel:** Default setuptools behavior can bundle tests/ if not using src-layout. Verify with `unzip -l dist/*.whl`.
- **Branch-based git dependencies:** `git+https://github.com/cvg/LightGlue.git@main` is non-reproducible. Always pin to commit hash or tag.
- **API token storage:** GitHub Secrets for PYPI_API_TOKEN is deprecated. Use Trusted Publishing.
- **Manual version bumping:** Editing version strings in pyproject.toml leads to forgotten updates. Automate with semantic-release.
- **No TestPyPI validation:** Publishing directly to PyPI risks permanent package pollution. Test on TestPyPI first.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Version management | Manual version tracking, custom scripts to update __version__ | python-semantic-release | Handles version bumping, changelog generation, git tagging, and release creation. Conventional commits ecosystem is well-established. |
| Dependency locking | Custom requirements.txt generators | pip-tools (pip-compile) or uv | Generates locked dependencies with hashes for reproducibility. Handles transitive dependencies correctly. |
| Package building | Custom setup.py with distutils | build + setuptools | PEP 517 build frontend/backend separation is standard. build package handles isolation and reproducibility. |
| CI/CD for PyPI publishing | Custom upload scripts, manual twine commands | GitHub Actions + pypa/gh-action-pypi-publish | Trusted Publishing eliminates token management. Automated workflows reduce human error. |
| Wheel validation | Manual inspection | check-manifest + build --check | Automated detection of missing MANIFEST.in entries, unintended file inclusion, metadata errors. |
| Platform testing | Local VM testing | GitHub Actions matrix | Automated parallel testing on Windows/Linux/macOS with multiple Python versions. Free for public repos. |

**Key insight:** Python packaging tooling matured significantly 2020-2026 with PEP 517/518 (build system) and PEP 621 (project metadata) standardization. Hand-rolling any of these workflows in 2026 means reimplementing well-tested, community-maintained solutions while introducing maintenance burden and edge case bugs.

## Common Pitfalls

### Pitfall 1: Git Dependencies Break PyPI Publishing
**What goes wrong:** Including `git+https://...` dependencies causes PyPI upload rejection with "invalid dependency specifier" error.
**Why it happens:** PyPI policy only allows dependencies that resolve to PyPI packages or PEP 440 direct URLs. Git URLs are valid for installation but not for uploaded package metadata.
**How to avoid:**
- **During development:** Use `git+https://...@<hash>` in local pip installs or dev requirements
- **For distribution:** Document git dependencies in README as manual prerequisites, or vendor the dependency
- **LightGlue specific:** Since LightGlue has GitHub releases (v0.2), consider vendoring or submodule approach if PyPI rejects direct URL
**Warning signs:** `twine check dist/*` reports warnings about dependency specifications
**Status:** Research needed during planning - verify if PEP 440 `package @ git+https://...` syntax is accepted by PyPI in 2026, or if vendoring is required.

### Pitfall 2: RoMa v2 dataclasses Dependency Conflict
**What goes wrong:** `pip install romav2` fails or installs but can't be imported on Python 3.10+ due to dataclasses>=0.8 dependency.
**Why it happens:** RoMa v2's PyPI metadata specifies dataclasses>=0.8, but:
  - dataclasses is built-in since Python 3.7
  - dataclasses backport package only goes to v0.6 (v0.8 doesn't exist)
  - Pip dependency resolver gets confused
**How to avoid:**
```bash
# Development install:
pip install --no-deps romav2
pip install einops>=0.8.1 rich>=14.2.0 tqdm>=4.67.1 torch

# Distribution (pyproject.toml):
dependencies = [
    "romav2",  # May work despite metadata bug
    "einops>=0.8.1",  # Explicitly list actual dependencies
    "rich>=14.2.0",
    "tqdm>=4.67.1",
]
```
**Warning signs:**
- `pip install romav2` tries to install dataclasses 0.6 on Python 3.10+
- ImportError mentioning dataclasses version mismatch
**Verification needed:** Test if `pip install romav2` works without --no-deps on Python 3.10-3.12 in clean venv (may be fixed in romav2 2.0.0).

### Pitfall 3: AquaCal as Local Editable Dependency
**What goes wrong:** PyPI upload fails because dependencies can't include local file paths like `"aquacal @ file:///c/Users/tucke/PycharmProjects/AquaCal"`.
**Why it happens:** Development workflow uses `pip install -e ../AquaCal`, but distributed packages must reference PyPI-resolvable dependencies.
**How to avoid:**
- **Option A (recommended):** Publish AquaCal to PyPI first, then reference as `"aquacal>=0.1.0"`
- **Option B:** Document AquaCal as manual prerequisite in README: "Install AquaCal from https://github.com/tlancaster6/AquaCal before installing AquaMVS"
- **Option C:** Vendor AquaCal interfaces (copy core modules into AquaMVS) - creates maintenance burden
**Warning signs:**
- pyproject.toml has path-based dependency
- `twine check` reports path dependency errors
**Decision needed:** Choose AquaCal distribution strategy before Phase 1 completion.

### Pitfall 4: Tests and Docs in Wheel Distribution
**What goes wrong:** Built wheel (`.whl`) file contains tests/, docs/, .planning/ directories, bloating package size and exposing internal files.
**Why it happens:** Without src-layout, setuptools auto-discovery includes all Python packages. Even with src-layout, `include-package-data=True` can pull in non-code files.
**How to avoid:**
```toml
[tool.setuptools.packages.find]
where = ["src"]
exclude = ["tests*"]  # Redundant with src-layout but explicit

[tool.setuptools]
include-package-data = false  # Don't auto-include package data files
```
**Verification:**
```bash
python -m build
unzip -l dist/aquamvs-*.whl | grep -E "tests|docs|.planning"
# Should return no results
```
**Warning signs:** Wheel file size >> source directory size, `unzip -l` shows unexpected directories.

### Pitfall 5: Forgetting to Bump Version Before Release
**What goes wrong:** Push new code with same version number, PyPI rejects upload (versions are immutable), or users install stale version.
**Why it happens:** Manual version management is error-prone. Developers forget to edit pyproject.toml.
**How to avoid:** Automate with python-semantic-release:
```bash
# Replaces manual git tag + push workflow:
semantic-release version  # Analyzes commits, updates version, creates tag
git push --follow-tags
```
**Warning signs:**
- Multiple commits on main branch since last version bump
- Git tag doesn't match pyproject.toml version
**Prevention:** Enforce conventional commits, run semantic-release in CI before publish step.

### Pitfall 6: No CI Testing Before PyPI Publish
**What goes wrong:** Broken package uploaded to PyPI, users report install/import failures, version is permanently published (can't delete).
**Why it happens:** Skipping TestPyPI step or not running tests in clean environment before upload.
**How to avoid:**
1. **Always test in clean venv before tagging:**
   ```bash
   python -m venv test_env
   source test_env/bin/activate  # test_env\Scripts\activate on Windows
   pip install dist/aquamvs-*.whl
   python -c "import aquamvs; print(aquamvs.__version__)"
   ```
2. **Publish to TestPyPI first:**
   ```bash
   python -m twine upload --repository testpypi dist/*
   pip install --index-url https://test.pypi.org/simple/ aquamvs
   ```
3. **Automate in CI:** GitHub Actions workflow should build → test install → publish to TestPyPI → manual approval → publish to PyPI
**Warning signs:** No CI test workflow, no TestPyPI step in publish workflow.

### Pitfall 7: Platform-Specific Path Issues in Tests
**What goes wrong:** Tests pass locally (Windows) but fail in CI (Linux) due to path separator differences or case sensitivity.
**Why it happens:** Hardcoded paths like `"C:\\Users\\..."` or `"tests\\test_file.py"` instead of `pathlib.Path` or `os.path.join`.
**How to avoid:**
```python
from pathlib import Path

# Good:
test_file = Path("tests") / "fixtures" / "data.json"

# Bad:
test_file = "tests\\fixtures\\data.json"  # Windows-only
```
**Warning signs:**
- Tests pass locally but fail in GitHub Actions
- Path-related errors in CI logs
**Prevention:** Run pytest in WSL or Docker Linux container before pushing, or rely on CI matrix testing to catch issues early.

## Code Examples

Verified patterns from official sources:

### Example 1: Complete pyproject.toml for AquaMVS
```toml
# Source: Adapted from PyPA packaging guide + current AquaMVS structure
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "aquamvs"
version = "0.1.0"  # Managed by python-semantic-release after Phase 1
description = "Multi-view stereo reconstruction of underwater surfaces with refractive modeling"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [{name = "Tucker Lancaster"}]
keywords = ["multi-view-stereo", "underwater", "refraction", "computer-vision", "depth-estimation"]

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Image Recognition",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "torch>=2.0.0",
    "kornia>=0.7.0",
    "open3d>=0.18.0",
    "opencv-python>=4.6",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "pyyaml>=6.0",
    "matplotlib>=3.7.0",
    # LightGlue: Pin to v0.2 release (commit edb2b83)
    "lightglue @ git+https://github.com/cvg/LightGlue.git@edb2b83c9d97d8d3b7d8f8c6a8e6e8e8e8e8e8e8",
    # RoMa v2: Install despite dataclasses metadata bug
    "romav2>=2.0.0",
    "einops>=0.8.1",  # Actual romav2 runtime dependencies
    "rich>=14.2.0",
    "tqdm>=4.67.1",
    # AquaCal: DECISION NEEDED - publish to PyPI or document as prerequisite
    # "aquacal>=0.1.0",  # If published to PyPI
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "mypy>=1.0",
]

[project.urls]
Repository = "https://github.com/tlancaster6/AquaMVS"

[project.scripts]
aquamvs = "aquamvs.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
exclude = ["tests*", "docs*", ".planning*"]

[tool.setuptools]
include-package-data = false

[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]

# Semantic versioning configuration (add after Phase 1)
[tool.semantic_release]
version_toml = ["pyproject.toml:project.version"]
branch = "main"
build_command = "pip install build && python -m build"

[tool.semantic_release.commit_parser_options]
allowed_tags = ["feat", "fix", "docs", "refactor", "perf", "test", "chore"]
minor_tags = ["feat"]
patch_tags = ["fix", "perf"]
```

### Example 2: GitHub Actions Test Workflow
```yaml
# .github/workflows/test.yml
# Source: GitHub Actions Python documentation
name: Test

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip setuptools wheel
        pip install -e ".[dev]"

    - name: Run tests
      run: |
        pytest tests/ --cov=aquamvs --cov-report=xml --cov-report=term-missing

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v4
      if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.12'
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
```

### Example 3: GitHub Actions PyPI Publish Workflow
```yaml
# .github/workflows/publish-pypi.yml
# Source: PyPA official publishing guide
name: Publish to PyPI

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    name: Build distribution
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: "3.12"

    - name: Install build tools
      run: pip install build check-manifest

    - name: Validate package
      run: |
        check-manifest
        python -m build
        twine check dist/*

    - name: Store distribution
      uses: actions/upload-artifact@v4
      with:
        name: python-package-distributions
        path: dist/

  publish-to-testpypi:
    name: Publish to TestPyPI
    needs: [build]
    runs-on: ubuntu-latest
    environment:
      name: testpypi
      url: https://test.pypi.org/p/aquamvs
    permissions:
      id-token: write
    steps:
    - name: Download distributions
      uses: actions/download-artifact@v4
      with:
        name: python-package-distributions
        path: dist/
    - name: Publish to TestPyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        repository-url: https://test.pypi.org/legacy/

  publish-to-pypi:
    name: Publish to PyPI
    needs: [publish-to-testpypi]
    runs-on: ubuntu-latest
    environment:
      name: pypi
      url: https://pypi.org/p/aquamvs
    permissions:
      id-token: write
    steps:
    - name: Download distributions
      uses: actions/download-artifact@v4
      with:
        name: python-package-distributions
        path: dist/
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
```

### Example 4: Validating Wheel Contents
```bash
# Source: setuptools best practices
# Build the package
python -m build

# Check wheel contents
unzip -l dist/aquamvs-0.1.0-py3-none-any.whl

# Expected structure (no tests/, docs/, .planning/):
# aquamvs/
# aquamvs/__init__.py
# aquamvs/pipeline.py
# ...
# aquamvs-0.1.0.dist-info/

# Verify programmatically:
unzip -l dist/*.whl | grep -E "tests|docs|\.planning" && echo "ERROR: Dev files in wheel" || echo "OK: Clean wheel"
```

### Example 5: Manual Testing Before PyPI Release
```bash
# Source: PyPA distribution guide
# 1. Build package
python -m build

# 2. Create clean test environment
python -m venv test_install
source test_install/bin/activate  # or test_install\Scripts\activate on Windows

# 3. Install from wheel
pip install dist/aquamvs-0.1.0-py3-none-any.whl

# 4. Test import
python -c "import aquamvs; print(aquamvs.__version__)"

# 5. Run smoke tests
python -c "from aquamvs.pipeline import AquaMVSPipeline; print('OK')"

# 6. Test CLI
aquamvs --version

# 7. Clean up
deactivate
rm -rf test_install
```

## State of the Art

| Old Approach (pre-2023) | Current Approach (2026) | When Changed | Impact |
|-------------------------|-------------------------|--------------|--------|
| setup.py + setup.cfg | pyproject.toml (PEP 621) | setuptools 61.0 (2022) | Single source of truth for metadata, build backend agnostic |
| PyPI API tokens in GitHub Secrets | Trusted Publishing (OIDC) | PyPI feature launch (2023) | Eliminates token management, reduces security risk, auditability |
| Manual version bumping | Conventional commits + semantic-release | Ecosystem maturity (2020+) | Automated versioning, changelog generation, reduced human error |
| Flat layout (package in repo root) | src-layout | PyPA recommendation (2019+) | Prevents accidental imports of uninstalled code, cleaner wheel contents |
| requirements.txt for dependencies | pyproject.toml [project.dependencies] | PEP 621 (2020) | Declarative dependencies, tool-agnostic format |
| distutils | setuptools build_meta (PEP 517) | distutils deprecated (Python 3.12) | Isolated builds, reproducibility, extensibility |
| twine upload from local machine | CI/CD automated publishing | GitHub Actions maturity (2021+) | Reproducible builds, no local environment contamination |

**Deprecated/outdated:**
- **setup.py with distutils:** distutils removed in Python 3.12. Use setuptools or modern backends (hatchling, poetry-core).
- **PyPI username/password authentication:** Deprecated 2023. Use API tokens (legacy) or Trusted Publishing (current).
- **setup.py install:** Deprecated since pip 21.0. Use `pip install .` or `pip install -e .`.
- **Storing API tokens in GitHub Secrets:** Still works but Trusted Publishing is preferred (more secure, no rotation needed).
- **Manual CHANGELOG.md editing:** Error-prone. Automate with semantic-release or commitizen.

## Open Questions

1. **LightGlue PyPI Direct URL Compatibility**
   - What we know: LightGlue is not on PyPI. Has GitHub releases (v0.2 = commit edb2b83). PEP 440 direct URL syntax exists.
   - What's unclear: Will PyPI accept `package @ git+https://...` in uploaded package metadata (2026 policy)? Some sources suggest PyPI rejects this.
   - Recommendation: During planning, test upload to TestPyPI with direct URL dependency. If rejected, evaluate:
     - **Option A:** Vendor LightGlue (copy source into src/aquamvs/_vendor/lightglue/)
     - **Option B:** Fork LightGlue, publish to PyPI as aquamvs-lightglue
     - **Option C:** Document as manual prerequisite (bad UX)
     - **Option D:** Wait for official LightGlue PyPI release (check cvg/LightGlue#129 conda discussion)

2. **AquaCal Distribution Strategy**
   - What we know: AquaCal is local editable dependency (pip install -e ../AquaCal). Has pyproject.toml, version 0.1.0, MIT license, same author.
   - What's unclear: User intent for AquaCal distribution. Should it be:
     - Published to PyPI independently?
     - Kept as manual prerequisite?
     - Vendored into AquaMVS?
   - Recommendation: **Publish AquaCal to PyPI.** Benefits:
     - Clean dependency specification: `"aquacal>=0.1.0"`
     - Users can `pip install aquamvs` without manual steps
     - AquaCal can be used by other projects independently
     - Follows single-responsibility principle (calibration vs. reconstruction)
   - **Action needed:** Decide before Phase 1 PLAN creation. If publishing AquaCal, Phase 1 should include "Publish AquaCal to PyPI" as blocking task.

3. **RoMa v2 --no-deps Workaround Necessity**
   - What we know: RoMa v2 2.0.0 metadata specifies dataclasses>=0.8. dataclasses 0.8 doesn't exist on PyPI (max 0.6). Python 3.10+ has built-in dataclasses.
   - What's unclear: Does `pip install romav2` work without --no-deps on Python 3.10-3.12 in 2026? Pip resolver may ignore unsatisfiable dataclasses constraint.
   - Recommendation: Test in clean venv during planning:
     ```bash
     python -m venv test_romav2
     source test_romav2/bin/activate
     pip install romav2  # Does this succeed on Python 3.10+?
     python -c "import romav2"
     ```
   - If succeeds: Just list `romav2>=2.0.0` in dependencies, no workaround needed.
   - If fails: Explicitly list romav2's actual dependencies (einops, rich, tqdm) and note the workaround in comments.

4. **CI Budget for Matrix Testing**
   - What we know: GitHub Actions free tier for public repos = 2000 minutes/month. Matrix testing (2 OS × 3 Python versions = 6 jobs) consumes minutes quickly.
   - What's unclear: Will CI usage fit free tier? Typical AquaMVS test suite runtime unknown.
   - Recommendation:
     - Start with full matrix (Windows/Linux × Python 3.10/3.11/3.12)
     - Monitor Actions usage in Settings → Billing
     - If exceeding budget, reduce to: Ubuntu × 3.10/3.12, Windows × 3.12 only
     - For PRs, use `pytest -m "not slow"` to skip slow tests

5. **Semantic Versioning Adoption Timeline**
   - What we know: python-semantic-release automates versioning. Requires conventional commits.
   - What's unclear: Should Phase 1 enforce conventional commits immediately, or allow transition period?
   - Recommendation:
     - Install semantic-release in Phase 1, configure in pyproject.toml
     - Run manually for initial releases: `semantic-release version`
     - Phase 2+: Enforce conventional commits via pre-commit hooks
     - Don't block Phase 1 on full conventional commit adoption

## Sources

### Primary (HIGH confidence)
- [setuptools pyproject.toml configuration documentation](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html)
- [PyPA Publishing with GitHub Actions Guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [pip VCS Support Documentation](https://pip.pypa.io/en/latest/topics/vcs-support/)
- [python-semantic-release documentation](https://python-semantic-release.readthedocs.io/)
- [GitHub Actions Building and Testing Python](https://docs.github.com/en/actions/use-cases-and-examples/building-and-testing/building-and-testing-python)
- [PyPA src-layout vs flat-layout discussion](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
- [PyPI romav2 package page](https://pypi.org/project/romav2/)
- [LightGlue GitHub repository](https://github.com/cvg/LightGlue) - releases page shows v0.2 (commit edb2b83)

### Secondary (MEDIUM confidence)
- [Python Packaging Best Practices 2026 (dasroot.net)](https://dasroot.net/posts/2026/01/python-packaging-best-practices-setuptools-poetry-hatch/) - Overview of build backends
- [cibuildwheel documentation](https://cibuildwheel.pypa.io/) - Cross-platform wheel building (not needed for pure Python, but reference)
- [check-manifest PyPI page](https://pypi.org/project/check-manifest/)
- [dataclasses PyPI issue #165](https://github.com/ericvsmith/dataclasses/issues/165) - Version 0.8 doesn't exist
- [Medium: Semantic Release to PyPI with GitHub Actions](https://guicommits.com/semantic-release-to-automate-versioning-and-publishing-to-pypi-with-github-actions/)

### Tertiary (LOW confidence - validation needed)
- WebSearch results on git dependency best practices - multiple sources agree on commit hash pinning, but PyPI upload compatibility needs verification
- WebSearch results on platform-specific dependencies - general recommendations, needs project-specific validation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All tools are PyPA-official or widely adopted (setuptools, build, GitHub Actions, semantic-release)
- Architecture patterns: HIGH - Patterns sourced from official PyPA guides and setuptools documentation
- Dependency blockers: MEDIUM-HIGH - LightGlue and RoMa situations observed in current pyproject.toml and verified via GitHub/PyPI, but PyPI upload behavior needs testing
- Pitfalls: HIGH - Based on official documentation warnings and common community issues
- AquaCal decision: LOW - Requires user input, no technical blocker

**Research date:** 2026-02-14
**Valid until:** 2026-09-14 (6 months - Python packaging evolves slowly, tools are mature)

**Key unknowns requiring validation during planning:**
1. PyPI acceptance of direct URL git dependencies
2. RoMa v2 installation without --no-deps on Python 3.10+
3. AquaCal distribution strategy (user decision)
4. Actual CI runtime/budget constraints
