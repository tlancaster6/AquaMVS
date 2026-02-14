# Pitfalls Research: Scientific Python Package Productionization

**Domain:** Scientific Python library (underwater MVS reconstruction)
**Researched:** 2026-02-14
**Confidence:** HIGH

## Critical Pitfalls

### Pitfall 1: Git Dependencies Without Commit Pins

**What goes wrong:**
Package depends on LightGlue via `git+https://github.com/cvg/LightGlue.git` without a commit hash. GitHub repository changes or disappears, breaking all future installs. PyPI rejects packages with unpinned git dependencies.

**Why it happens:**
Git dependencies are convenient during development but become time bombs in production. Developers assume upstream repos remain stable and accessible forever.

**How to avoid:**
Pin to specific commit hash: `git+https://github.com/cvg/LightGlue.git@abc123def456`. Better: fork and vendor the dependency, or wait for PyPI release. Before publishing to PyPI, resolve all git dependencies to either PyPI packages or commit-pinned URLs.

**Warning signs:**
- Dependency specifier contains `git+` without `@<hash>`
- `pip install` fails intermittently with "cannot find compatible version"
- CI builds break when upstream repo force-pushes or renames branches

**Phase to address:**
Phase 1 (Dependency Resolution) - Must resolve before PyPI publication. LightGlue specifically needs either: wait for PyPI release, vendor the code, or document that users must install from git separately.

---

### Pitfall 2: Local Editable Dependencies in Production Package

**What goes wrong:**
Package declares dependency on AquaCal via `path = "../AquaCal"` or `-e ../AquaCal`. PyPI upload fails because path dependencies are not allowed. Users cannot install package because the local path doesn't exist on their systems.

**Why it happens:**
Editable dependencies are essential for development but fundamentally incompatible with distributing packages. The project.dependencies standard does not support developer-oriented information like editable installations and relative paths.

**How to avoid:**
For local dependencies not on PyPI: (1) Publish dependency to PyPI first, or (2) instruct users to install from git with pinned version, or (3) vendor the required code if small and stable. Document the installation order in README. Use tool-specific dev-dependency sections for editable installs during development.

**Warning signs:**
- `pyproject.toml` contains `path =` in dependencies
- `pip install .` works but `pip install dist/aquamvs-*.whl` fails
- PyPI upload rejected with "invalid dependency specifier"

**Phase to address:**
Phase 1 (Dependency Resolution) - Critical blocker. Must either: publish AquaCal to PyPI first, or extract only the needed interfaces and vendor them (if license permits and coupling is minimal), or document AquaCal as a prerequisite installation step.

---

### Pitfall 3: Platform-Specific Hacks Hardcoded in Package

**What goes wrong:**
Triton import workaround hardcoded for Windows (`sys.modules["triton"] = None`). Package installs on Linux and attempts workaround unnecessarily, or worse, breaks when Triton is actually needed on Linux GPU systems.

**Why it happens:**
Windows-specific issues are solved with global workarounds instead of platform-conditional code. Developer tests on Windows only and doesn't notice Linux impact.

**How to avoid:**
Use platform markers in dependencies: `triton >= 2.0; sys_platform == 'linux'` to conditionally install. Or use runtime platform checks: `if sys.platform == 'win32': sys.modules["triton"] = None`. Environment markers allow declaring dependencies like `"pywin32 >= 1.0; platform_system=='Windows'"` to only install on Windows.

**Warning signs:**
- Unconditional `sys.modules` manipulation at module level
- Import guards with `try/except ImportError` that swallow errors globally
- Bug reports from users on different platforms than primary development OS

**Phase to address:**
Phase 2 (Refactoring) - Include in cleanup sweep. Audit all platform-specific code and convert to conditional checks with clear documentation of why each workaround exists and which platforms require it.

---

### Pitfall 4: Test Directories Installed to site-packages

**What goes wrong:**
`tests/` directory is included in wheel at top-level. Users installing package get `site-packages/tests/` polluting their Python environment. Name collision occurs when multiple packages install to `site-packages/tests/`, creating a mish-mash of unrelated test code.

**Why it happens:**
Default setuptools behavior includes all directories. Developers don't verify wheel contents before distribution. A common mistake is to put tests in a tests/ directory at the root of your project (outside of your Python package) and then include this directory in your project's wheels.

**How to avoid:**
In pyproject.toml, exclude test directories from wheel: configure package discovery to only include `src/aquamvs`. Use `where = ["src"]` for setuptools, or `packages = [{include = "aquamvs", from = "src"}]` for Poetry. Verify wheel contents: `unzip -l dist/*.whl | grep tests` should return empty.

**Warning signs:**
- `unzip -l dist/*.whl` shows `tests/` or `test/` at root level
- `examples/`, `docs/`, `data/` directories appear in wheel listing
- Users report conflicts with other packages' test modules

**Phase to address:**
Phase 3 (Packaging Configuration) - Verify during initial packaging setup. Add check to CI that inspects wheel contents and fails if forbidden directories are present.

---

### Pitfall 5: `--no-deps` Workaround Becomes Installation Requirement

**What goes wrong:**
RoMa v2 requires installation with `--no-deps` due to dependency conflicts. Documentation tells users to use `--no-deps`, which disables all dependency resolution. Users installing AquaMVS miss other required dependencies or get broken transitive dependencies.

**Why it happens:**
Upstream package (RoMa) declares incompatible constraints (e.g., requires numpy<2.0 while PyTorch requires numpy>=2.0). Quick workaround is `--no-deps` which bypasses the resolver, but this creates a support nightmare.

**How to avoid:**
Never document `--no-deps` as installation instruction. Instead: (1) vendor the dependency if feasible, (2) fork and fix dependency constraints, (3) use dependency groups to separate conflicting dependencies, or (4) wait for upstream fix and pin to known-working versions. If temporary workaround is unavoidable, provide explicit list of all required versions so users can install manually.

**Warning signs:**
- README contains `pip install --no-deps`
- GitHub issues report "package X not found" after following install instructions
- Package works in dev environment but fails for users

**Phase to address:**
Phase 1 (Dependency Resolution) - Must resolve before PyPI. Investigate RoMa's actual runtime requirements (not just declared dependencies). Test if newer numpy works despite metadata. Contact RoMa maintainers or switch to alternative dense matching library if needed.

---

### Pitfall 6: GPU Dependencies Break CPU-Only Installations

**What goes wrong:**
Package depends on `torch` without specifying CPU/CUDA variant. Users on CPU-only machines get massive CUDA toolkit downloads (2+ GB) they cannot use. CI tests pass on GPU runners but fail on standard CPU runners. As of mid-2023, PyPI and Python packaging tools are completely unaware of GPUs and CUDA.

**Why it happens:**
PyTorch has separate package indices for CPU and CUDA variants, but there's no standard way to express "GPU optional" in PyPI metadata. Developers test on GPU machines and don't notice CPU installation issues.

**How to avoid:**
Document that users should install PyTorch separately first, following official PyTorch installation instructions for their platform. Make GPU support optional with runtime checks: `torch.cuda.is_available()` and graceful fallback. Or use extras: `pip install aquamvs[gpu]` vs `pip install aquamvs[cpu]`. Provide clear installation instructions for both paths in README.

**Warning signs:**
- CPU-only CI jobs take 10+ minutes to install dependencies
- Users report multi-gigabyte downloads for simple tasks
- Import fails with "CUDA not available" errors

**Phase to address:**
Phase 3 (Packaging Configuration) - Add extras_require with separate dependency sets. Update documentation with platform-specific installation paths. Test both installation methods in CI.

---

### Pitfall 7: Headless Rendering Not Configured for CI

**What goes wrong:**
Open3D visualization code runs fine locally but crashes in CI with "cannot open display" errors. Tests that save visualizations or generate meshes fail. Open3D headless rendering requires compile-time flag `-DENABLE_HEADLESS_RENDERING=ON` which isn't enabled in PyPI wheels.

**Why it happens:**
GUI libraries expect X11/Wayland display on Linux. CI runners are headless. Open3D's OffscreenRenderer needs special configuration (OSMesa or EGL with `EGL_PLATFORM=surfaceless`). Developers test locally with displays available.

**How to avoid:**
For Ubuntu 20.04+ with Mesa v20.2+, set environment variable `EGL_PLATFORM=surfaceless` for OffscreenRenderer. For older systems, use xvfb-run wrapper in CI. Separate rendering tests from core algorithm tests so headless issues don't block entire suite. Use pytest markers to skip rendering tests in headless environments.

**Warning signs:**
- CI logs show "cannot open display :0"
- Tests marked with `@pytest.mark.skipif` for "no display"
- Local tests pass but CI fails on same code

**Phase to address:**
Phase 4 (Testing Infrastructure) - Configure CI environment variables and test markers. Verify that OffscreenRenderer works in headless mode. Add documentation for developers running tests without display.

---

### Pitfall 8: Breaking API Changes Without Migration Path

**What goes wrong:**
Refactoring 995-line pipeline.py splits functions across modules. Existing user code with `from aquamvs.pipeline import run_reconstruction` breaks. No deprecation warnings, no compatibility shim. Users upgrade and their code crashes.

**Why it happens:**
Internal refactoring focuses on clean architecture but forgets backward compatibility. Semantic versioning requires major version bump for breaking changes, but developers forget or don't want to jump to 2.0 yet.

**How to avoid:**
Before refactoring public APIs: (1) add deprecation warnings to old locations, (2) provide compatibility imports that forward to new locations, (3) document migration in CHANGELOG with before/after examples, (4) wait at least one minor version before removing old APIs. Follow semantic versioning strictly: breaking changes require major version bump.

**Warning signs:**
- Functions moved between modules without import redirects
- No deprecation warnings in current codebase
- CHANGELOG says "refactored" but doesn't mention API changes

**Phase to address:**
Phase 2 (Refactoring) - Essential before touching public APIs. Create migration strategy document. Add deprecation warnings and compatibility layer first, refactor internals second, only remove deprecated code in major version bump.

---

### Pitfall 9: Configuration Sprawl Without Validation

**What goes wrong:**
9+ dataclasses for configuration grow organically without coherent structure. Invalid configurations silently produce wrong results (e.g., negative depth ranges, mismatched array shapes). Users spend hours debugging when config validation would catch errors at startup.

**Why it happens:**
Dataclasses make it easy to add fields without thinking about validation. Each module adds its own config class without central coordination. Type hints exist but aren't enforced at runtime.

**How to avoid:**
Centralize configuration using hierarchical structure with a root config object. Use Pydantic instead of plain dataclasses for runtime validation. Add validation methods that check cross-field constraints (e.g., `assert depth_max > depth_min`). Fail fast at initialization with clear error messages rather than producing invalid outputs.

**Warning signs:**
- 9+ separate dataclass config objects
- No validation beyond type hints
- Bug reports with invalid config values that should have been rejected

**Phase to address:**
Phase 2 (Refactoring) - Consolidate during config restructure. Migrate from dataclasses to Pydantic for validation. Add comprehensive validation tests that try to break the config with invalid values.

---

### Pitfall 10: Version Pinning Creates Dependency Hell

**What goes wrong:**
Package pins all dependencies with exact versions: `numpy==1.24.3, torch==2.0.1`. Users cannot install AquaMVS alongside other packages because version conflicts are unresolvable. TensorFlow used to put an upper cap on everything, which was a complete mess.

**Why it happens:**
Fear of breaking changes leads to over-pinning. Lock files (poetry.lock, requirements.txt) are confused with package dependencies. Applications pin versions, but libraries should specify ranges.

**How to avoid:**
Libraries should specify minimum versions with lower bounds: `numpy >= 1.24.0`. Avoid upper bounds unless there's a known incompatibility. By setting upper bounds, users and package dependency resolvers are much more likely to reach unresolvable dependencies over time. Test against multiple dependency versions in CI (oldest supported and latest). Use lock files for development reproducibility but not for distribution.

**Warning signs:**
- pyproject.toml shows `==` for all dependencies
- Upper bounds on all packages (`<2.0`)
- Users report "could not find a version that satisfies" errors

**Phase to address:**
Phase 3 (Packaging Configuration) - Audit all dependencies. Convert exact pins to minimum version requirements. Test compatibility range in CI matrix. Document known incompatibilities only.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| `git+https://` without commit pin | Fast prototyping | Unreproducible builds, PyPI rejection | Never for published packages |
| `--no-deps` workaround | Bypasses resolver | Users get broken installs | Never - document proper resolution |
| Exact version pins (`==`) | Reproducible now | Dependency hell later | Applications only, never libraries |
| Platform-specific hacks at module level | Works on dev machine | Breaks other platforms | Never - use conditional imports |
| Skipping wheel content verification | Faster releases | Users get tests/examples pollution | Never - automate the check |
| Local path dependencies | Easy dev iteration | Cannot distribute | Development only, never in production |
| Monolithic config classes | Quick to add fields | Impossible to maintain | MVP only, refactor before 1.0 |

## Integration Gotchas

Common mistakes when connecting to external dependencies.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| PyTorch | Assuming CUDA wheels on all platforms | Document separate installation, check `torch.cuda.is_available()` |
| Open3D | Using visualization in CI without headless config | Set `EGL_PLATFORM=surfaceless`, use xvfb-run, or skip rendering tests |
| NumPy 2.0 | Not testing against both numpy 1.x and 2.x | CI matrix with multiple numpy versions, use compatible API subset |
| Git dependencies | Expecting GitHub repos to be stable | Pin commits, mirror/vendor code, or wait for PyPI release |
| AquaCal | Tight coupling via editable install | Define minimal interface, vendor if small, or publish AquaCal first |

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Loading full model at import time | Slow imports (10+ seconds) | Lazy load models on first use | Any import-time operation |
| Uncompressed wheel with large data files | 500+ MB wheel, slow PyPI download | Exclude data from wheel, download on first use | Wheels > 100 MB |
| Synchronous I/O for large videos | Pipeline blocks on video read | Use async I/O or streaming | Videos > 1 GB |
| Dense GPU operations without batching | OOM on large reconstructions | Process in tiles/chunks with configurable batch size | Large image sets (100+ frames) |

## Dependency Management Mistakes

Domain-specific packaging and dependency issues.

| Mistake | Risk | Prevention |
|---------|------|------------|
| Not separating dev dependencies from install requires | Users get pytest, black, sphinx unnecessarily | Use `[tool.poetry.dev-dependencies]` or extras |
| Upper bounds on all packages | Unresolvable conflicts with other packages | Only lower bounds unless known incompatibility |
| Missing python_requires in metadata | Installs on Python 3.8 then crashes with 3.10+ syntax | Set `requires-python = ">=3.10"` in pyproject.toml |
| Conditional imports without optional dependencies | ImportError for users who don't need feature | Use extras: `pip install aquamvs[visualization]` |
| Git dependencies in install_requires | PyPI upload fails, cannot resolve in other projects | Resolve to PyPI package or document manual install |

## Documentation Pitfalls

Common documentation mistakes in scientific packages.

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| No installation examples for different platforms | Linux users install CUDA on CPU-only machines | Separate install docs for Windows/Linux, CPU/GPU |
| API reference without examples | Users read docstrings but don't understand usage | Add code examples to every public function |
| No migration guide for refactoring | Breaking changes surprise users | Document old → new API mapping with examples |
| Missing type hints in signatures | IDE autocomplete doesn't work | Add full type hints, generate docs from signatures |
| Configuration reference without defaults | Users don't know what values are reasonable | Show default config YAML with comments explaining each field |

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Packaging:** Wheel uploaded to PyPI - verify install on fresh machine works without local paths or git dependencies
- [ ] **Dependencies:** `pip install aquamvs` - verify no `--no-deps` or manual steps required
- [ ] **Platform testing:** CI tests pass - verify runs on both Windows and Linux, not just dev machine
- [ ] **Headless rendering:** Tests work in CI - verify Open3D/visualization tests run without display
- [ ] **GPU optional:** CPU-only install works - verify doesn't require CUDA when device='cpu'
- [ ] **API stability:** Public functions documented - verify no undocumented breaking changes from refactoring
- [ ] **Config validation:** Invalid configs fail loudly - verify nonsensical values rejected at initialization
- [ ] **Version ranges:** Compatible with ecosystem - verify no exact pins blocking other packages
- [ ] **Migration path:** Deprecated APIs warn - verify old imports still work with warnings before removal

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Git dependency without pin | MEDIUM | Fork repo, pin to last-known-good commit, or vendor code |
| Test directory in wheel | LOW | Fix package config, bump patch version, re-upload to PyPI |
| Breaking API changes shipped | HIGH | Restore old API with deprecation warnings in patch release, delay breaking change to major version |
| Platform-specific code breaks other platform | MEDIUM | Add platform checks, release patch, add platform to CI matrix |
| Dependency version conflict | HIGH | Relax constraints if possible, or bump major version with new requirements |
| Local path dependency shipped | HIGH | Cannot recover - must publish dependency first or vendor code |
| No headless rendering config | LOW | Document environment variables, add xvfb to CI, skip rendering tests |
| Configuration accepts invalid values | MEDIUM | Add validation in patch, update docs, add regression tests |

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Git dependencies without pins | Phase 1: Dependency Resolution | Check pyproject.toml has no unpinned git+ URLs |
| Local editable dependencies | Phase 1: Dependency Resolution | Verify pip install from wheel works on clean machine |
| Platform-specific hacks | Phase 2: Refactoring | Run tests on Windows and Linux in CI |
| Test directories in wheel | Phase 3: Packaging | `unzip -l dist/*.whl` contains no tests/ or docs/ |
| `--no-deps` workaround | Phase 1: Dependency Resolution | Installation works with standard pip install (no flags) |
| GPU dependencies break CPU | Phase 3: Packaging | Test CPU-only install in CI without CUDA |
| Headless rendering not configured | Phase 4: Testing | CI runs visualization tests successfully |
| Breaking API changes | Phase 2: Refactoring | Old imports work with deprecation warnings |
| Configuration sprawl | Phase 2: Refactoring | Single entry point with validation rejects invalid configs |
| Version pinning | Phase 3: Packaging | All dependencies use >= without upper bounds (unless justified) |

## Sources

**Packaging Best Practices:**
- [Knowledge Bits - Common Python Packaging Mistakes](https://jwodder.github.io/kbits/posts/pypkg-mistakes/)
- [The State of Python Packaging in 2026](https://learn.repoforge.io/posts/the-state-of-python-packaging-in-2026/)
- [Python Packaging User Guide - Scientific Packages](https://packaging.python.org/guides/installing-scientific-packages/)

**Dependency Management:**
- [IBM Data Science Best Practices - Dependency Management](https://ibm.github.io/data-science-best-practices/dependency_management.html)
- [The Absolute Minimum Everyone Must Know About Managing Python Dependencies in 2025](https://brojonat.com/posts/python-dependencies-2025/)
- [Python Dependency Hell: The Ultimate 2025 Fix Guide](https://junkangworld.com/blog/python-dependency-hell-the-ultimate-2025-fix-guide)

**Git and Platform Dependencies:**
- [Poetry - Dependency Specification](https://python-poetry.org/docs/dependency-specification/)
- [setuptools - Dependencies Management](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html)
- [PEP 508 - Dependency Specification](https://peps.python.org/pep-0508/)

**GPU and Headless Rendering:**
- [Packaging projects with GPU code](https://pypackaging-native.github.io/key-issues/gpus/)
- [Open3D Headless Rendering Documentation](https://www.open3d.org/docs/release/tutorial/visualization/headless_rendering.html)
- [Open3D CPU Rendering](https://www.open3d.org/docs/latest/tutorial/visualization/cpu_rendering.html)

**Version Pinning:**
- [Should You Use Upper Bound Version Constraints?](https://iscinumpy.dev/post/bound-version-constraints/)
- [Confessions of an asymmetric hypocrite: on Python dependencies](https://www.nijho.lt/post/dependencies/)

**Configuration and Refactoring:**
- [Best practices for configurations in Python pipelines](https://belux.micropole.com/blog/python/blog-best-practices-for-configurations-in-python-based-pipelines/)
- [Python Dataclasses: The Complete Guide for 2026](https://devtoolbox.dedyn.io/blog/python-dataclasses-guide)
- [Python Refactoring: Techniques, Tools, and Best Practices](https://www.codesee.io/learning-center/python-refactoring)

**Editable Dependencies:**
- [PEP 660 - Editable Installs](https://peps.python.org/pep-0660/)
- [Dependencies in Python Packaging](https://kfchou.github.io/package-depdencies/)

**Semantic Versioning:**
- [Creating New Versions of Your Python Package](https://www.pyopensci.org/python-package-guide/package-structure-code/python-package-versions.html)
- [Python Package Versioning: SemVer, CalVer, and Best Practices](https://inventivehq.com/blog/python-package-versioning-guide)

---
*Pitfalls research for: Scientific Python Package Productionization (AquaMVS)*
*Researched: 2026-02-14*
