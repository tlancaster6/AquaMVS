---
phase: 01-dependency-resolution-and-packaging-foundations
plan: 02
subsystem: ci-cd
tags: [github-actions, ci, testing, publishing, trusted-publishing, pypi]

# Dependency graph
requires:
  - phase: 01-01
    provides: PyPI-compatible dependency specification with prerequisite workflow
provides:
  - GitHub Actions test workflow with platform matrix (2 OS x 3 Python versions)
  - GitHub Actions publish workflow with Trusted Publishing (OIDC)
  - Automated TestPyPI validation before PyPI upload
affects: [deployment, testing, release-automation]

# Tech tracking
tech-stack:
  added: [github-actions, pypa-gh-action-pypi-publish]
  patterns: [matrix-testing, trusted-publishing, staged-deployment]

key-files:
  created: [.github/workflows/test.yml, .github/workflows/publish.yml]
  modified: []

key-decisions:
  - "Matrix testing across Ubuntu and Windows with Python 3.10, 3.11, 3.12 (6 combinations)"
  - "PyTorch CPU-only in CI for faster builds and smaller resource footprint"
  - "Git prerequisites installed from requirements-prereqs.txt before package install"
  - "Trusted Publishing (OIDC) eliminates API token management for PyPI uploads"
  - "Three-stage publish pipeline: build -> TestPyPI -> PyPI with manual approval gate"

patterns-established:
  - "Prerequisite installation in CI: PyTorch CPU -> requirements-prereqs.txt -> pip install -e .[dev]"
  - "fail-fast: false in matrix to run all platform/Python combinations even if one fails"
  - "TestPyPI validation before production PyPI upload"

# Metrics
duration: 5min
completed: 2026-02-14
---

# Phase 01 Plan 02: CI/CD Setup Summary

**GitHub Actions workflows for automated testing across platforms and secure PyPI publishing via Trusted Publishing (OIDC)**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-14T16:22:00Z
- **Completed:** 2026-02-14T16:27:00Z
- **Tasks:** 2
- **Files created:** 2

## Accomplishments
- Created test workflow with 6-combination matrix (Ubuntu + Windows × Python 3.10-3.12)
- Configured prerequisite installation workflow (PyTorch CPU -> git deps -> package)
- Created publish workflow with three-stage pipeline (build -> TestPyPI -> PyPI)
- Implemented Trusted Publishing (OIDC) for secure, token-free PyPI uploads
- Added TestPyPI validation step before production release

## Task Commits

Each task was committed atomically:

1. **Task 1: Create test workflow with platform matrix** - `f606fea` (feat)
2. **Task 2: Create publish workflow with Trusted Publishing** - `6a56825` (feat)

## Files Created/Modified
- `.github/workflows/test.yml` - Matrix test workflow (2 OS × 3 Python = 6 combinations)
  - Installs PyTorch CPU-only for faster CI
  - Installs git prerequisites from requirements-prereqs.txt
  - Runs pytest with coverage, excludes slow tests
  - Uses pip cache for faster dependency installation

- `.github/workflows/publish.yml` - Three-stage publish workflow
  - Build job: creates wheel and sdist
  - TestPyPI job: validates package on test instance
  - PyPI job: publishes to production (requires manual approval)
  - All stages use Trusted Publishing (OIDC) with id-token: write

## Decisions Made

**Task 1 (Test Workflow):**
- Matrix: Ubuntu + Windows × Python 3.10, 3.11, 3.12 (6 combinations total)
- PyTorch CPU-only installation from pytorch.org CPU index for faster CI builds
- Prerequisites from requirements-prereqs.txt installed BEFORE package (prereq-docs strategy)
- fail-fast: false to ensure all matrix combinations run even if one fails
- Coverage reporting with XML output (for potential future integrations)
- PYTHONUNBUFFERED=1 for real-time test output visibility

**Task 2 (Publish Workflow):**
- Trigger on version tags (v*) for semantic versioning compatibility
- Three-stage pipeline ensures safety: build → TestPyPI → PyPI
- Trusted Publishing (OIDC) eliminates need for API tokens
- TestPyPI environment for validation uploads
- PyPI environment with manual approval gate (configured in GitHub settings)
- Uses pypa/gh-action-pypi-publish@release/v1 for both TestPyPI and PyPI

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - both workflows created and validated successfully. YAML syntax confirmed valid.

## User Setup Required

**GitHub repository configuration needed:**

1. **Trusted Publishing setup on PyPI:**
   - Visit https://pypi.org/manage/account/publishing/
   - Add publisher: github.com/tlancaster6/AquaMVS, workflow: publish.yml, environment: pypi

2. **Trusted Publishing setup on TestPyPI:**
   - Visit https://test.pypi.org/manage/account/publishing/
   - Add publisher: github.com/tlancaster6/AquaMVS, workflow: publish.yml, environment: testpypi

3. **GitHub environments:**
   - Create `testpypi` environment (no reviewers needed)
   - Create `pypi` environment with required reviewers for manual approval gate

4. **First CI run validation:**
   - Push to main or create PR to trigger test workflow
   - Verify all 6 matrix combinations (2 OS × 3 Python) pass
   - If platform-specific failures occur, create follow-up tasks to resolve

## CI Workflow Verification

**Local validation completed:**
- ✓ YAML syntax valid for both workflows
- ✓ test.yml matrix correctly configured (2 OS × 3 Python = 6 jobs)
- ✓ publish.yml uses Trusted Publishing (id-token: write)
- ✓ Three-stage pipeline correctly sequenced (build -> testpypi -> pypi)
- ✓ Prerequisite installation reflects prereq-docs strategy

**Remote validation pending:**
- CI test workflow execution will be verified on first push/PR to main
- If tests fail in CI due to platform-specific issues, create follow-up plan
- Publish workflow will be verified on first version tag release

## Next Phase Readiness

**Ready for Phase 02 (Integration Testing):**
- Automated testing infrastructure in place
- Platform coverage includes Windows (primary development environment)
- Coverage reporting configured for tracking test completeness
- CI will validate all future changes before merge

**Ready for future releases:**
- Publish workflow ready for v0.1.0 release after Phase 01 completion
- TestPyPI validation ensures package is installable before PyPI upload
- No manual token management required (Trusted Publishing)

**No blockers.**

## Self-Check: PASSED

All claims verified:
- ✓ .github/workflows/test.yml exists
- ✓ .github/workflows/publish.yml exists
- ✓ Task 1 commit f606fea exists
- ✓ Task 2 commit 6a56825 exists
- ✓ test.yml matrix has 2 OS × 3 Python versions
- ✓ publish.yml uses id-token: write for both TestPyPI and PyPI
- ✓ publish.yml has three-stage pipeline (build -> testpypi -> pypi)
- ✓ test.yml installs prerequisites from requirements-prereqs.txt

---
*Phase: 01-dependency-resolution-and-packaging-foundations*
*Completed: 2026-02-14*
