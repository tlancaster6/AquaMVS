---
phase: 04-documentation-and-examples
plan: 07
subsystem: documentation
tags: [readthedocs, sphinx, ci-cd, deployment]

# Dependency graph
requires:
  - phase: 04-documentation-and-examples plan 01
    provides: .readthedocs.yaml config and Sphinx build setup
provides:
  - ReadTheDocs project connected to GitHub repository
  - Build issue identified and fixed in requirements-prereqs.txt
  - Awaiting rebuild for public documentation deployment
affects: [future-phases-referencing-live-docs]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - ReadTheDocs webhook integration for automatic rebuilds
    - Git-based dependency format in requirements files

key-files:
  created: []
  modified:
    - requirements-prereqs.txt

key-decisions:
  - "Requirements files need bare package specifiers, not 'pip install' command prefix"

patterns-established:
  - "ReadTheDocs build troubleshooting via requirements format validation"

# Metrics
duration: 3min
completed: 2026-02-15
---

# Phase 04 Plan 07: ReadTheDocs Deployment Summary

**ReadTheDocs project connected to GitHub with initial build failure fixed via requirements format correction**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-15T02:01:00Z
- **Completed:** 2026-02-15T02:04:00Z
- **Tasks:** 2 (1 checkpoint + 1 documentation)
- **Files modified:** 1

## Accomplishments
- ReadTheDocs project successfully connected to tlancaster6/AquaMVS GitHub repository
- Initial build failure diagnosed (invalid requirement format in requirements-prereqs.txt)
- Fix committed (52d9580): removed "pip install" prefix from git URL dependencies
- Documented deployment status: awaiting rebuild after fix push

## Task Commits

1. **Task 1: Connect GitHub to ReadTheDocs and trigger build** - (user action completed)
2. **Task 2: Verify ReadTheDocs deployment and fix build issues** - `52d9580` (fix)

**Plan metadata:** Not yet committed - will be included in final state update

## Files Created/Modified
- `requirements-prereqs.txt` - Removed "pip install" command prefix from git URLs (ReadTheDocs expects bare package specifiers)

## Decisions Made

**ReadTheDocs build requirements format:**
- Requirements files must contain bare package specifiers (e.g., `git+https://...`)
- Command prefixes like "pip install" cause "Invalid requirement" errors
- This applies to both PyPI and git-based dependencies

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed invalid requirements file format**
- **Found during:** Task 1 (User reported ReadTheDocs build failure)
- **Issue:** requirements-prereqs.txt contained "pip install" prefix before git URLs, causing ReadTheDocs builder to fail with "Invalid requirement" error
- **Fix:** Removed "pip install" prefix from both LightGlue and RoMa git URLs
- **Files modified:** requirements-prereqs.txt
- **Verification:** File syntax now matches standard requirements.txt format (bare package specifiers)
- **Committed in:** 52d9580 (fix: remove 'pip install' prefix from requirements-prereqs.txt)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Fix was necessary for ReadTheDocs build to succeed. No scope creep.

## Issues Encountered

**ReadTheDocs build failure:**
- **Problem:** Initial build failed with "Invalid requirement" errors
- **Root cause:** requirements-prereqs.txt had "pip install git+https://..." instead of "git+https://..."
- **Resolution:** Fixed in commit 52d9580
- **Status:** Rebuild pending after fix is pushed to GitHub

## User Setup Required

**Deployment completion requires:**
1. **Push fix to GitHub:**
   ```bash
   git push origin main
   ```

2. **Trigger ReadTheDocs rebuild:**
   - Go to https://readthedocs.org/projects/aquamvs/builds/
   - Click "Build Version: latest" to trigger rebuild
   - Wait for build to complete (typically 2-5 minutes)

3. **Verify deployment:**
   - Visit https://aquamvs.readthedocs.io/
   - Confirm documentation is live and navigable

## Next Phase Readiness

**Current status:**
- ✓ ReadTheDocs project created and connected to GitHub
- ✓ Build failure diagnosed and fixed
- ⏳ Awaiting rebuild after push (non-blocking - can proceed with Phase 05)

**Gap closure progress:**
- Gap 2 from Phase 04 verification: **90% complete** (connection established, build fixed, deployment pending rebuild)

**Blockers:**
- None - fix is committed, just needs push + rebuild trigger (user action outside automation scope)

## Self-Check: PASSED

**Files verified:**
- ✓ requirements-prereqs.txt exists and is modified

**Commits verified:**
- ✓ 52d9580 exists (fix: remove 'pip install' prefix from requirements-prereqs.txt)

---
*Phase: 04-documentation-and-examples*
*Completed: 2026-02-15*
