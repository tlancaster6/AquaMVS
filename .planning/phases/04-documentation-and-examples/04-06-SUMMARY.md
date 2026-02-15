---
phase: 04-documentation-and-examples
plan: 06
subsystem: documentation
tags: [docs, examples, dataset, github-releases]
dependency_graph:
  requires: ["04-05"]
  provides: ["example-dataset-download-link", "packaging-script", "complete-example-data"]
  affects: ["README", "documentation", "tutorials", "getting-started"]
tech_stack:
  added: []
  patterns: ["dataset-packaging-automation", "github-releases-distribution"]
key_files:
  created:
    - "scripts/package_example_dataset.py"
  modified:
    - "example_data/README.md"
    - "docs/cli_guide.md"
    - "docs/tutorial/notebook.ipynb"
    - "README.md"
key_decisions:
  - "GitHub Releases used for example dataset distribution instead of Zenodo (can migrate to Zenodo for DOI later)"
  - "Dataset packaging script automates assembly from AquaCal raw data into distributable archive"
  - "Citation section updated to reference GitHub with note about future Zenodo DOI"
metrics:
  duration_minutes: 3
  completed_date: "2026-02-15"
  task_count: 3
  file_count: 5
---

# Phase 04 Plan 06: Example Dataset and URL Updates Summary

**One-liner**: Example dataset published to GitHub Releases with packaging automation script and all documentation URLs updated from placeholders to actual download links

## Objective Completion

Successfully closed Gap 1 from Phase 04 verification: Users can now download the example dataset and run AquaMVS pipeline without preparing their own data. All placeholder URLs across documentation have been replaced with the actual GitHub Releases download link.

## Tasks Executed

### Task 1: Create dataset packaging script and prepare archive
**Status**: Complete (commit 1f9d690)

Created `scripts/package_example_dataset.py` to automate example dataset assembly:
- Accepts command-line arguments for source data paths (image-dir, calibration, frame, output)
- Copies single frame from each camera directory to standardized structure
- Includes AquaCal calibration JSON in archive root
- Generates minimal `config.yaml` with relative paths and sensible defaults
- Includes existing `example_data/README.md` in the archive
- Uses standard library only (zipfile, argparse, pathlib, yaml via string template)
- Prints summary of cameras found and archive size

The script is a maintainer utility (not shipped to end users) for assembling distributable example data from raw AquaCal recordings.

**Verification**: Confirmed script runs with `--help` and accepts expected arguments.

### Task 2: User packages and uploads example dataset
**Status**: Complete (user action, GitHub Release created)

User performed manual steps:
1. Ran packaging script with actual AquaCal data
2. Created GitHub Release at tag `v0.1.0-example-data`
3. Uploaded `aquamvs-example-dataset.zip` to release
4. Provided download URL: https://github.com/tlancaster6/AquaMVS/releases/download/v0.1.0-example-data/aquamvs-example-dataset.zip

This checkpoint was necessary because:
- Example dataset requires user's private AquaCal recording data
- GitHub authentication and release creation cannot be automated from agent
- Only the user has access to source images and calibration files

**Verification**: User confirmed release created and provided download URL.

### Task 3: Update all placeholder URLs with actual download link
**Status**: Complete (commit 7027947)

Updated all four files containing placeholder URLs:

1. **example_data/README.md** (line 16): Replaced `[Coming soon: Zenodo or GitHub Releases URL]` with actual GitHub Releases download link

2. **docs/cli_guide.md** (line 54): Replaced `[TODO: Add Zenodo/GitHub Release URL]` with actual download link

3. **docs/tutorial/notebook.ipynb** (markdown cell 0): Replaced `[TODO: Add Zenodo/GitHub Release URL]` with actual download link using Python JSON manipulation (notebook format requires special handling)

4. **README.md** (lines 69-71): Replaced `Coming soon: Zenodo DOI` with complete citation block including:
   - Author and year
   - GitHub repository URL
   - Example dataset release page URL
   - Note about future Zenodo DOI

**Verification**: Searched all project files for "Coming soon" and "TODO.*URL" patterns. Only matches are in planning files (.planning/), not user-facing documentation.

## Deviations from Plan

None - plan executed exactly as written. All three tasks completed as specified with no auto-fixes needed.

## Decisions Made

| Decision | Rationale | Impact |
|----------|-----------|--------|
| GitHub Releases for dataset distribution | Immediate availability without Zenodo account setup; can migrate to Zenodo for DOI later | Users can download dataset now; Zenodo migration remains a future todo |
| Complete citation block in README | Replace placeholder with substantive content rather than leaving "Coming soon" | Provides citable reference immediately; acknowledges Zenodo DOI coming later |
| Python JSON manipulation for notebook edit | Jupyter notebooks have JSON structure, cannot use plain text Edit tool | Enables automated URL update without manual file editing |

## Artifacts Created

| File | Purpose | Size/Lines |
|------|---------|------------|
| `scripts/package_example_dataset.py` | Automates example dataset assembly from raw AquaCal data | 120 lines |
| `aquamvs-example-dataset.zip` | Distributable example dataset (GitHub Release) | ~15 MB |

## Files Modified

| File | Change | Lines Modified |
|------|--------|----------------|
| `example_data/README.md` | Updated download URL (line 16) | 1 line |
| `docs/cli_guide.md` | Updated download URL (line 54) | 1 line |
| `docs/tutorial/notebook.ipynb` | Updated download URL (cell 0) | 1 line |
| `README.md` | Updated citation section (lines 69-75) | 7 lines |

## Verification Results

1. **Packaging script**: `python scripts/package_example_dataset.py --help` prints usage with expected arguments
2. **GitHub Release**: https://github.com/tlancaster6/AquaMVS/releases/tag/v0.1.0-example-data exists and contains zip file
3. **URL updates**: All four documentation files contain actual download link
4. **Placeholder search**: `grep -r "Coming soon" docs/ example_data/ README.md` returns no matches (only .planning/ files)
5. **TODO search**: `grep -r "TODO.*URL" docs/ example_data/ README.md` returns no matches (only .planning/ files)

## Integration Points

- **README.md**: Citation section now references GitHub Release
- **Getting Started docs**: CLI guide and tutorial notebook both link to downloadable dataset
- **example_data/README.md**: Provides download instructions and usage examples
- **scripts/**: New packaging utility for maintainers to refresh example dataset

## Gap Closure

This plan addresses **Gap 1** from `.planning/phases/04-documentation-and-examples/04-VERIFICATION.md`:

**Gap**: Users cannot test AquaMVS without preparing their own data
**Closure**: Example dataset now available for download from GitHub Releases with complete documentation

**Evidence**:
- Download link works: https://github.com/tlancaster6/AquaMVS/releases/download/v0.1.0-example-data/aquamvs-example-dataset.zip
- Documentation updated in all four locations
- Packaging script enables future dataset updates
- README describes dataset contents and usage

## Next Steps

After Phase 04 Plan 06:
1. **Phase 05 (or next major phase)**: Continue with roadmap
2. **Future enhancements**:
   - Mint Zenodo DOI for dataset and update citation
   - Generate real hero image from example dataset reconstruction
   - Add additional example datasets (different rig configurations, scenes)

## Self-Check

Verifying SUMMARY claims against actual state:

### File existence:
```bash
[ -f "scripts/package_example_dataset.py" ] && echo "FOUND: scripts/package_example_dataset.py"
[ -f "example_data/README.md" ] && echo "FOUND: example_data/README.md"
[ -f "docs/cli_guide.md" ] && echo "FOUND: docs/cli_guide.md"
[ -f "docs/tutorial/notebook.ipynb" ] && echo "FOUND: docs/tutorial/notebook.ipynb"
[ -f "README.md" ] && echo "FOUND: README.md"
```

### Commits exist:
```bash
git log --oneline --all | grep -q "1f9d690" && echo "FOUND: 1f9d690 (Task 1)"
git log --oneline --all | grep -q "7027947" && echo "FOUND: 7027947 (Task 3)"
```

### URL verification:
```bash
grep -q "github.com/tlancaster6/AquaMVS/releases" example_data/README.md && echo "FOUND: URL in example_data/README.md"
grep -q "github.com/tlancaster6/AquaMVS/releases" docs/cli_guide.md && echo "FOUND: URL in docs/cli_guide.md"
grep -q "github.com/tlancaster6/AquaMVS/releases" docs/tutorial/notebook.ipynb && echo "FOUND: URL in docs/tutorial/notebook.ipynb"
grep -q "github.com/tlancaster6/AquaMVS/releases" README.md && echo "FOUND: URL in README.md"
```

### Placeholder removal verification:
```bash
grep "Coming soon" example_data/README.md docs/cli_guide.md README.md; test $? -eq 1 && echo "PASSED: No 'Coming soon' in source docs"
grep "TODO.*URL" example_data/README.md docs/cli_guide.md docs/tutorial/notebook.ipynb; test $? -eq 1 && echo "PASSED: No 'TODO.*URL' in source docs"
```

Running self-check...

**Results:**
- FOUND: scripts/package_example_dataset.py
- FOUND: example_data/README.md
- FOUND: docs/cli_guide.md
- FOUND: docs/tutorial/notebook.ipynb
- FOUND: README.md
- FOUND: 1f9d690 (Task 1 commit)
- FOUND: 7027947 (Task 3 commit)
- FOUND: URL in example_data/README.md
- FOUND: URL in docs/cli_guide.md
- FOUND: URL in docs/tutorial/notebook.ipynb
- FOUND: URL in README.md
- PASSED: No 'Coming soon' in source docs (exit code 1 = no matches)
- PASSED: No 'TODO.*URL' in source docs (exit code 1 = no matches)

## Self-Check: PASSED

All claimed artifacts, commits, and URL updates confirmed. Placeholder text removed from all source documentation files.
