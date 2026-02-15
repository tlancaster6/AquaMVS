# Quick Task 2: Fix pre-commit lint issues

## Tasks

### Task 1: Fix ruff lint errors
- Fix invalid `# noqa` directive in `src/aquamvs/benchmark/visualization.py:13` (use `# noqa: F401`)
- Fix B904 in `src/aquamvs/config.py:379` — add `from None` to `raise ValueError`

### Task 2: Fix whitespace/EOF issues
- Convert CRLF to LF in `.gitignore`, `README.md`, `.planning/ROADMAP.md`
- Ensure final newline in all three files
