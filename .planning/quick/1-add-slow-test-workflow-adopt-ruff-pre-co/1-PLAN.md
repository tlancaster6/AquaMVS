---
phase: quick
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - .github/workflows/slow-tests.yml
  - .github/workflows/test.yml
  - .github/workflows/lint.yml
  - .github/workflows/docs.yml
  - .pre-commit-config.yaml
  - pyproject.toml
  - docs/conf.py
  - docs/index.rst
  - docs/Makefile
  - docs/make.bat
autonomous: true
must_haves:
  truths:
    - "Slow tests can be triggered manually via workflow_dispatch"
    - "Ruff enforces linting and formatting in CI and locally via pre-commit"
    - "Test workflow uploads coverage reports"
    - "Docs workflow builds Sphinx documentation on push to main"
  artifacts:
    - path: ".github/workflows/slow-tests.yml"
      provides: "Manual slow test workflow"
      contains: "workflow_dispatch"
    - path: ".github/workflows/lint.yml"
      provides: "Ruff lint CI enforcement"
      contains: "ruff"
    - path: ".pre-commit-config.yaml"
      provides: "Pre-commit hooks for Ruff"
      contains: "ruff"
    - path: "pyproject.toml"
      provides: "Ruff config, Sphinx dev deps, Black removed"
      contains: "tool.ruff"
    - path: ".github/workflows/docs.yml"
      provides: "Sphinx docs build workflow"
      contains: "sphinx-build"
    - path: "docs/conf.py"
      provides: "Sphinx configuration"
      contains: "aquamvs"
  key_links:
    - from: ".pre-commit-config.yaml"
      to: "pyproject.toml"
      via: "ruff reads config from pyproject.toml"
      pattern: "tool\\.ruff"
    - from: ".github/workflows/lint.yml"
      to: "pyproject.toml"
      via: "ruff in CI uses same config"
      pattern: "ruff check"
---

<objective>
Add slow-test workflow, adopt Ruff+pre-commit (replacing Black), add coverage upload to test workflow, and add docs workflow with Sphinx scaffolding.

Purpose: Complete CI/CD tooling with linting, coverage, slow tests, and documentation infrastructure.
Output: 4 workflow files, pre-commit config, Ruff config in pyproject.toml, Sphinx docs/ scaffold.
</objective>

<execution_context>
@C:/Users/tucke/.claude/get-shit-done/workflows/execute-plan.md
@C:/Users/tucke/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.github/workflows/test.yml
@.github/workflows/publish.yml
@pyproject.toml
</context>

<tasks>

<task type="auto">
  <name>Task 1: Adopt Ruff, add pre-commit, update pyproject.toml</name>
  <files>pyproject.toml, .pre-commit-config.yaml</files>
  <action>
1. In `pyproject.toml`:
   - Replace `"black"` in `[project.optional-dependencies] dev` with `"ruff"`, `"pre-commit"`.
   - Add `"sphinx"`, `"sphinx-rtd-theme"` to the dev dependencies.
   - Remove any `[tool.black]` section if present.
   - Add `[tool.ruff]` configuration:
     ```toml
     [tool.ruff]
     target-version = "py310"
     line-length = 88
     src = ["src", "tests"]

     [tool.ruff.lint]
     select = ["E", "F", "W", "I", "UP", "B", "SIM"]
     ignore = ["E501"]

     [tool.ruff.lint.isort]
     known-first-party = ["aquamvs"]
     ```
   - The `select` rules: E/F/W (pyflakes+pycodestyle), I (isort), UP (pyupgrade for 3.10+), B (bugbear), SIM (simplify). E501 ignored because line-length is handled by formatter.

2. Create `.pre-commit-config.yaml`:
   ```yaml
   repos:
     - repo: https://github.com/astral-sh/ruff-pre-commit
       rev: v0.9.6
       hooks:
         - id: ruff
           args: [--fix]
         - id: ruff-format
   ```

3. Run `ruff format src/ tests/` and `ruff check --fix src/ tests/` to migrate existing code from Black formatting to Ruff. Fix any remaining lint errors manually. The formatting should be nearly identical since Ruff's formatter is Black-compatible.
  </action>
  <verify>
Run `ruff check src/ tests/` with zero errors. Run `ruff format --check src/ tests/` with zero reformats needed.
  </verify>
  <done>pyproject.toml has Ruff config, Black removed from deps, pre-commit config exists, all code passes Ruff lint and format checks.</done>
</task>

<task type="auto">
  <name>Task 2: Add slow-tests, lint, and docs workflows; update test workflow with coverage upload</name>
  <files>.github/workflows/slow-tests.yml, .github/workflows/lint.yml, .github/workflows/docs.yml, .github/workflows/test.yml</files>
  <action>
1. Create `.github/workflows/slow-tests.yml`:
   - `name: Slow Tests`
   - Trigger: `workflow_dispatch` only (manual trigger with optional inputs).
   - Add input `python-version` with default `"3.12"` and input `os` with default `"ubuntu-latest"`.
   - Single job `slow-test` running on the selected OS with selected Python.
   - Steps: checkout, setup-python (with pip cache), install PyTorch CPU, install git-based prerequisites (`pip install -r requirements-prereqs.txt`), install package (`pip install -e ".[dev]"`), run `pytest tests/ -m slow --timeout=600 -v`.
   - Same env: `PYTHONUNBUFFERED: 1`.

2. Create `.github/workflows/lint.yml`:
   - `name: Lint`
   - Trigger: push to main, pull_request to main.
   - Single job on `ubuntu-latest`, Python 3.12.
   - Steps: checkout, setup-python, `pip install ruff`, `ruff check src/ tests/`, `ruff format --check src/ tests/`.
   - Lean workflow: no project install needed, just ruff.

3. Update `.github/workflows/test.yml`:
   - After the existing `Run tests` step, add a step to upload coverage:
     ```yaml
     - name: Upload coverage
       if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.12'
       uses: actions/upload-artifact@v4
       with:
         name: coverage-report
         path: coverage.xml
     ```
   - The existing pytest command already produces `--cov-report=xml` so `coverage.xml` is already generated. No change to the pytest command needed.

4. Create `.github/workflows/docs.yml`:
   - `name: Docs`
   - Trigger: push to main (paths: `docs/**`, `src/**/*.py`), pull_request to main.
   - Single job on `ubuntu-latest`, Python 3.12.
   - Steps: checkout, setup-python (with pip cache), install PyTorch CPU, install git-based prerequisites, install package with dev deps, run `sphinx-build -W -b html docs/ docs/_build/html` (the `-W` flag treats warnings as errors).
   - Upload built docs as artifact:
     ```yaml
     - name: Upload docs
       uses: actions/upload-artifact@v4
       with:
         name: docs-html
         path: docs/_build/html/
     ```
  </action>
  <verify>
Validate YAML syntax: `python -c "import yaml; yaml.safe_load(open('.github/workflows/slow-tests.yml')); yaml.safe_load(open('.github/workflows/lint.yml')); yaml.safe_load(open('.github/workflows/docs.yml')); yaml.safe_load(open('.github/workflows/test.yml')); print('All valid')"`.
  </verify>
  <done>Four workflow files exist with correct triggers: slow-tests (workflow_dispatch), lint (push/PR with ruff), docs (push/PR building sphinx), test (existing + coverage upload on ubuntu/3.12 only).</done>
</task>

<task type="auto">
  <name>Task 3: Create Sphinx documentation scaffolding</name>
  <files>docs/conf.py, docs/index.rst, docs/Makefile, docs/make.bat</files>
  <action>
1. Create `docs/conf.py`:
   ```python
   project = "AquaMVS"
   copyright = "2024, Tucker Lancaster"
   author = "Tucker Lancaster"
   extensions = [
       "sphinx.ext.autodoc",
       "sphinx.ext.napoleon",
       "sphinx.ext.viewcode",
       "sphinx.ext.intersphinx",
   ]
   templates_path = ["_templates"]
   exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
   html_theme = "sphinx_rtd_theme"
   intersphinx_mapping = {
       "python": ("https://docs.python.org/3", None),
       "torch": ("https://pytorch.org/docs/stable", None),
       "numpy": ("https://numpy.org/doc/stable", None),
   }
   autodoc_typehints = "description"
   napoleon_google_docstrings = True
   ```

2. Create `docs/index.rst`:
   ```rst
   AquaMVS
   =======

   Multi-view stereo reconstruction of underwater surfaces with refractive modeling.

   .. toctree::
      :maxdepth: 2
      :caption: Contents:

   Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
   ```

3. Create `docs/Makefile` (standard Sphinx Makefile):
   ```makefile
   SPHINXOPTS    ?=
   SPHINXBUILD   ?= sphinx-build
   SOURCEDIR     = .
   BUILDDIR      = _build

   help:
   	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

   .PHONY: help Makefile

   %: Makefile
   	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
   ```

4. Create `docs/make.bat` (standard Sphinx make.bat for Windows):
   ```bat
   @ECHO OFF
   pushd %~dp0
   if "%SPHINXBUILD%" == "" set SPHINXBUILD=sphinx-build
   set SOURCEDIR=.
   set BUILDDIR=_build
   %SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
   popd
   ```

5. Add `docs/_build/` to `.gitignore` if not already present.
  </action>
  <verify>
Run `sphinx-build -W -b html docs/ docs/_build/html` and confirm it builds successfully with zero warnings. Then `rm -rf docs/_build` to clean up.
  </verify>
  <done>docs/ directory contains conf.py (with autodoc, napoleon, RTD theme, intersphinx), index.rst, Makefile, make.bat. Sphinx builds cleanly.</done>
</task>

</tasks>

<verification>
1. `ruff check src/ tests/` passes with no errors
2. `ruff format --check src/ tests/` reports no changes needed
3. All 4 workflow YAML files parse without errors
4. `sphinx-build -W -b html docs/ docs/_build/html` builds successfully
5. `pyproject.toml` contains `[tool.ruff]` and no Black references in dev deps
6. `.pre-commit-config.yaml` exists with ruff hooks
</verification>

<success_criteria>
- Ruff replaces Black for formatting and adds linting, configured in pyproject.toml
- Pre-commit config wires Ruff for local development
- Slow-tests workflow exists with workflow_dispatch trigger
- Lint workflow enforces Ruff in CI on push/PR
- Test workflow uploads coverage artifact for ubuntu/3.12 matrix entry
- Docs workflow builds Sphinx on push/PR
- Sphinx scaffolding in docs/ builds without warnings
</success_criteria>

<output>
After completion, create `.planning/quick/1-add-slow-test-workflow-adopt-ruff-pre-co/1-SUMMARY.md`
</output>
