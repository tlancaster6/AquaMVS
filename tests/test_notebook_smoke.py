"""Smoke tests for Jupyter notebooks.

Validates that all code cells in documentation notebooks:
1. Parse as valid Python (syntax check)
2. Have resolvable import statements (import check)

Does NOT execute notebooks (no dataset or GPU required).
"""

import ast
import importlib
import json
from pathlib import Path

import pytest

NOTEBOOK_PATHS = sorted(
    p for p in Path("docs").rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts
)

# Modules that require hardware/dataset and should be skipped in import check
SKIP_IMPORT_MODULES = {"google.colab"}


def _extract_imports(source: str) -> list[str]:
    """Extract top-level module names from import statements.

    Args:
        source: Python source code from a notebook cell.

    Returns:
        List of top-level module names referenced by import statements.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module.split(".")[0])
    return modules


@pytest.mark.parametrize("nb_path", NOTEBOOK_PATHS, ids=lambda p: p.stem)
def test_notebook_syntax(nb_path):
    """All notebook code cells must parse as valid Python."""
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if not source.strip():
            continue
        # Skip cells that are shell commands (start with !)
        if source.strip().startswith("!"):
            continue
        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"{nb_path.name} cell {i}: {e}")


@pytest.mark.parametrize("nb_path", NOTEBOOK_PATHS, ids=lambda p: p.stem)
def test_notebook_imports(nb_path):
    """All import statements in notebook code cells must resolve."""
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if not source.strip() or source.strip().startswith("!"):
            continue
        for mod in _extract_imports(source):
            if mod in SKIP_IMPORT_MODULES:
                continue
            try:
                importlib.import_module(mod)
            except ImportError:
                pytest.fail(f"{nb_path.name} cell {i}: cannot import '{mod}'")
