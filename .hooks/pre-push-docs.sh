#!/usr/bin/env bash
sphinx-build -W --keep-going -b html docs docs/_build/html || {
    echo ""
    echo "Push failed: documentation build errors."
    echo 'Run "sphinx-build -W --keep-going -b html docs docs/_build/html" to reproduce.'
    exit 1
}
