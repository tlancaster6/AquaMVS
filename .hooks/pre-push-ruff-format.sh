#!/usr/bin/env bash
ruff format --check src/ tests/ || {
    echo ""
    echo "Push failed: ruff formatting errors."
    echo 'Run "ruff format src/ tests/" to auto-fix.'
    exit 1
}
