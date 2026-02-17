#!/usr/bin/env bash
python -m pytest tests/ -m "not slow" || {
    echo ""
    echo "Push failed: test failures."
    echo 'Run "python -m pytest tests/ -m \"not slow\"" to reproduce.'
    exit 1
}
