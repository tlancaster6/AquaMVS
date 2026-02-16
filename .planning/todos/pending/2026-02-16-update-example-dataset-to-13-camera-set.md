---
created: 2026-02-16T15:21:46.303Z
title: Update example dataset to 13-camera set
area: docs
files:
  - docs/examples/example-dataset/README.md
  - scripts/package_example_dataset.py
---

## Problem

The current example dataset is missing one camera due to a recording mishap, shipping with only 12 of the expected 13 cameras. Since AquaMVS is designed for a 12-ring + 1-center camera rig, a 12-camera dataset is missing the center/auxiliary camera entirely. This is likely to confuse new users who expect all 13 cameras to be present, and may cause errors or unexpected behavior when following tutorials.

## Solution

Re-record or source a clean 13-camera capture and repackage the example dataset. Update the dataset packaging script and any documentation referencing camera counts. Ensure the GitHub Release asset is updated with the new archive.
