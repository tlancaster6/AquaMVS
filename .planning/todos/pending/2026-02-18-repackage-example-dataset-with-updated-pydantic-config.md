---
created: 2026-02-18T00:00:53.812Z
title: Repackage example dataset with updated Pydantic config
area: docs
files:
  - scripts/package_example_dataset.py
  - docs/tutorial/aquamvs-example-dataset/config.yaml
---

## Problem

The example dataset zip on GitHub Releases (v0.1.0-example-data) contains a pre-Pydantic config.yaml with legacy keys (`dense_stereo`, `viz`, `save_clouds`, `save_meshes`, flat `matcher_type`/`device` at root level, etc.) that won't load with the current `PipelineConfig` schema. The migration layer will emit warnings for some keys, but others like `dense_stereo` and `viz` are unrecognized and may cause validation errors.

## Solution

Update `scripts/package_example_dataset.py` to generate a current-schema config via `aquamvs init --preset balanced` instead of hardcoding a legacy YAML template. Then repackage the zip and upload to the GitHub Release (tag `v0.1.0-example-data`). Consider combining with the 13-camera dataset update todo if a re-record happens first.
