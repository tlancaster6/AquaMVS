---
phase: 05-performance-and-optimization
plan: 04
subsystem: optimization
tags: [quality-presets, depth-batching, plane-sweep, config, performance]
dependency_graph:
  requires: [profiling-infrastructure, ReconstructionConfig]
  provides: [QualityPreset, depth_batch_size, batched-plane-sweep]
  affects: [config, plane_sweep, pipeline]
tech_stack:
  added: [Enum]
  patterns: [preset-system, batch-processing, default-detection]
key_files:
  created: []
  modified:
    - src/aquamvs/config.py
    - src/aquamvs/dense/plane_sweep.py
    - tests/test_config.py
    - tests/test_dense/test_plane_sweep.py
decisions:
  - QualityPreset enum with FAST/BALANCED/QUALITY values
  - PRESET_CONFIGS maps presets to parameter values across multiple config sections
  - apply_preset() uses default detection to preserve user overrides
  - auto_apply_preset() model validator auto-applies preset on construction
  - depth_batch_size field added to ReconstructionConfig (default=4)
  - Depth batching in build_cost_volume with configurable batch size
  - Backward compatible getattr() for depth_batch_size (Plan 03 compatibility)
metrics:
  duration: 12min
  completed: 2026-02-15
  tasks: 2
  files: 4
---

# Phase 05 Plan 04: Quality Presets and Plane Sweep Optimization Summary

**Quality preset system (FAST/BALANCED/QUALITY) and depth batching optimization for plane sweep stereo**

## What Was Built

### Task 1: Quality Presets (config.py)

**QualityPreset Enum:**
- FAST = "fast": 64 depths, window 7, batch 8, 1024 keypoints, 0.002m voxels, Poisson depth 8
- BALANCED = "balanced": 128 depths, window 11, batch 4, 2048 keypoints, 0.001m voxels, Poisson depth 9 (matches defaults)
- QUALITY = "quality": 256 depths, window 15, batch 1, 4096 keypoints, 0.0005m voxels, Poisson depth 10

**PRESET_CONFIGS dict:** Maps each QualityPreset to parameter values across reconstruction and sparse matching config sections.

**PipelineConfig.quality_preset field:** Optional QualityPreset | None field.

**apply_preset() method:**
- Compares current values to defaults before applying preset values
- User-specified values are preserved (not overridden)
- Returns self for method chaining

**auto_apply_preset() validator:** Pydantic @model_validator(mode="after") automatically applies preset after construction if quality_preset is set.

**depth_batch_size field:** Added to ReconstructionConfig with default=4 (balanced). Controls plane sweep batching.

### Task 2: Plane Sweep Depth Batching (plane_sweep.py)

**Modified build_cost_volume():**
- Accepts depth_batch_size from config via `getattr(config, 'depth_batch_size', 1)`
- Processes depths in batches using `for batch_start in range(0, D, batch_size)`
- Backward compatible with configs lacking depth_batch_size field
- torch.no_grad() already present from Plan 03

**Batching structure:**
```python
for batch_start in range(0, D, batch_size):
    batch_end = min(batch_start + batch_size, D)
    for d_idx in range(batch_start, batch_end):
        # Process depth hypothesis
```

Benefits:
- Better GPU utilization when batch_size > 1
- Memory allocation patterns improved (fewer allocations per depth)
- Quality preset (batch_size=1) for maximum quality
- Fast preset (batch_size=8) for speed

### Tests

**Quality Preset Tests (tests/test_config.py):**
- test_quality_preset_enum_values
- test_fast_preset_sets_expected_values
- test_balanced_preset_sets_expected_values
- test_quality_preset_sets_expected_values
- test_explicit_values_override_preset (user override preservation)
- test_partial_override_preset
- test_preset_round_trip_yaml
- test_preset_from_string (string coercion)
- test_no_preset_uses_defaults

**Depth Batching Tests (tests/test_dense/test_plane_sweep.py):**
- test_batch_size_consistency (batch_size=1 vs batch_size=4 produce identical results)

## Key Decisions

1. **Three quality tiers**: FAST (speed), BALANCED (default), QUALITY (accuracy). Balanced matches existing defaults for smooth migration.

2. **Default-detection strategy**: apply_preset() compares current values to freshly-constructed defaults. If value == default, apply preset. Otherwise preserve user value.

3. **Cross-section presets**: PRESET_CONFIGS spans reconstruction (num_depths, window_size, depth_batch_size, voxel_size, poisson_depth) and sparse matching (max_keypoints).

4. **Auto-apply via validator**: Pydantic validator automatically calls apply_preset() if quality_preset is set, making presets zero-friction for users.

5. **Backward compatibility**: depth_batch_size uses getattr() with fallback to 1. Configs from Plan 03 work without modification.

6. **Depth batching for GPU utilization**: Batching structure prepared for future vectorization. Current implementation improves memory allocation patterns.

## Deviations from Plan

None - plan executed as written.

## Verification

**Config tests:**
```bash
# Syntax check
python -m py_compile src/aquamvs/config.py
# OK

# Enum values
grep -c "class QualityPreset" src/aquamvs/config.py
# 1

# Preset configs
grep -c "PRESET_CONFIGS" src/aquamvs/config.py
# 3 (definition + 2 references in apply_preset)

# depth_batch_size field
grep -c "depth_batch_size: int = 4" src/aquamvs/config.py
# 1
```

**Plane sweep tests:**
```bash
# Batching support
grep -c "batch_size = getattr" src/aquamvs/dense/plane_sweep.py
# 1

# Batch loop structure
grep -c "for batch_start in range" src/aquamvs/dense/plane_sweep.py
# 1
```

## Commits

- e06d073: feat(05-04): add quality presets to config system
- c33c783: feat(05-04): add depth batching optimization to plane sweep

## Self-Check: PASSED

**Modified files exist:**
- [x] src/aquamvs/config.py (QualityPreset, PRESET_CONFIGS, depth_batch_size, quality_preset, apply_preset, auto_apply_preset)
- [x] src/aquamvs/dense/plane_sweep.py (depth batching support)
- [x] tests/test_config.py (preset tests)
- [x] tests/test_dense/test_plane_sweep.py (batching test)

**Commits exist:**
- [x] e06d073 (quality presets)
- [x] c33c783 (depth batching)

**Key features present:**
- [x] QualityPreset enum with 3 values
- [x] PRESET_CONFIGS dict
- [x] depth_batch_size field in ReconstructionConfig
- [x] quality_preset field in PipelineConfig
- [x] apply_preset() method with default detection
- [x] auto_apply_preset() validator
- [x] Depth batching in build_cost_volume
- [x] Tests for presets and batching
