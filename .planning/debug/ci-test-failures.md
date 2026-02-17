---
status: investigating
trigger: "Investigate and fix 25 test failures + 5 errors in CI"
created: 2026-02-14T00:00:00Z
updated: 2026-02-14T00:00:00Z
---

## Current Focus

hypothesis: Tests expect old config structure that was reorganized into grouped Pydantic models during Phase 02/03 refactoring
test: Read current config.py structure and compare with test expectations
expecting: Tests will need updates to match new nested config structure
next_action: Read src/aquamvs/config.py to understand current structure

## Symptoms

expected: All tests pass on CI (GitHub Actions Ubuntu, Python 3.12)
actual: 25 failures, 5 errors, 445 passed, 109 skipped
errors: Config attribute errors, CLI argument mismatches, function signature changes, Pydantic validation changes, visualization pipeline issues, mock/patch path issues, surface method validation
reproduction: `pytest tests/ -m "not slow"` on CI
started: Tests likely broke during Phase 02/03 refactoring but weren't caught

## Eliminated

## Evidence

- timestamp: 2026-02-14T00:01:00Z
  checked: src/aquamvs/config.py
  found: Config refactored into nested Pydantic models (PreprocessingConfig, SparseMatchingConfig, DenseMatchingConfig, ReconstructionConfig, RuntimeConfig)
  implication: Tests still reference old flat structure (e.g., config.device, config.visualization, config.benchmark)

- timestamp: 2026-02-14T00:02:00Z
  checked: test_cli.py line 399
  found: Test accesses config.device.device but should access config.runtime.device
  implication: Tests need to use new nested path config.runtime.device

- timestamp: 2026-02-14T00:03:00Z
  checked: test_cli.py line 512
  found: Test expects run_command to be called without 'quiet' param, but it's now in RuntimeConfig
  implication: CLI command may have added quiet parameter

- timestamp: 2026-02-14T00:04:00Z
  checked: test_pipeline.py lines 16-27
  found: Tests import old config class aliases (DenseStereoConfig, DeviceConfig, etc.)
  implication: Tests need to import new config classes or use nested config structure

- timestamp: 2026-02-14T00:05:00Z
  checked: test_roma_depth.py line 9
  found: Tests import FusionConfig separately, should use ReconstructionConfig
  implication: Function signatures may have changed to accept consolidated configs

## Resolution

root_cause: Phase 02/03 config refactoring consolidated multiple config classes into grouped Pydantic models, but tests were not updated to match the new structure.
fix: Updated all test files to use new nested config structure:
  - test_cli.py: config.runtime.device instead of config.device.device
  - test_cli.py: Added quiet=False to run_command call expectations
  - test_cli.py: dense_stereo -> reconstruction in config
  - test_pipeline.py: Imported new config classes, updated all fixtures
  - test_pipeline.py: config.runtime.viz_enabled instead of config.visualization.enabled
  - test_pipeline.py: Fixed _collect_height_maps patch path
  - test_roma_depth.py: FusionConfig -> ReconstructionConfig, fusion_config -> reconstruction_config
  - test_benchmark/test_runner.py: config.runtime.benchmark_* instead of config.benchmark.*
  - test_features/test_extraction.py: FeatureExtractionConfig -> SparseMatchingConfig, Pydantic ValidationError
  - test_surface.py: SurfaceConfig -> ReconstructionConfig, Pydantic ValidationError
verification: Run pytest tests/ -m "not slow" to verify all tests pass
files_changed: [
  "tests/test_cli.py",
  "tests/test_pipeline.py",
  "tests/test_dense/test_roma_depth.py",
  "tests/test_benchmark/test_runner.py",
  "tests/test_features/test_extraction.py",
  "tests/test_surface.py"
]
