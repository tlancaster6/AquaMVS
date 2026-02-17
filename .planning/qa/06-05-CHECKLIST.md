# Phase 06-05: CLI QA Execution — Checklist

## Task 1: Run profiler and Pipeline API smoke test

### Step 1: Run profiler
- [x] 1.1: Run `aquamvs profile config.yaml --frame 0`
- [ ] 1.2: If NotImplementedError is raised (stub from Phase 5), log it as a known issue
- [ ] 1.3: If profiler succeeds, verify all stages are present (undistortion, matching, depth_estimation, fusion, surface)
- [ ] 1.4: Check timing breakdown: does depth_estimation appear as the bottleneck?
- [ ] 1.5: Verify memory reporting is present in output

### Step 2: Test --output-dir flag (if profiler works)
- [ ] 2.1: Run `aquamvs profile config.yaml --frame 0 --output-dir ./profiling`
- [ ] 2.2: Check if Chrome trace JSON is produced (log if not implemented)

### Step 3: Pipeline API smoke test
- [ ] 3.1: Run the following Python command:
  ```
  python -c "
  from aquamvs import Pipeline
  from aquamvs.config import PipelineConfig

  config = PipelineConfig.from_yaml('config.yaml')
  config.runtime.device = 'cuda'
  pipeline = Pipeline(config)
  pipeline.run()
  print('Pipeline API: OK')
  "
  ```
- [ ] 3.2: Verify command exits with code 0 and prints "Pipeline API: OK"
- [ ] 3.3: Verify output matches that of `aquamvs run config.yaml`
- [ ] 3.4: If it fails with import or API errors, diagnose and fix

### Step 4: Final issue summary
- [ ] 4.1: Review .planning/qa/issues-found.md
- [ ] 4.2: Add a final summary section listing all issues found across Plans 01-05
- [ ] 4.3: Categorize issues as: fixed inline, logged for later, or won't fix

## Task 2: User review and final QA assessment

- [ ] 5.1: Review profiler output (if produced) and verify stage breakdown makes sense
- [ ] 5.2: Confirm Pipeline API produced same output as CLI `run` command
- [ ] 5.3: Review .planning/qa/issues-found.md and verify all critical issues are documented
- [ ] 5.4: Perform final QA assessment: all 7 CLI commands tested, both matchers validated, mesh export working, profiler tested, Pipeline API confirmed
- [ ] 5.5: Determine if any critical issues need addressing before v1.0
- [ ] 5.6: Signal completion by typing "approved"
