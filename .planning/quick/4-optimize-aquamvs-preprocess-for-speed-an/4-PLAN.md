---
phase: quick-4
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/aquamvs/preprocess.py
  - src/aquamvs/cli.py
autonomous: true
must_haves:
  truths:
    - "With large framestep, only frames in each output window are decoded (not all intermediate frames)"
    - "User can pass --exact-seek to force sequential reading (old behavior)"
    - "User can pass --window-step N to subsample frames within each median window"
    - "Buffer operations are O(1) amortized (deque, not list pop(0))"
    - "Pre-allocated numpy array avoids per-frame reallocation"
  artifacts:
    - path: "src/aquamvs/preprocess.py"
      provides: "Hybrid seek + sequential modes, window_step, deque buffer, pre-allocated stack"
    - path: "src/aquamvs/cli.py"
      provides: "--exact-seek and --window-step CLI flags"
  key_links:
    - from: "src/aquamvs/cli.py"
      to: "src/aquamvs/preprocess.py"
      via: "process_batch(exact_seek=, window_step=)"
      pattern: "exact_seek|window_step"
---

<objective>
Optimize `aquamvs preprocess` for speed and memory on long videos with large framestep values.

Purpose: A 12-hour video at 30fps has ~1.3M frames. With framestep=30 the current code still decodes all 1.3M frames sequentially. The hybrid seek approach decodes only ~window frames per output position, reducing I/O by orders of magnitude.

Output: Updated preprocess.py with hybrid seek mode (default), exact-seek fallback, window-step subsampling, and efficient buffer/array management. Updated CLI with new flags.
</objective>

<execution_context>
@C:/Users/tucke/.claude/get-shit-done/workflows/execute-plan.md
@C:/Users/tucke/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/aquamvs/preprocess.py
@src/aquamvs/cli.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Rewrite process_video_temporal_median with hybrid seek and optimized buffers</name>
  <files>src/aquamvs/preprocess.py</files>
  <action>
Rewrite `process_video_temporal_median` to support two modes and add new parameters:

**New function signature:**
```python
def process_video_temporal_median(
    video_path: Path,
    output_dir: Path,
    window: int = 30,
    framestep: int = 1,
    output_format: str = "png",
    exact_seek: bool = False,
    window_step: int = 1,
) -> int:
```

**Hybrid seek mode (default, exact_seek=False):**

For each output position `out_frame_idx` (spaced by `framestep`):
1. Compute window start: `seek_pos = max(0, out_frame_idx - window + 1)`
2. `cap.set(cv2.CAP_PROP_POS_FRAMES, seek_pos)` to seek directly
3. Read `window` frames (or fewer at start of video), applying `window_step` subsampling:
   - If `window_step > 1`, only keep every Nth frame read within the window
4. Compute median of collected frames
5. Write output

Log a warning on first use: "Using hybrid seek mode. If output quality looks wrong (seek artifacts), re-run with --exact-seek"

Key detail: cv2 seek is not always frame-accurate with compressed video. This is why --exact-seek exists as fallback. The hybrid mode is a pragmatic tradeoff — for temporal median, off-by-a-few-frames is acceptable.

**Exact seek mode (exact_seek=True):**

Keep the current sequential-read approach but with these optimizations:
- Use `collections.deque(maxlen=window)` instead of list + pop(0)
- Pre-allocate a numpy array `frame_stack = np.empty((window, height, width, 3), dtype=np.uint8)` and copy frames into it from the deque before median, avoiding `np.array(buffer)` reallocation
- Apply `window_step` subsampling: only add every Nth frame to the deque (count frames within the current window region)

**window_step logic (both modes):**

When `window_step > 1`, subsample frames entering the median buffer. For a window of 90 with window_step=3, only 30 frames enter the median computation. The subsampling is uniform within the window (take every 3rd frame).

In hybrid mode: read frames from seek_pos, keep every window_step-th frame read.
In exact mode: within the sliding window, keep every window_step-th frame (simplest: only append to deque when `(frame_idx - window_start) % window_step == 0`, or just track a counter).

**Efficient median computation (both modes):**

Pre-allocate the numpy stack array once before the main loop with shape `(effective_window_size, height, width, 3)` where `effective_window_size = ceil(window / window_step)`. Copy from deque/list into it before each median call. This avoids re-creating the array each iteration.

**Update process_batch** to pass through `exact_seek` and `window_step` parameters.

**Update process_batch signature:**
```python
def process_batch(
    input_path: Path,
    output_dir: Path | None = None,
    window: int = 30,
    framestep: int = 1,
    output_format: str = "png",
    exact_seek: bool = False,
    window_step: int = 1,
) -> dict[str, int]:
```

Keep all existing logging. Add timing log at end: "Completed in {elapsed:.1f}s ({output_count / elapsed:.1f} frames/s)".
  </action>
  <verify>
Run: `python -c "from aquamvs.preprocess import process_video_temporal_median, process_batch; print('Import OK')"`

Verify function signatures accept the new parameters:
`python -c "import inspect; from aquamvs.preprocess import process_video_temporal_median; sig = inspect.signature(process_video_temporal_median); print(sig); assert 'exact_seek' in sig.parameters; assert 'window_step' in sig.parameters; print('Signature OK')"`
  </verify>
  <done>
process_video_temporal_median supports hybrid seek (default) and exact-seek modes. window_step subsampling works in both modes. Buffer uses deque in exact mode, pre-allocated numpy array for median in both modes. process_batch passes through new params.
  </done>
</task>

<task type="auto">
  <name>Task 2: Add CLI flags and wire through to process_batch</name>
  <files>src/aquamvs/cli.py</files>
  <action>
In the preprocess subcommand argument block (after the --format argument, around line 804), add two new arguments:

```python
preprocess_parser.add_argument(
    "--exact-seek",
    action="store_true",
    default=False,
    help="Force sequential frame reading (slower but avoids seek inaccuracy in compressed video)",
)
preprocess_parser.add_argument(
    "--window-step",
    type=int,
    default=1,
    help="Sample every Nth frame within the median window (default: 1 = use all frames). "
         "E.g., --window 90 --window-step 3 uses 30 frames per median.",
)
```

In `preprocess_command` (around line 529), update the `process_batch` call to pass the new args:

```python
results = process_batch(
    input_path=input_path,
    output_dir=args.output_dir,
    window=args.window,
    framestep=args.framestep,
    output_format=args.format,
    exact_seek=args.exact_seek,
    window_step=args.window_step,
)
```

Note: argparse converts `--exact-seek` to `args.exact_seek` and `--window-step` to `args.window_step` automatically.
  </action>
  <verify>
Run: `python -m aquamvs --help` (should not error)
Run: `python -m aquamvs preprocess --help` (should show --exact-seek and --window-step flags)
  </verify>
  <done>
CLI exposes --exact-seek (store_true) and --window-step (int, default 1). Both are wired through preprocess_command to process_batch.
  </done>
</task>

<task type="auto">
  <name>Task 3: Smoke test with real video and timing comparison</name>
  <files></files>
  <action>
Run a quick timing comparison using the test video. Do NOT modify any source files in this task.

**Test 1 - Hybrid seek (new default):**
```bash
time python -m aquamvs preprocess "C:/Users/tucke/Desktop/021026/021026_prereset/videos/e3v82e0-20260210T095008-095511.mp4" \
  --output-dir "C:/Users/tucke/Desktop/021026/021026_prereset/temporal_medians/hybrid_test" \
  --window 30 --framestep 30 --format png
```

**Test 2 - Exact seek (old behavior):**
```bash
time python -m aquamvs preprocess "C:/Users/tucke/Desktop/021026/021026_prereset/videos/e3v82e0-20260210T095008-095511.mp4" \
  --output-dir "C:/Users/tucke/Desktop/021026/021026_prereset/temporal_medians/exact_test" \
  --window 30 --framestep 30 --exact-seek --format png
```

**Test 3 - Hybrid seek with window-step:**
```bash
time python -m aquamvs preprocess "C:/Users/tucke/Desktop/021026/021026_prereset/videos/e3v82e0-20260210T095008-095511.mp4" \
  --output-dir "C:/Users/tucke/Desktop/021026/021026_prereset/temporal_medians/windowstep_test" \
  --window 90 --framestep 30 --window-step 3 --format png
```

Report timing for each. If hybrid mode produces output, visually spot-check that the first output frame exists and has reasonable file size (> 100KB for a PNG frame).

If any test fails with a Python error, diagnose and fix the issue in preprocess.py before re-running.
  </action>
  <verify>
All three test runs complete without errors. Hybrid mode is faster than exact mode. Output PNG files exist and have reasonable file sizes.
  </verify>
  <done>
Timing comparison shows hybrid seek is significantly faster than exact seek for framestep=30. All modes produce valid output frames.
  </done>
</task>

</tasks>

<verification>
- `python -c "from aquamvs.preprocess import process_video_temporal_median; print('OK')"` imports clean
- `python -m aquamvs preprocess --help` shows all flags including --exact-seek, --window-step
- Hybrid seek mode produces output frames matching approximate count: total_frames / framestep
- Exact seek mode produces identical frame count to hybrid (may differ by 1-2 at boundaries)
- window_step reduces frames in median buffer (visible in log output)
</verification>

<success_criteria>
- Hybrid seek mode (default) decodes only ~window frames per output position instead of all frames
- --exact-seek flag provides fallback to sequential reading
- --window-step N subsamples within the median window
- Buffer uses deque (exact mode), pre-allocated numpy array for median stack
- All three test runs complete successfully with timing data
</success_criteria>

<output>
After completion, create `.planning/quick/4-optimize-aquamvs-preprocess-for-speed-an/4-SUMMARY.md`
</output>
