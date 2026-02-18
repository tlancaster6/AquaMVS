# QA Issues Found — Phase 06 CLI Execution

## General Notes:
- "aquamvs preprocess" was changed to "aquamvs temporal-filter" partway through QA. Any references to preprocess
in this doc refer to what is now temporal-filter

## Issue Template

When logging an issue, use this format:

```
### [Phase].[Plan] / [Step] — [Brief Title]
- **Description**: What went wrong
- **Impact**: Blocker / High / Low / Non-blocking
- **Status**: Fixed inline / Logged for later / Won't fix
- **Notes**: Any additional context or workaround
```

---

## Issues

### 6.1 /  1.1: natsort import
- **Description**: modulenotfound error for natsort
- **Impact**: blocker
- **Status**: fixed but needs cleanup
- **Notes**: natsort was recently added to the aquacal dependencies. Root bug is self-resolving, because natsort is now 
in aqucacal dependencies. But in the previous stage, instead of just installing the new dependency, we did some hacky
workaround (see .planning/phases/05-performance-and-optimization) like utility inlining in .benchmarks/ to avoid the
import entirely. These should be cleaned up now that the import works. 

### 6.1 / 2.1: preprocessing output video long
- **Description**: when running aquamvs preprocess with --format mp4, the output video is about as long as the input
video despite having many fewer frames. Add a flag for --output-fps, default to 30, and update the underlying code
to enforce.
- **Impact**: non-blocker
- **Status**: logged for later
- **Notes**: possible bug source indicator in console output: each video prints "writing video to xxxx.mp4 @0.0 fps"

### 6.1 / 2.1: optimize preprocessing
- **Description**: temporal median preprocessing is too slow
- **Impact**: Medium
- **Status**: partial fix implemented, further steps logged for later
- **Notes**: hybrid seeking mode, window-step sampling, and some calculation optimizations implemented,
but there is room for improvement. We need to (1) determine whether video decoding or median calculation is
the true bottleneck, and (2) implement a fix. If decoding is the bottleneck, explore hardware-accelerated decoding.
if its median calculation, switch to the C-optimized bottleneck.median or bottleneck.nanmedian function, with 
np.percentile(stack, 50, method='nearest') with overwrite_input=True (avoids float64 copy by using an in-place partial 
sort) as a fallback.

### 6.1 / 2.1: Non-blocking NAL unit error
- **Description**: Invalid NAL unit size error during preprocessing
- **Impact**: Low
- **Status**: logged for later
- **Notes**: During preprocessing, ffmpeg surfaces a non-blocking NAL-related error. Despite this, output appears good.
We need to determine whether this error indicates a real problem. If it is, implement a fix. If not, either silence 
this specific error or make a note in the docs that this is expected output. Sample error output below. 

10:38:01 [INFO] aquamvs.preprocess: Processing e3v831e-20260210T095008-095511.mp4: 1600x1200 @ 30.0 fps, 9120 frames
10:38:01 [INFO] aquamvs.preprocess: Window size: 300 frames, framestep: 900
10:41:18 [INFO] aquamvs.preprocess: Processed 9000/9120 frames, output 10 median frames
[h264 @ 000001eb9c426740] Invalid NAL unit size (395781 > 65136).
[h264 @ 000001eb9c426740] Error splitting the input into NAL units.
10:41:18 [INFO] aquamvs.preprocess: Complete: processed 9090 frames, produced 10 output frames

### 6.1 / 3.1: incomplete support for image dir input
- **Description**: We implemented support for using image directories as the source material instead of videos, but did
not properly wire it up to the CLI
- **Impact**: blocking 
- **Status**: fixed
- **Notes**: Renamed PipelineConfig.camera_video_map to camera_input_map throughout the entire codebase 
(src, tests, and documentation) to accurately reflect support for both video files and image directory inputs. 
Updated the CLI init command to rename --video-dir to --input-dir and added proper file structure detection that checks
for both video files (e.g., cam1.mp4) and image subdirectories (e.g., cam1/*.jpg). Modified error messages and 
validation logic to handle both input types, preventing false "No video files found" errors when using image 
directories. All 10 source files, 4 test files, and 4 documentation files were updated to maintain consistency with the new naming convention and dual input       support. 

### 6.2 / 1.3
- **Description**: unclear behavior of quality presets
- **Impact**: low
- **Status**: logged for later
- **Notes**: After aquamvs init, we get a config file with default values set and the "quality_preset" left as null. If
we change the quality preset, does it silently override other parameters in the config? If so, maybe the quality
preset functionality should instead be a feature of aquamvs init. User could optionally supply a quality preset level
during init, which would reflect in the config generated.

### 6.2 / 2.3: no summary output
- **Description**: running in sparse mode did not produce any output to the summary directory
- **Impact**: non-blocking
- **Status**: logged for later
- **Notes**: If this is the expected behavior then no change is needed. Only a bug if there is a visualization function
that is failing to run. 

### 6.3 / 3.2: missing output when running roma in full mode
- **Description**: running roma branch in full mode completes without error or warning, but seems to be missing some 
outputs, such as the fused point cloud visualizations. 
- **Impact**: medium
- **Status**: fixed
- **Notes**:

### 6.3 / 4.1: benchmarking is broken
- **Description**: CLI benchmark command was trying to import legacy functions. Stripped those out, but now it doesn't output anything very meaningful
- **Impact**: Medium
- **Status**: logged for later
- **Notes**: It appears the addition of the new benchmark mode did not go very well. We need to revisit the phase 5
planning docs to see what happened and likely perform a careful multi-step rebuild

### 6.4 / 3.1: STL write failure
- **Description**: ply to stl conversion fails
- **Impact**: low
- **Status**: logged for later
- **Notes**: non-critical. Fix if easy, remove stl option if not. Error log below: 

07:43:28 [INFO] aquamvs.surface: Loading mesh from output_roma\frame_000000\mesh\surface.ply
07:43:28 [INFO] aquamvs.surface: Loaded mesh: 398468 vertices, 795745 faces
07:43:28 [WARNING] aquamvs.surface: STL format does not support vertex colors (colors will be lost)
07:43:28 [INFO] aquamvs.surface: Exporting to output_roma\frame_000000\mesh\surface.stl (format: .stl)
[Open3D WARNING] Write STL failed: compute normals first.
Error: Export failed: Failed to write mesh to output_roma\frame_000000\mesh\surface.stl
Traceback (most recent call last):
  File "C:\Users\tucke\PycharmProjects\AquaMVS\src\aquamvs\cli.py", line 643, in export_mesh_command
    export_mesh(
  File "C:\Users\tucke\PycharmProjects\AquaMVS\src\aquamvs\surface.py", line 409, in export_mesh
    raise RuntimeError(f"Failed to write mesh to {output_path}")
RuntimeError: Failed to write mesh to output_roma\frame_000000\mesh\surface.stl

### 6.5 / 1.1 ImageDirectorySet missing read_frame method
- **Description**: ImageDirectorySet does not have all the methods that VideoSet does
- **Impact**: medium
- **Status**: blockers fixed, further refactor logged
- **Notes**: We should refactor to add an abstraction layer above ImageDirectorySet set and VideoSet. Possibly a class
that, base on input type, routes to the correct VideoSet or ImageDirectorySet class. VideoSet and ImageDirectorySet
should probably also share a base class that serves as a contract for shared methods and properties. To remove immediate
blockers, we added a read_frame method to ImageDirectorySet

### 6.5 / 1.1 — Profiler OOM errors
- **Description**: the profiler code, as originally written, used the torch profiler, which accumulates millions of
kernels over the full pipeline and causes an OOM/hang while trying to serialize them during __exit__
- **Impact**: Medium
- **Status**: fixed
- **Notes**: dropped torch.profiler entirely, switched to time.perf_counter + torch.cuda.Event for wall/GPU timing per 
stage, tracemalloc for CPU memory peaks per stage, and torch.cuda.memory_allocated / max_memory_allocated for GPU memory 
per stage

### 6.5 / 1.1: no profiler output
- **Description**: profiler now runs without error (after making the fix above) but the table/report printed to the 
console looks empty
- **Impact**: medium
- **Status**: logged for later
- **Notes**: See full console output below:

"
(AquaMVS) PS C:\Users\tucke\Desktop\021026\021026_prereset\temporal_medians> aquamvs profile config.yaml --frame 0
09:55:58 [WARNING] aquamvs.config: Unknown config keys in RuntimeConfig (ignored): ['save_depth_maps', 'save_point_cloud', 'save_mesh']

Profiling frame 0...

09:55:59 [INFO] aquamvs.pipeline.builder: Loading calibration from C:\Users\tucke\Desktop\021026\021026_prereset\temporal_medians\calibration.json
09:55:59 [WARNING] aquamvs.pipeline.builder: Ring cameras in calibration but missing input (skipped): ['e3v82f9']
09:55:59 [INFO] aquamvs.pipeline.builder: Found 11 ring cameras, 1 auxiliary cameras (of 12/1 in calibration)
09:55:59 [INFO] aquamvs.pipeline.builder: Computing undistortion maps
09:55:59 [INFO] aquamvs.pipeline.builder: Creating projection models
09:55:59 [INFO] aquamvs.pipeline.builder: Selecting camera pairs
09:55:59 [INFO] aquamvs.masks: Loaded 13 mask(s) from C:\Users\tucke\Desktop\021026\021026_prereset\temporal_medians\masks
09:55:59 [INFO] aquamvs.pipeline.builder: Config saved to C:\Users\tucke\Desktop\021026\021026_prereset\temporal_medians\output\config.yaml
09:55:59 [INFO] aquamvs.io: Detected 5 frames across 12 cameras (image directory input)
09:56:00 [INFO] aquamvs.pipeline.stages.undistortion: Frame 0: undistorting images
09:56:01 [INFO] aquamvs.pipeline.stages.undistortion: undistortion: 1096.7 ms
09:56:01 [INFO] aquamvs.pipeline.stages.dense_matching: Frame 0: running RoMa v2 dense matching (full mode)
Using cache found in C:\Users\tucke/.cache\torch\hub\facebookresearch_dinov3_adc254450203739c8149213a7a69d8d905b4fcfa
09:56:04 [INFO] dinov3: using base=100 for rope new
09:56:04 [INFO] dinov3: using min_period=None for rope new
09:56:04 [INFO] dinov3: using max_period=None for rope new
09:56:04 [INFO] dinov3: using normalize_coords=separate for rope new
09:56:04 [INFO] dinov3: using shift_coords=None for rope new
09:56:04 [INFO] dinov3: using rescale_coords=2 for rope new
09:56:04 [INFO] dinov3: using jitter_coords=None for rope new
09:56:04 [INFO] dinov3: using dtype=fp32 for rope new
09:56:04 [INFO] dinov3: using mlp layer as FFN
2026-02-17 09:56:15 INFO     romav2.romav2 - romav2:116 in __init__ - RoMa
                             v2 initialized.
10:04:33 [INFO] aquamvs.pipeline.stages.dense_matching: Frame 0: converting RoMa warps to depth maps
10:04:39 [INFO] aquamvs.pipeline.stages.dense_matching: dense_matching: 517887.9 ms
10:04:39 [INFO] aquamvs.pipeline.stages.fusion: Frame 0: skipping geometric consistency filter (RoMa path)
10:04:39 [INFO] aquamvs.pipeline.stages.fusion: Frame 0: fusing depth maps
10:06:23 [INFO] aquamvs.pipeline.stages.fusion: Frame 0: removed 499882 outliers (3.9%) from fused cloud
10:06:23 [INFO] aquamvs.pipeline.stages.fusion: fusion: 104270.3 ms
10:06:23 [INFO] aquamvs.pipeline.stages.surface: Frame 0: reconstructing surface
10:07:35 [INFO] aquamvs.pipeline.stages.surface: surface_reconstruction: 72269.8 ms
10:07:35 [INFO] aquamvs.pipeline.runner: Frame 0: complete
Profile Report (device: cuda)
Total time: 0.00 ms
Peak memory: 0.00 MB

+---------+-------------+-------------+----------------+-----------------+
| Stage   | Wall (ms)   | CUDA (ms)   | CPU Mem (MB)   | CUDA Mem (MB)   |
+=========+=============+=============+================+=================+
+---------+-------------+-------------+----------------+-----------------+

Top 3 Bottlenecks:
"

### 6.5 / 3.1 pipeline api references deprecated config values
- **Description**: when running the pipeline via the API, we get "Unknown config keys in RuntimeConfig 
(ignored): ['save_depth_maps', 'save_point_cloud', 'save_mesh']".
- **Impact**: low
- **Status**: logged for later
- **Notes**: We removed these config parameters to ensure everything needed for our 2-pass calculate-then-visualize
pipeline remodel was saved automatically. Probably just need to strip out some dead code, but confirm. 

---

## Final Summary

### Fixed During QA (5 issues)
| Issue | Description |
|-------|-------------|
| 6.1/1.1 | natsort import — fixed (root cause resolved by AquaCal dep update) |
| 6.1/3.1 | Incomplete image dir input — renamed camera_video_map → camera_input_map throughout |
| 6.3/3.2 | Missing output in RoMa full mode — fixed |
| 6.5/1.1 | ImageDirectorySet missing read_frame — added method |
| 6.5/1.1 | Profiler OOM — replaced torch.profiler with manual timing |

### Logged for Phase 7 (9 issues)

**Medium priority:**
| Issue | Description | Notes |
|-------|-------------|-------|
| 6.1/1.1 | natsort workaround cleanup | Remove hacky utility inlining in .benchmarks/ |
| 6.1/2.1 | Optimize preprocessing | Determine bottleneck (decoding vs median), apply targeted fix |
| 6.3/4.1 | Benchmarking broken | Needs careful rebuild per Phase 5 planning docs |
| 6.5/1.1 | Profiler output empty | Timing hooks not wired to report table |
| 6.5/1.1 | ImageDirectorySet refactor | Add shared base class / abstraction over VideoSet and ImageDirectorySet |

**Low priority:**
| Issue | Description | Notes |
|-------|-------------|-------|
| 6.1/2.1 | Preprocessing output video FPS | Add --output-fps flag, fix "writing @0.0 fps" |
| 6.1/2.1 | NAL unit error | Investigate if real problem; silence or document if benign |
| 6.2/1.3 | Quality presets unclear | Move preset application to init-time instead of silent override |
| 6.2/2.3 | No summary output in sparse mode | Check if expected or missing visualization |
| 6.4/3.1 | STL write failure | Compute normals before write, or remove STL option |
| 6.5/3.1 | Deprecated config keys warning | Strip dead save_depth_maps/save_point_cloud/save_mesh code |
