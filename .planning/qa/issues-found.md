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

### 6.1 / 1.1 — Profiler OOM errors
- **Description**: the profiler code, as originally written, used the torch profiler, which accumulates millions of
kernels over the full pipeline and causes an OOM/hang while trying to serialize them during __exit__
- **Impact**: Medium
- **Status**: fixed
- **Notes**: dropped torch.profiler entirely, switched to time.perf_counter + torch.cuda.Event for wall/GPU timing per 
stage, tracemalloc for CPU memory peaks per stage, and torch.cuda.memory_allocated / max_memory_allocated for GPU memory 
per stage

### [Phase].[Plan] / [Step] — [Brief Title]
- **Description**: 
- **Impact**: 
- **Status**: 
- **Notes**:

### [Phase].[Plan] / [Step] — [Brief Title]
- **Description**: 
- **Impact**: 
- **Status**: 
- **Notes**:

### [Phase].[Plan] / [Step] — [Brief Title]
- **Description**: 
- **Impact**: 
- **Status**: 
- **Notes**:

### [Phase].[Plan] / [Step] — [Brief Title]
- **Description**: 
- **Impact**: 
- **Status**: 
- **Notes**:
