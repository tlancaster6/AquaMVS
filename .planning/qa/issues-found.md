# QA Issues Found — Phase 06 CLI Execution

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

### 6.1 / 2.1: remove preprocess --format option
- **Description**: remove non-png options for preprocess
- **Impact**: low
- **Status**: logged for later
- **Notes**: preprocess currently support jpeg output, but this introduces compressions artifacts and the size savings
are minimal. Switch to only png output and remove --format CLI option

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

### [Phase].[Plan] / [Step] — [Brief Title]
- **Description**: 
- **Impact**: 
- **Status**: 
- **Notes**:
