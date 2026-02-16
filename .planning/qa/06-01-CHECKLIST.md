# Phase 06-01: CLI QA Execution — Checklist

## Task 1: Run preprocess, init, and export-refs commands

### Step 1: Verify environment
- [x] 1.1: Run `python -c "import aquamvs; print('OK')"` to confirm package imports
- [x] 1.2: Run `nvidia-smi` to check GPU availability and VRAM

### Step 2: Run preprocess
- [x] 2.1: Run `aquamvs preprocess /path/to/videos --output-dir ./preprocessed --window 30 --framestep 3600 --format png` (adjust framestep based on actual FPS)
- [x] 2.2: Run `ls ./preprocessed/ | wc -l` and verify all frame directories exist
- [x] 2.3: Run `ls ./preprocessed/frame_000000/*.png | wc -l` and verify all png files exist

### Step 3: Run init
- [x] 3.1: Run `aquamvs init --video-dir ./preprocessed/frame_000000 --pattern "^([a-z0-9]+)\.png$" --calibration /path/to/calibration.json --output-dir ./output --config config.yaml` (adjust pattern if image filenames differ)
- [x] 3.2: Run `cat config.yaml` to verify valid YAML with camera_video_map entries
- [x] 3.3: Check that calibration_path is correct and all 13 cameras are matched in config.yaml

### Step 4: Run export-refs
- [x] 4.1: Run `aquamvs export-refs config.yaml --frame 0`
- [x] 4.2: Run `ls ./output/reference_images/*.png | wc -l` to verify 13 undistorted PNGs exist
- [x] 4.3: Verify all output files are non-zero size

### Step 5: Create issues tracking file
- [x] 5.1: Create `.planning/qa/issues-found.md` with any non-blocking issues encountered, or "No issues found" if clean run

## Task 2: User review and ROI mask creation

- [x] 6.1: Open preprocessed frame image (e.g., `preprocessed/frame_000000/e3v82e0.png`) and verify fish/debris removal effectiveness
- [x] 6.2: Open `config.yaml` and verify camera names, paths, and calibration reference are correct
- [x] 6.3: Open reference images in `output/reference_images/` and verify undistortion looks correct (straight lines should be straight, no severe cropping)
- [x] 6.4: Create ROI masks using preferred image editor — save as binary grayscale PNGs (0=exclude, 255=include) in `masks/` directory, named exactly `{camera_name}.png` (e.g., `e3v82e0.png`)
- [x] 6.5: Edit `config.yaml` to add `mask_dir: /absolute/path/to/masks`
- [x] 6.6: Signal completion by typing "approved"
