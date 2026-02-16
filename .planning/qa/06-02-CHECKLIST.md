# Phase 06-02: CLI QA Execution — Checklist

## Task 1: Run LightGlue+full reconstruction pipeline

### Step 1: Verify config is ready
- [x] 1.1: Confirm config.yaml has mask_dir set (from Plan 01)
- [x] 1.2: Confirm config.yaml has matcher_type set to "lightglue" (or default)
- [x] 1.3: Verify reconstruction section uses reasonable defaults or BALANCED quality preset
- [x] 1.4: Set runtime.device to "cuda" in config.yaml if not already set

### Step 2: Test video input path
- [x] 2.1: Copy config.yaml to config_video.yaml
- [x] 2.2: Edit config_video.yaml to point camera_video_map entries at raw video files (not preprocessed image directories)
- [x] 2.3: Run `aquamvs run config_video.yaml --sparse --device cuda`
- [x] 2.4: Verify command exits with code 0 and sparse cloud PLY file is produced with > 0 points
- [x] 2.5: Remove or rename video-based output directory to avoid interference with main run

### Step 3: Run full pipeline from preprocessed images
- [x] 3.1: Run `aquamvs run config.yaml --device cuda`
- [x] 3.2: Monitor for CUDA OOM errors; if hit, reduce quality (adjust depth_batch_size, use FAST preset, or reduce num_depths)
- [x] 3.3: Monitor for import errors, path issues, or calibration mismatches; fix any blockers immediately

### Step 4: Verify outputs exist
- [x] 4.1: Run `ls -lR output/frame_000000/` to check directory structure
- [x] 4.2: Verify depth maps: non-zero NPZ and PNG files in output/frame_000000/depth_maps/
- [x] 4.3: Verify sparse cloud: run `python -c "import open3d as o3d; pcd = o3d.io.read_point_cloud('output/frame_000000/sparse_cloud_frame_000000.ply'); print(len(pcd.points), 'points')"`
- [x] 4.4: Verify fused cloud exists: output/frame_000000/fused_cloud_*.ply
- [x] 4.5: Verify mesh exists: output/frame_000000/mesh_*.ply
- [x] 4.6: Save the output directory path for benchmark comparison in Plan 03

### Step 5: Log any non-blocking issues
- [x] 5.1: Append any issues to .planning/qa/issues-found.md

## Task 2: User review and quality assessment

- [x] 6.1: Open depth map PNGs (output/frame_000000/depth_maps/*_depth.png) and verify smooth depth gradients, no all-NaN maps, depth range ~1-2m
- [x] 6.2: Open sparse_cloud_*.ply in MeshLab or Open3D viewer and verify points cluster near expected surface with no extreme outliers
- [x] 6.3: Open fused_cloud_*.ply and verify it's denser than sparse cloud with effective outlier removal
- [x] 6.4: Open mesh_*.ply and verify surface continuity with no disconnected patches and reasonable triangle count
- [x] 6.5: Check if consistency maps were saved (if enabled in config)
- [x] 6.6: Signal completion by typing "approved"
