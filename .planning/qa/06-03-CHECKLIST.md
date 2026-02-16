# Phase 06-03: CLI QA Execution — Checklist

## Task 1: Run RoMa+full reconstruction and benchmark comparison

### Step 1: Prepare RoMa config
- [ ] 1.1: Copy config.yaml to config_roma.yaml
- [ ] 1.2: Edit config_roma.yaml and set matcher_type to "roma"
- [ ] 1.3: Edit config_roma.yaml and set output_dir to ./output_roma (to avoid overwriting LightGlue results)
- [ ] 1.4: Verify mask_dir, calibration_path, and device settings are kept same as LightGlue run

### Step 2: Run RoMa pipeline
- [ ] 2.1: Run `aquamvs run config_roma.yaml --device cuda`
- [ ] 2.2: Monitor GPU memory for CUDA OOM errors (RoMa is slower and more VRAM-hungry than LightGlue)
- [ ] 2.3: If CUDA OOM occurs, reduce image resolution or use FAST preset
- [ ] 2.4: Fix any blockers immediately

### Step 3: Verify RoMa outputs
- [ ] 3.1: Run `ls output_roma/frame_000000/depth_maps/*.npz` to verify depth map files exist
- [ ] 3.2: Run `ls output_roma/frame_000000/*.ply` to verify sparse cloud, fused cloud, and mesh files exist
- [ ] 3.3: Run `python -c "import open3d as o3d; pcd = o3d.io.read_point_cloud('output_roma/frame_000000/sparse_cloud_frame_000000.ply'); print(len(pcd.points), 'points')"` and verify point count
- [ ] 3.4: Verify RoMa typically produces denser matches than LightGlue

### Step 4: Run benchmark comparison
- [ ] 4.1: Run `aquamvs benchmark config.yaml --compare ./output ./output_roma`
- [ ] 4.2: Review comparison output for side-by-side metrics (point counts, mesh stats, runtime)
- [ ] 4.3: If --compare fails, diagnose and fix (may need adjusted benchmark config format)

### Step 5: Log issues
- [ ] 5.1: Append any non-blocking issues to .planning/qa/issues-found.md

## Task 2: User review and comparison assessment

- [ ] 6.1: Open RoMa mesh (output_roma/frame_000000/mesh_*.ply) and compare surface quality to LightGlue mesh
- [ ] 6.2: Open RoMa depth maps and compare smoothness and coverage to LightGlue depth maps
- [ ] 6.3: Review benchmark comparison output and verify metrics make sense (RoMa typically denser but slower)
- [ ] 6.4: Verify both matchers reconstruct the same approximate surface shape
- [ ] 6.5: Signal completion by typing "approved"
