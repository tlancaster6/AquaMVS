# Codebase Concerns

**Analysis Date:** 2026-02-14

## Tech Debt

**Pipeline module complexity:**
- Issue: `src/aquamvs/pipeline.py` is 995 lines (largest module), containing orchestration logic, visualization wiring, frame processing, and multiple conditional branches (lightglue vs roma, sparse vs full modes)
- Files: `src/aquamvs/pipeline.py`
- Impact: Difficult to follow control flow; changes to any pipeline stage require reading entire 995-line file; testing individual paths requires mocking multiple stages; future matcher or mode additions require more branching
- Fix approach: Factor orchestration logic into separate sub-modules: `_lightglue_stage()`, `_roma_stage()`, `_sparse_mode()`, `_full_mode()` helper functions within same file (easier review) or extract to dedicated `pipeline/stages.py` module if further growth occurs

**Config dataclass proliferation:**
- Issue: `src/aquamvs/config.py` is 465 lines with 9+ dataclasses (PipelineConfig, FeatureExtractionConfig, MatchingConfig, DenseMatchingConfig, FusionConfig, SurfaceConfig, OutputConfig, VizConfig, ColorNormConfig, BenchmarkConfig). Adding new pipeline stage requires new dataclass + integration into PipelineConfig + YAML serialization handling
- Files: `src/aquamvs/config.py`
- Impact: Config validation scattered across dataclass __post_init__ methods; hard to enforce cross-stage constraints (e.g., "if matcher_type=roma then DenseMatchingConfig applies, not MatchingConfig"); YAML round-trip requires boilerplate per new config
- Fix approach: Consolidate common config patterns (thresholds, ranges, flags) into reusable base classes; create a ConfigValidator class that enforces stage-specific rules; document config invariants as comments in each dataclass

**RoMa v2 dependency workaround:**
- Issue: `romav2` package on PyPI declares `dataclasses>=0.8` which doesn't exist for Python 3.10+. Documented workaround: install with `--no-deps` then explicit dependencies (einops, rich, tqdm) in pyproject.toml
- Files: `pyproject.toml`, `src/aquamvs/features/roma.py`
- Impact: Users following standard pip install get build error; requires manual workaround; brittle if romav2 version updates
- Fix approach: Maintain a local requirements-roma.txt with workaround documented, or pin romav2 to known-good version that fixes the bug; consider upstream PR to romav2 repository

**Hardcoded Windows Triton workaround:**
- Issue: `src/aquamvs/features/roma.py:_run_roma()` disables RoMa compilation (`compile=False`) to avoid Triton requirement on Windows. This is a platform-specific hack that may not be necessary on Linux or may break if RoMa compilation becomes critical for performance
- Files: `src/aquamvs/features/roma.py`
- Impact: Compilation disabled universally; potential performance loss if compilation becomes important; future RoMa versions may remove compilation option
- Fix approach: Check platform at runtime (`sys.platform == "win32"`); enable compilation only on non-Windows; add performance benchmark comparing compiled vs uncompiled on target deployment platform

## Known Bugs

**Pipeline mode dispatcher complexity:**
- Symptoms: Four execution paths (lightglue+sparse, lightglue+full, roma+sparse, roma+full) converge at different stages; visualizations route through shared viz functions but receive different inputs (sparse cloud vs fused cloud); dense stereo skipped in roma+full but not lightglue+full
- Files: `src/aquamvs/pipeline.py:process_frame()` (lines 300-700 approx), `src/aquamvs/config.py:PipelineConfig` (lines 1-50)
- Trigger: Any change to depth map generation (new filter, new cost function, new matching path) must account for both roma+full and lightglue+full divergence points
- Workaround: Extensive code comments at each branch point; all four paths tested in integration tests; matcher_type and pipeline_mode are independent config fields (not combined into enum)

**Visualization module initialization on headless systems:**
- Symptoms: `src/aquamvs/visualization/scene.py` probes Open3D's OffscreenRenderer availability at module import time via try/except. On Windows without GPU rendering context, C++ EGL errors print to native stderr even though Python catches the exception
- Files: `src/aquamvs/visualization/scene.py` (module-level OFFSCREEN_AVAILABLE flag)
- Trigger: Running on headless Windows CI or Windows without NVIDIA drivers
- Workaround: OFFSCREEN_AVAILABLE flag checked before every render call; renders gracefully skip with log warning if unavailable; test suite skips rendering tests with `@pytest.mark.skipif(not OFFSCREEN_AVAILABLE)`

**AquaCal VideoSet dependency in pipeline:**
- Symptoms: `src/aquamvs/pipeline.py` imports `VideoSet` from AquaCal and calls `VideoSet.iterate_frames()` directly. If AquaCal changes the VideoSet API or frame iteration contract, pipeline breaks
- Files: `src/aquamvs/pipeline.py` (lines 10-11, 260+), `src/aquamvs/cli.py` (line 225)
- Impact: Couples AquaMVS pipeline to AquaCal's I/O layer; violates the "AquaCal isolated to calibration.py" design goal stated in DESIGN.md
- Fix approach: Create an abstraction layer `aquamvs/io.py` with a `FrameIterator` protocol that wraps VideoSet; calibration.py imports AquaCal, io.py wraps it, pipeline.py uses protocol only

## Security Considerations

**No validation of external calibration data:**
- Risk: `src/aquamvs/calibration.py:load_calibration_data()` loads JSON directly from AquaCal without validating tensor ranges (K intrinsics, R rotation matrices, focal lengths, distortion coefficients). Malformed calibration could produce NaN or inf projections that crash downstream stages silently or produce corrupted output
- Files: `src/aquamvs/calibration.py` (load_calibration_data function)
- Current mitigation: RefractiveProjectionModel.__init__ does not validate parameter ranges; no assertions on R being orthogonal, K being positive, etc.
- Recommendations: Add validation in `CalibrationData.__post_init__()`: check K intrinsic sanity (focal length > 0, principal point in image bounds), verify R is orthogonal (det(R) ≈ 1, R @ R^T ≈ I), check water_z is physically plausible (> 0), clamp n_air and n_water to reasonable ranges (1.0-1.5)

**Config YAML can execute arbitrary code (low risk):**
- Risk: `src/aquamvs/config.py:PipelineConfig.from_yaml()` uses `yaml.safe_load()` which prevents code injection. However, user-provided mask_dir paths are passed directly to `Path.glob()` and file reads without validation
- Files: `src/aquamvs/config.py`, `src/aquamvs/masks.py`
- Current mitigation: Path sanitization via pathlib (prevents directory traversal with ..), `mask_dir` is optional (default None)
- Recommendations: Validate mask_dir is within expected parent directory; document that masks are trusted inputs (user-drawn images); add symlink traversal protection if deploying where untrusted paths possible

## Performance Bottlenecks

**RoMa warp-to-depth consensus filtering is vectorized but data movement is expensive:**
- Problem: `src/aquamvs/dense/roma_depth.py:aggregate_pairwise_depths()` is fully vectorized with torch operations but processes all pixels at full resolution (~560x560 x N_sources per pair). With 12 cameras x 4 sources each, that is 12 x 4 x 314K = 15M tensor elements. Upsampling to full resolution (1600x1200) blows this up further
- Files: `src/aquamvs/dense/roma_depth.py` (aggregate_pairwise_depths, roma_warps_to_depth_maps)
- Cause: Memory allocation and tensor operations; unnecessary NaN propagation during upsampling
- Improvement path: Profile memory usage for typical 1600x1200 rig; consider chunked processing if memory becomes bottleneck; cache upsampling interpolators across multiple depth maps

**Plane sweep stereo processes one reference camera at a time (necessary but slow):**
- Problem: `src/aquamvs/dense/plane_sweep.py:build_cost_volume()` iterates depth hypotheses one at a time (not batched by depth) to keep cost volume in single-GPU memory. For 128-192 depths and 5 source cameras, this is ~640-960 full-resolution warps per reference camera. With 12 ring cameras, that is 7680-11520 warps per frame
- Files: `src/aquamvs/dense/plane_sweep.py` (build_cost_volume, plane_sweep_stereo)
- Cause: Single-GPU constraint; no async I/O or tiling; no multi-GPU scheduling
- Improvement path: (Post-v1) Implement tiling for larger GPUs; async source image loading; multi-GPU batching. For v1, acceptable since plane sweep is deterministic and correctly implemented; room for optimization is clear but not critical

**Geometric consistency filtering is O(N_cameras) per pixel:**
- Problem: `src/aquamvs/fusion.py:filter_depth_map()` back-projects each pixel, reprojects into all other cameras, and samples their depth maps. For 12 cameras and ~2M pixels, this is 24M depth map samples per filter pass
- Files: `src/aquamvs/fusion.py` (filter_depth_map, filter_all_depth_maps)
- Cause: Full O(N^2) camera pairing; no occlusion-aware pruning
- Improvement path: For sparse filtering (fast loop), this is acceptable. Could optimize by grouping cameras by visibility (only test cameras that can actually see pixel), but COLMAP-style improvements are deferred to post-v1

**Sparse cloud outlier filtering uses percentiles which requires sort:**
- Problem: `src/aquamvs/triangulation.py:compute_depth_ranges()` calls `torch.quantile()` on all sparse points per camera. quantile requires sorting (O(N log N)). With thousands of points and 12 cameras, this adds up
- Files: `src/aquamvs/triangulation.py` (compute_depth_ranges)
- Cause: Robust to outliers but not optimized for speed
- Improvement path: Profile on real data; if bottleneck, switch to faster robust range estimation (e.g., median absolute deviation with fixed multiplier instead of percentiles)

## Fragile Areas

**Feature extraction with CLAHE is order-dependent:**
- Files: `src/aquamvs/features/extraction.py` (_apply_clahe, extract_features)
- Why fragile: CLAHE preprocessing enhances texture but changes image statistics fundamentally. Swapping CLAHE on/off changes detection results; swapping position (before/after normalization) changes results; clip_limit parameter is not rigorously tuned
- Safe modification: (1) Keep CLAHE off by default to avoid surprising users with unexpected keypoint distribution, (2) when enabling, use conservative clip_limit (2.0 default is reasonable), (3) document that CLAHE is for weakly-textured surfaces only, (4) test that CLAHE on/off produces reasonable point counts before deploying
- Test coverage: Tests exist for CLAHE on/off, but no stress tests on realistic underwater images with sand/sediment textures; no sensitivity analysis on clip_limit

**RoMa matcher is closed-source and inference-only:**
- Files: `src/aquamvs/features/roma.py` (match_pair_roma, _run_roma, _extract_correspondences)
- Why fragile: RoMa v2 model weights and architecture are not controllable; DINOv3 backbone claims rotation robustness but is unverified on 30-60 deg roll angles typical of ring rig; coordinate conventions (warp resolution vs image resolution) are implicit in RoMa's output and must be reverse-engineered from docs
- Safe modification: (1) Never assume RoMa output coordinate system; always validate against test geometries, (2) test P.42 rotation robustness evaluation regularly when upgrading RoMa versions, (3) have fallback to LightGlue if RoMa fails on any frame (already possible via config.matcher_type)
- Test coverage: P.42 rotation robustness tests exist but are marked @pytest.mark.slow and may be skipped in CI; no stress tests on failure modes (e.g., what happens when RoMa returns all-zero warp certainty)

**Undistortion map generation is one-time only:**
- Files: `src/aquamvs/calibration.py` (compute_undistortion_maps), `src/aquamvs/pipeline.py:setup_pipeline()`
- Why fragile: If camera calibration changes between sessions, pipeline doesn't detect it; reloads same undistortion maps; produces corrupted output without warning
- Safe modification: (1) Store hash of calibration JSON in output directory, (2) verify hash on reload matches current calibration, (3) warn and re-compute if mismatch detected. (Currently not implemented.)
- Test coverage: Tests mock calibration data but don't test cross-session recalibration scenarios

**Best-view color selection uses dot product but doesn't handle occlusions:**
- Files: `src/aquamvs/coloring.py` (best_view_colors), `src/aquamvs/pipeline.py` (_sparse_cloud_to_open3d)
- Why fragile: Viewing angle is computed as dot(view_dir, normal) but doesn't check if other geometry occludes the point from that camera. A camera may be most perpendicular but see the point through the sand, producing wrong color
- Safe modification: (1) This is a known limitation; document that color quality depends on mesh/point cloud coverage, (2) future multi-camera weighted blending could mitigate by averaging adjacent cameras' views, (3) do not attempt occlusion inference without proper ray-tracing (out of scope)
- Test coverage: Tests verify camera selection logic but assume open visibility; no occlusion stress tests

## Scaling Limits

**Single-GPU execution limits batch size and frame rate:**
- Current capacity: 1600x1200 image pairs, 128-192 depth hypotheses, 5 source views → ~300MB per camera during plane sweep; 12 cameras x 5-10 min = 60-120 min per frame on mid-range GPU
- Limit: Upgrading to 2K or 4K images would overflow single GPU memory; multiple frames in parallel would require multi-GPU infrastructure
- Scaling path: (Post-v1) Implement tiled processing within single GPU (slice images into 512x512 regions, process sequentially), or multi-GPU batching with torch.nn.DataParallel or distributed training framework

**Config YAML growth without schema validation:**
- Current capacity: ~500 lines per typical config (camera names, stage parameters, output settings). Configuration sprawl as new stages added (temporal median, advanced fusion, etc.)
- Limit: YAML becomes unwieldy; users misunderstand parameter interactions; no IDE schema validation
- Scaling path: Convert config to Pydantic v2 models with built-in validation and JSON schema generation; provide VS Code schema plugin for config files

**Sparse cloud deduplication via voxel grid is memory-bound:**
- Current capacity: ~300K-500K triangulated points per frame (typical sparse rig); voxel dedup reduces to ~50K-100K (voxel size 1mm); Open3D's voxel_down_sample creates full sparse grid in memory
- Limit: Ultra-dense sparse clouds (millions of points from RoMa) would cause OOM; no streaming voxel dedup
- Scaling path: (Post-v1) Implement streaming voxel grid or octree-based dedup for large point clouds

## Dependencies at Risk

**LightGlue installed from git (no released version constraint):**
- Risk: `pyproject.toml` depends on `lightglue @ git+https://...` without pinning commit hash. Upstream changes could break compatibility
- Impact: Fresh installs may pick up latest development version; repeatable builds impossible
- Migration plan: (1) Pin to specific commit hash in git URL (https://github.com/cvg/LightGlue.git@abc123def456), (2) monitor upstream releases for official PyPI package, (3) consider vendoring LightGlue detection/matching code if upstream becomes unmaintained

**OpenCV version constraint is loose:**
- Risk: `pyproject.toml` specifies `opencv-python>=4.6` with no upper bound; major versions (4.x → 5.x) may break API compatibility
- Impact: `cv2.remap()` (undistortion), `cv2.getOptimalNewCameraMatrix()` (calibration) are stable but future versions could change
- Migration plan: Add upper bound `opencv-python>=4.6,<5.0` if 5.0 is released; test against latest 4.x version regularly

**AquaCal is editable dependency without version pin:**
- Risk: AquaCal is installed locally via `pip install -e ../AquaCal`, not from PyPI. AquaMVS only imports from `calibration.py` and `io/video.py`; changes to these modules break AquaMVS without warning
- Impact: Couples development; can't reproduce past AquaMVS runs without matching AquaCal checkout
- Migration plan: (Future) Publish AquaCal to PyPI with semantic versioning; pin AquaMVS to compatible AquaCal version range; document expected AquaCal version in README

## Missing Critical Features

**No temporal median preprocessing (P.49-P.51):**
- Problem: Fish and transient objects contaminate sparse clouds; temporal median preprocessing is designed but not implemented. Pipeline currently requires external frame selection or masking
- Blocks: Users cannot easily remove fish from underwater footage; time-series reconstructions include floating debris as false 3D geometry
- Alternative: Use ROI masks (P.35) to exclude tank regions where fish are common; use output.save_point_cloud=False if only mesh output is desired

**No automated pair selection (future improvement):**
- Problem: Pair selection is fixed N-nearest-neighbors by proximity. Overlap-aware pair selection would reject pairs with insufficient image overlap, improving matching success and reducing wasted computation
- Blocks: Users with non-uniform camera rigs (e.g., some cameras pointing sideways) cannot configure pairs correctly
- Workaround: `pairs.py:select_pairs()` is user-callable; developers can write custom pair selection scripts and override via manual PipelineConfig construction

**No pinhole mode for comparison:**
- Problem: DESIGN.md lists pinhole mode as future work; users cannot compare refractive vs. pinhole reconstruction on same data to quantify refraction impact
- Blocks: Validation that refractive model is necessary; comparison studies with standard MVS
- Workaround: Use external MVS pipeline (COLMAP, OpenMVS) on same data for comparison

## Test Coverage Gaps

**Untested: RoMa v2 on extreme camera rolls:**
- What's not tested: P.42 rotation robustness tests check 30-60 deg rolls but are marked @pytest.mark.slow and may be skipped; no stress tests on 90-180 deg misalignments
- Files: `tests/test_features/test_roma.py` (P.42 test is lightweight)
- Risk: RoMa may fail silently on untested roll angles; users won't know until running on real rig
- Priority: Medium (low likelihood on actual ring rig, but high impact if it fails in field)

**Untested: Full pipeline on real-world textured vs. textureless surfaces:**
- What's not tested: Pipeline integration tests use synthetic checkerboard patterns; real underwater sand, sediment, and biological crusts have very different statistical properties
- Files: `tests/test_integration.py` (uses synthetic 0.02m checkerboard)
- Risk: CLAHE, ALIKED, or geometric consistency filtering may perform differently on real textures; detection starvation may be worse than expected
- Priority: High (core use case; cannot be validated in unit tests)

**Untested: Fusion consistency filtering on large baseline camera pairs:**
- What's not tested: Geometric consistency filtering is tested on overlapping views but not on edge-case baselines (e.g., opposite sides of rig, 60-90 deg apart)
- Files: `tests/test_fusion.py` (uses nearby cameras, ~30 deg apart)
- Risk: Wide-baseline pairs may not overlap; depth map resampling may fail; consistency checks become noisy
- Priority: Medium (affects pair selection tuning but not core fusion correctness)

**Untested: Color normalization on cameras with very different exposure:**
- What's not tested: Color normalization (histogram matching, gain compensation) is implemented but tests use synthetic uniform-color inputs
- Files: `src/aquamvs/coloring.py`, `tests/test_coloring.py` (all synthetic)
- Risk: Real underwater tanks have uneven lighting; some cameras may be shadowed; normalization may overcorrect or introduce artifacts
- Priority: Low (non-critical for reconstruction quality, affects visualization only)

---

*Concerns audit: 2026-02-14*
