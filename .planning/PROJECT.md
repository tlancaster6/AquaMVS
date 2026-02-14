# AquaMVS

## What This Is

AquaMVS is a multi-view stereo reconstruction library for underwater scenes viewed through a flat water surface. It uses refractive camera models (Snell's law at the air-water interface) to produce high-resolution 3D reconstructions from synchronized above-water cameras. It supports both sparse (LightGlue + plane sweep) and dense (RoMa v2) reconstruction pathways and is designed as both a CLI pipeline and importable Python library.

## Core Value

Accurate refractive multi-view stereo reconstruction — cameras in air, geometry underwater, Snell's law bridging the two. If refraction isn't modeled correctly, nothing else matters.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

- ✓ Refractive projection model (Snell's law ray casting and projection) — existing
- ✓ Sparse feature extraction (SuperPoint, ALIKED, DISK via LightGlue) — existing
- ✓ Dense matching via RoMa v2 (warp-based and correspondence-based) — existing
- ✓ Plane-sweep stereo with NCC/SSIM cost metrics — existing
- ✓ Geometric consistency filtering across cameras — existing
- ✓ Depth map fusion (backprojection, voxel dedup, confidence weighting) — existing
- ✓ Surface reconstruction (Poisson, BPA, heightfield) — existing
- ✓ Best-view vertex coloring — existing
- ✓ CLI with init, run, export-refs, benchmark commands — existing
- ✓ YAML-based pipeline configuration — existing
- ✓ Visualization (depth maps, features, 3D scene, rig diagram) — existing
- ✓ Benchmark suite (parameter sweep, HTML reports) — existing
- ✓ Evaluation metrics (ICP alignment, cloud-to-cloud distance, reprojection error) — existing

### Active

<!-- Current scope. Building toward these. -->

- [ ] Runtime optimization (RoMa bottleneck investigation, plane sweep optimization)
- [ ] Internal accuracy benchmarks (RoMa vs LightGlue pathway comparison)
- [ ] Ground truth evaluation (charuco board calibration target comparison)
- [ ] Config simplification (reduce sprawl, improve validation, smarter defaults)
- [ ] Pipeline refactoring (factor 995-line pipeline.py, reduce code duplication)
- [ ] CLI polish (progress bars, clear error messages)
- [ ] API documentation
- [ ] User tutorials and examples
- [ ] PyPI packaging and publication
- [ ] Dependency pinning and reproducible installs (LightGlue git pin, RoMa workaround)

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- External MVS comparison (COLMAP, OpenMVS) — nice-to-have but not required for v1
- Temporal median preprocessing (fish removal) — designed but deferred
- Pinhole mode for comparison — future work
- Multi-GPU support — post-v1 optimization
- Automated overlap-aware pair selection — future improvement
- Mobile or web interface — desktop CLI/library only

## Context

- The pipeline is working and producing good results, especially the RoMa pathway
- AquaCal (refractive camera calibration) is a sibling project, installed as local editable dependency
- Target users: researchers (underwater 3D reconstruction, marine science) and applied engineers (aquaculture, surveying)
- Camera rig: 12 ring cameras (standard lens) + 1 center camera (auxiliary fisheye), looking down through flat water surface
- Existing codebase: ~15 modules, ~5K lines, test suite with integration tests
- Windows development environment (Git Bash/MINGW64), Linux deployment possible
- Known tech debt: pipeline.py complexity, config proliferation, RoMa PyPI workaround, LightGlue unpinned

## Constraints

- **Tech stack**: Python 3.10+, PyTorch for all math, NumPy only at AquaCal boundary
- **Dependency**: AquaCal must be published to PyPI (or vendored) before AquaMVS can be pip-installable by others
- **Platform**: Must work on Windows (Git Bash) and Linux
- **RoMa**: romav2 requires `--no-deps` install workaround on Python 3.10+ due to PyPI bug
- **LightGlue**: Installed from git, no stable PyPI release
- **GPU**: Optional — CPU must work, CUDA is opt-in via config

## Key Decisions

<!-- Decisions that constrain future work. Add throughout project lifecycle. -->

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| PyTorch over NumPy for internal math | GPU support, differentiability, device-agnostic code | ✓ Good |
| Dual pathway (LightGlue sparse + RoMa dense) | Different accuracy/speed tradeoffs for different use cases | ✓ Good |
| Ray depth (not world Z) for depth maps | Consistent with refractive geometry, avoids confusion | ✓ Good |
| Protocol-based ProjectionModel | Future-proofs for pinhole or other projection backends | ✓ Good |
| Package as both CLI + library | Serves both pipeline users and custom workflow developers | — Pending |

---
*Last updated: 2026-02-14 after initialization*
