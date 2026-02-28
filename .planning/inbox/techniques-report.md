# AquaMVS: Key Techniques Report

This report summarizes the principal algorithms and methods used in the AquaMVS
multi-view stereo reconstruction pipeline, organized by pipeline stage. AquaMVS
supports two primary analysis pathways — a **sparse-to-dense pathway** using
learned keypoint matching (LightGlue) followed by plane-sweep stereo, and a
**dense matching pathway** using transformer-based dense correspondence
estimation (RoMa v2) — both unified by a shared refractive projection model,
depth-map fusion stage, and surface reconstruction backend.

---

## 1. Refractive Camera Model

All geometric operations pass through a refractive projection model that
accounts for light bending at a flat air–water interface via Snell's law.

**Back-projection (cast_ray):** Pixels are unprojected through a standard
pinhole model into camera-frame rays, transformed to world frame, intersected
with the horizontal water surface plane, and then refracted into the water
volume using the vectorized Snell's law relation:

$$\mathbf{d}_t = \frac{n_1}{n_2} \mathbf{d}_i + \left(\cos\theta_t - \frac{n_1}{n_2} \cos\theta_i\right) \hat{\mathbf{n}}$$

The resulting ray origins lie on the water surface and directions point
downward into the water.

**Forward projection (project):** Given a 3D underwater point, the
surface intersection point that satisfies Snell's law is found via
Newton–Raphson iteration (10 steps), then projected through the pinhole model
to pixel coordinates. This avoids the closed-form quartic and converges
reliably from a straight-line initial guess.

**Lens undistortion** is applied as a preprocessing step using OpenCV's
`initUndistortRectifyMap` (pinhole or fisheye model, depending on camera type),
producing updated intrinsics consumed by the refractive model.

> **Citations:** [1] (Snell's law in multi-media geometry), [2] (OpenCV camera model)

---

## 2. Sparse Feature Extraction and Matching (LightGlue Pathway)

### 2.1 Keypoint Detection

Three learned keypoint detectors are supported:

- **SuperPoint** (default): A self-supervised CNN trained with homographic
  adaptation that jointly detects interest points and computes 256-D
  descriptors.
- **ALIKED**: A differentiable keypoint detector producing 128-D descriptors,
  designed for end-to-end training with descriptor matching losses.
- **DISK**: A reinforcement-learning-based detector that optimizes a matching
  reward, producing 128-D descriptors.

Optional **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
preprocessing improves keypoint yield in low-contrast underwater imagery.

> **Citations:** [3] (SuperPoint), [4] (ALIKED), [5] (DISK), [6] (CLAHE)

### 2.2 Feature Matching with LightGlue

Keypoint correspondences between camera pairs are established using
**LightGlue**, a lightweight learned feature matcher with adaptive early-exit
inference. LightGlue operates on keypoints and descriptors from any of the
three extractors above. Camera pairs are canonicalized to avoid duplicate
triangulations from bidirectional matching.

> **Citations:** [7] (LightGlue)

---

## 3. Dense Matching (RoMa Pathway)

**RoMa v2** is a transformer-based dense matcher that produces a full warp
field mapping every pixel in a reference image to a corresponding location in
a source image, along with a per-pixel overlap certainty score. In the dense
pathway, raw warp fields are retained and converted directly to pairwise depth
maps via ray triangulation (Section 5.2), bypassing discrete keypoint
extraction entirely.

When used in a sparse context (benchmarking), correspondences are extracted
from the warp field using certainty thresholding followed by **spatially-uniform
subsampling** (32×32 grid bins, top-k per bin) to preserve spatial coverage
rather than clustering near high-certainty patches.

> **Citations:** [8] (RoMa)

---

## 4. Sparse Triangulation and Depth Range Estimation

### 4.1 Linear Least-Squares Ray Triangulation

Matched correspondences are triangulated by casting refracted rays through both
cameras and solving a linear least-squares system that minimizes the sum of
squared distances from the 3D point to all rays:

$$\mathbf{A} = \sum_i (\mathbf{I} - \mathbf{d}_i \mathbf{d}_i^\top), \quad \mathbf{b} = \sum_i \mathbf{A}_i \mathbf{o}_i, \quad \mathbf{A}\mathbf{p} = \mathbf{b}$$

A batched variant using `torch.einsum` for outer products enables efficient
vectorized triangulation of thousands of ray pairs simultaneously.

### 4.2 Triangulation Quality Filters

Three filters reject unreliable triangulations:
1. **Positive ray depth** on both rays (point must be in front of both cameras)
2. **Minimum intersection angle** (default 2°) to reject near-parallel rays
   where depth is poorly constrained
3. **Reprojection error** through both refractive projection models (default
   threshold: 3 px)

Additionally, a physical plausibility filter removes points above the water
surface or beyond a maximum expected depth.

### 4.3 Robust Depth Range Estimation

Sparse triangulated points are projected into each camera's ray-depth space.
The depth range for subsequent plane-sweep stereo is set to the **2nd–98th
percentile** of projected depths (plus a configurable margin), preventing
outlier sparse points from inflating the hypothesis range.

> **Citations:** [9] (midpoint triangulation)

---

## 5. Dense Depth Estimation

### 5.1 Plane-Sweep Stereo (LightGlue Pathway)

A classical plane-sweep stereo approach adapted for refractive geometry:

1. **Depth hypotheses** are sampled uniformly in ray-depth space (default 128
   planes).
2. For each hypothesis, reference pixels are back-projected through the
   refractive model to 3D, then forward-projected into source cameras and
   sampled via bilinear interpolation (`F.grid_sample`).
3. A **cost volume** (H × W × D) is constructed by evaluating photometric
   similarity at each depth hypothesis, averaged across all source views.
4. Depth is extracted via **winner-take-all** with **sub-pixel parabolic
   refinement** (fitting a parabola to the three cost samples around the
   minimum).

Two photometric cost functions are supported:
- **NCC (Normalized Cross-Correlation):** Box-filter windows with NaN-aware
  valid-pixel counting. Cost = 1 − NCC.
- **SSIM (Structural Similarity Index Measure):** Gaussian-weighted windows
  (σ = 1.5) with the Wang et al. (2004) constants. Cost = 1 − SSIM.

**Confidence** is estimated as the geometric mean of cost-based confidence
(1 − best_cost) and distinctness confidence (1 − best_cost / mean_cost along
the depth axis). Boundary hypotheses (at depth limits) are masked as invalid.

> **Citations:** [10] (plane-sweep stereo), [11] (SSIM), [12] (sub-pixel refinement in MVS)

### 5.2 Warp-to-Depth Triangulation (RoMa Pathway)

Rather than sweeping depth hypotheses, the dense warp field from RoMa is
converted to a pairwise depth map geometrically:

1. Each warp correspondence is converted to pixel coordinates in both cameras.
2. Refracted rays are cast through both cameras' projection models.
3. The batched linear least-squares triangulation (Section 4.1) yields 3D
   points, from which ray depth in the reference camera is extracted.

**Multi-view depth aggregation** combines pairwise depth maps for each
reference camera via a two-pass robust median: an initial median across all
source views is computed, inliers within a tolerance are identified, and a
refined inlier median is taken as the final depth. Pixels with fewer than a
minimum number of agreeing views are discarded. Confidence is the fraction
of agreeing source views.

**Normalized convolution upsampling** is used to resample warp-resolution
depth maps to full image resolution without eroding valid-pixel boundaries,
by separately upsampling the depth×mask and mask fields and dividing.

---

## 6. Multi-View Depth Fusion

### 6.1 Geometric Consistency Filtering (LightGlue Pathway)

Each pixel in a reference camera's depth map is cross-validated against all
other cameras' depth maps: the pixel is back-projected to 3D, projected into
each target camera's depth map, and the target's depth at that location is
compared to the expected depth. Pixels with fewer than a minimum number of
consistent views are discarded. This step is skipped in the RoMa pathway,
where multi-view consistency is already enforced during depth aggregation.

### 6.2 Point Cloud Fusion

Filtered/aggregated depth maps from all ring cameras are back-projected to 3D
and concatenated. The fused cloud undergoes:
- **Voxel-grid downsampling** to remove redundant points from overlapping views
- **Normal estimation** via KD-tree hybrid search (radius proportional to voxel
  size, max 30 neighbors)
- **Normal orientation** toward the camera origin (pointing upward from the
  underwater surface)
- **Statistical outlier removal** based on mean distance to k-nearest neighbors

> **Citations:** [13] (Open3D)

---

## 7. Surface Reconstruction

Three surface reconstruction methods are supported:

### 7.1 Screened Poisson Surface Reconstruction (default)

Produces a watertight mesh from oriented point clouds by solving a Poisson
equation relating the indicator function gradient to the point normals. A
density-based trimming step (1st percentile threshold) removes hallucinated
geometry in unobserved regions.

> **Citations:** [14] (Poisson surface reconstruction), [15] (screened variant)

### 7.2 Ball Pivoting Algorithm

A region-growing triangulation method that rolls a virtual ball of specified
radii over the point cloud, connecting triplets of points it contacts.
Multi-scale radii (1×, 2×, 4× the mean nearest-neighbor distance) are used.
Does not hallucinate beyond the data extent.

> **Citations:** [16] (Ball Pivoting Algorithm)

### 7.3 Height-Field Interpolation

Projects the point cloud onto a regular XY grid and interpolates Z values
using Delaunay-based linear interpolation (`scipy.interpolate.griddata`). The
grid is triangulated into a mesh. Best suited for approximately planar
surfaces such as sand beds.

---

## 8. Vertex Coloring

Mesh vertices are colored using a **best-view selection** strategy: for each
vertex, the camera with the most frontal viewing angle relative to the surface
normal (highest |dot(view_direction, normal)|) is selected from all ring
cameras whose projection falls within image bounds. Auxiliary (center/fisheye)
cameras are excluded due to compressed-FOV artifacts after undistortion.

---

## 9. Preprocessing

### 9.1 Temporal Median Filtering

A sliding-window temporal median filter over video frames removes transient
objects (fish, bubbles, debris) prior to reconstruction. Window subsampling
reduces memory requirements.

### 9.2 Cross-Camera Color Normalization

Two optional methods equalize color response across cameras:
- **Gain normalization:** Per-channel multiplicative gain matching each
  camera's mean to the cross-camera mean.
- **Histogram matching:** Per-channel CDF matching to the average CDF across
  all cameras via lookup table.

---

## Citations

[1] T. Treibitz, Y. Y. Schechner, C. Kunz, and H. Singh, "Flat Refractive
    Geometry," *IEEE Trans. Pattern Analysis and Machine Intelligence*,
    vol. 34, no. 1, pp. 51–65, 2012.

[2] Z. Zhang, "A Flexible New Technique for Camera Calibration," *IEEE Trans.
    Pattern Analysis and Machine Intelligence*, vol. 22, no. 11,
    pp. 1330–1334, 2000.

[3] D. DeTone, T. Malisiewicz, and A. Rabinovich, "SuperPoint:
    Self-Supervised Interest Point Detection and Description," in *Proc. CVPR
    Workshops*, 2018.

[4] X. Zhao, X. Wu, J. Miao, W. Chen, P. C. Y. Chen, and Z. Li, "ALIKED:
    A Lighter Keypoint and Descriptor Extraction Network via Deformable
    Transformation," *IEEE Trans. Instrumentation and Measurement*, vol. 72,
    2023.

[5] M. J. Tyszkiewicz, P. Fua, and E. Trulls, "DISK: Learning Local Features
    with Policy Gradient," in *Proc. NeurIPS*, 2020.

[6] K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization," in
    *Graphics Gems IV*, P. S. Heckbert, Ed. Academic Press, 1994,
    pp. 474–485.

[7] P. Lindenberger, P. Sarlin, and M. Pollefeys, "LightGlue: Local Feature
    Matching at Light Speed," in *Proc. ICCV*, 2023.

[8] J. Edstedt, Q. Sun, G. Bökman, M. Wadenbäck, and M. Felsberg, "RoMa:
    Robust Dense Feature Matching," in *Proc. CVPR*, 2024.

[9] R. I. Hartley and A. Zisserman, *Multiple View Geometry in Computer
    Vision*, 2nd ed. Cambridge University Press, 2004.

[10] R. T. Collins, "A Space-Sweep Approach to True Multi-Image Matching,"
     in *Proc. CVPR*, 1996.

[11] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image Quality
     Assessment: From Error Visibility to Structural Similarity," *IEEE Trans.
     Image Processing*, vol. 13, no. 4, pp. 600–612, 2004.

[12] Y. Furukawa and J. Ponce, "Accurate, Dense, and Robust Multiview
     Stereopsis," *IEEE Trans. Pattern Analysis and Machine Intelligence*,
     vol. 32, no. 8, pp. 1362–1376, 2010.

[13] Q.-Y. Zhou, J. Park, and V. Koltun, "Open3D: A Modern Library for 3D
     Data Processing," arXiv:1801.09847, 2018.

[14] M. Kazhdan, M. Bolitho, and H. Hoppe, "Poisson Surface Reconstruction,"
     in *Proc. Eurographics Symposium on Geometry Processing*, 2006.

[15] M. Kazhdan and H. Hoppe, "Screened Poisson Surface Reconstruction,"
     *ACM Trans. Graphics*, vol. 32, no. 3, 2013.

[16] F. Bernardini, J. Mittleman, H. Rushmeier, C. Silva, and G. Taubin,
     "The Ball-Pivoting Algorithm for Surface Reconstruction," *IEEE Trans.
     Visualization and Computer Graphics*, vol. 5, no. 4, pp. 349–359, 1999.
