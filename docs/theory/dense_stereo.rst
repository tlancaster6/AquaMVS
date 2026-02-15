Dense Stereo Matching
=====================

This page explains the mathematical foundations of dense depth estimation
in AquaMVS. The goal is to compute a dense depth map for each reference
camera, assigning a depth value to every pixel. Two complementary approaches
are available: plane sweep stereo (with optional sparse feature guidance)
and dense feature matching.

Overview
--------

Dense depth estimation is the core reconstruction step, transforming 2D
images into metric 3D information. For each reference camera, we aim to
produce:

* **Depth map**: :math:`D(u, v)` giving ray depth at each pixel
* **Confidence map**: :math:`C(u, v)` indicating reliability of the depth estimate

AquaMVS offers two main pathways:

1. **Sparse → Dense**: Extract sparse features (SuperPoint + LightGlue),
   triangulate to get 3D points, use them to guide plane sweep depth range.

2. **Dense Matching**: Use RoMa v2 for dense correspondence between camera
   pairs, triangulate dense matches directly.

Both approaches account for refractive ray geometry (see :doc:`refractive_geometry`).

Sparse Feature Matching
------------------------

Sparse features provide initial 3D information and depth range priors for
plane sweep stereo.

**Feature Extraction**
    SuperPoint detects keypoints and computes descriptors on undistorted images.
    For a camera with :math:`H \times W` image:

    * Keypoints: :math:`\{(u_i, v_i)\}_{i=1}^N` (sub-pixel locations)
    * Descriptors: :math:`\{\mathbf{f}_i \in \mathbb{R}^{256}\}_{i=1}^N` (L2-normalized)

**Feature Matching**
    LightGlue matches descriptors between reference and source cameras using
    a learned correspondence network. For cameras :math:`A` and :math:`B`:

    * Input: Descriptors :math:`\{\mathbf{f}_i^A\}, \{\mathbf{f}_j^B\}`
    * Output: Match set :math:`\mathcal{M} = \{(i, j) : \mathbf{f}_i^A \leftrightarrow \mathbf{f}_j^B\}`

    LightGlue uses attention mechanisms to refine matches and prune outliers.

**Cross-Pair Triangulation**
    For each match :math:`(i, j)` with pixel coordinates :math:`(u_i^A, v_i^A)`
    and :math:`(u_j^B, v_j^B)`:

    1. Cast rays through both pixels using refractive ray model:

       .. math::

           \mathbf{r}_A: \mathbf{p}(t) = \mathbf{O}_A + t \, \mathbf{d}_A

       .. math::

           \mathbf{r}_B: \mathbf{p}(s) = \mathbf{O}_B + s \, \mathbf{d}_B

    2. Find closest point of approach (3D point minimizing distance to both rays).

    3. Compute ray depths :math:`t^*` and :math:`s^*` for the closest point.

    The sparse point cloud :math:`\{\mathbf{p}_k\}` is used to:

    * **Filter outliers**: Statistical outlier removal (points with few neighbors)
    * **Estimate depth range**: Compute percentile-based depth bounds per camera
      (e.g., 5th to 95th percentile avoids outlier contamination)

**Depth Range Computation**
    For reference camera :math:`R`, sparse points are projected onto reference
    rays to get ray depths :math:`\{d_1, \ldots, d_M\}`. The plane sweep depth
    range is:

    .. math::

        d_{\min} = \text{percentile}(\{d_i\}, 5\%), \quad
        d_{\max} = \text{percentile}(\{d_i\}, 95\%)

    This provides adaptive depth bounds without requiring manual specification.

Plane Sweep Stereo
------------------

Plane sweep stereo evaluates photometric similarity at discrete depth
hypotheses to build a cost volume, then extracts the best-matching depth
per pixel.

**Algorithm Overview**
    For each reference pixel :math:`(u, v)`:

    1. Sample depth hypotheses :math:`\{d_1, \ldots, d_D\}` uniformly in
       :math:`[d_{\min}, d_{\max}]`.

    2. For each depth :math:`d_k`:

       a. Back-project to 3D: :math:`\mathbf{p}_k = \mathbf{O} + d_k \, \mathbf{d}`
          (using refractive ray model).

       b. Project into each source camera :math:`S_j` to get pixel location
          :math:`(u_j, v_j)`.

       c. Sample source image :math:`I_j` at :math:`(u_j, v_j)` via bilinear
          interpolation.

       d. Compute photometric cost between reference and warped source patches.

    3. Aggregate costs across all source cameras.

    4. Select depth with minimum cost: :math:`\hat{d}(u, v) = \arg\min_k C(u, v, k)`.

.. mermaid::

   graph LR
     subgraph "Reference Camera"
       P["Pixel (u,v)"]
     end
     subgraph "Depth Hypotheses"
       D1["d₁"] --> X1["3D Point p₁"]
       D2["d₂"] --> X2["3D Point p₂"]
       DN["dₙ"] --> XN["3D Point pₙ"]
     end
     subgraph "Source Cameras"
       S1["Source 1<br/>Sample I₁(u₁,v₁)"]
       S2["Source 2<br/>Sample I₂(u₂,v₂)"]
     end
     P --> D1
     P --> D2
     P --> DN
     X1 --> S1
     X1 --> S2
     X2 --> S1
     X2 --> S2
     XN --> S1
     XN --> S2
     style X1 fill:#81c784,stroke:#388e3c
     style X2 fill:#81c784,stroke:#388e3c
     style XN fill:#81c784,stroke:#388e3c

**Cost Volume**
    The cost volume is a 3D tensor:

    .. math::

        \mathbf{C} \in \mathbb{R}^{H \times W \times D}

    where :math:`C(u, v, k)` is the aggregated photometric cost at pixel
    :math:`(u, v)` and depth hypothesis :math:`d_k`.

**Photometric Cost Function**
    AquaMVS uses **Normalized Cross-Correlation (NCC)** to measure local
    patch similarity. For reference pixel :math:`(u, v)` and source pixel
    :math:`(u', v')`, NCC in an :math:`w \times w` window is:

    .. math::

        \text{NCC}(u, v; u', v') = \frac{\sum_{(i,j) \in W} (I_R(i,j) - \bar{I}_R) (I_S(i,j) - \bar{I}_S)}{\sqrt{\sum_{(i,j) \in W} (I_R(i,j) - \bar{I}_R)^2} \sqrt{\sum_{(i,j) \in W} (I_S(i,j) - \bar{I}_S)^2}}

    where :math:`\bar{I}_R` and :math:`\bar{I}_S` are local means in the window
    :math:`W` centered at :math:`(u,v)` and :math:`(u',v')`, respectively.

    The cost is defined as:

    .. math::

        \text{Cost} = 1 - \text{NCC}

    so that:

    * **Cost = 0**: Perfect correlation (identical patches)
    * **Cost = 1**: Uncorrelated
    * **Cost = 2**: Perfect anti-correlation

**Cost Aggregation**
    When multiple source cameras are available, costs are combined via
    averaging:

    .. math::

        C(u, v, k) = \frac{1}{M} \sum_{j=1}^{M} \text{Cost}_j(u, v, k)

    where :math:`M` is the number of source cameras.

**Winner-Take-All Depth Selection**
    The depth estimate at each pixel is:

    .. math::

        \hat{d}(u, v) = d_{k^*}, \quad k^* = \arg\min_k C(u, v, k)

**Confidence Estimation**
    Confidence is derived from the cost distribution. A sharp minimum indicates
    high confidence. AquaMVS uses the **cost ratio**:

    .. math::

        \text{Confidence}(u, v) = 1 - \frac{C(u, v, k^*)}{C(u, v, k_2)}

    where :math:`k^*` is the best depth and :math:`k_2` is the second-best.
    High confidence when the best cost is much lower than the second-best.

    Confidence values are in :math:`[0, 1]`, with 1 indicating high reliability.

Dense Matching Alternative
---------------------------

As an alternative to plane sweep, AquaMVS supports **RoMa v2** for dense
correspondence estimation.

**RoMa Overview**
    RoMa (Robust Matching) v2 is a learned dense matcher that produces
    per-pixel correspondence fields between image pairs. Unlike sparse
    matchers, it predicts a match for every pixel.

    For cameras :math:`A` and :math:`B`:

    * Input: Undistorted images :math:`I_A, I_B`
    * Output: Dense correspondence map :math:`\mathbf{F}: (u_A, v_A) \to (u_B, v_B)`
    * Confidence map indicating match reliability

**Dense Triangulation**
    For each pixel :math:`(u_A, v_A)` in camera :math:`A` with match
    :math:`(u_B, v_B)` in camera :math:`B`:

    1. Cast rays through both pixels.
    2. Triangulate to get 3D point :math:`\mathbf{p}`.
    3. Compute ray depth for camera :math:`A`.

    This produces a dense point cloud directly, without plane sweep.

**Comparison: Plane Sweep vs. Dense Matching**

    +-------------------+-------------------------+---------------------------+
    | Aspect            | Plane Sweep             | Dense Matching (RoMa)     |
    +===================+=========================+===========================+
    | Coverage          | Full dense depth map    | Full dense matches        |
    +-------------------+-------------------------+---------------------------+
    | Speed             | Slower (evaluates all   | Faster (single forward    |
    |                   | depth hypotheses)       | pass)                     |
    +-------------------+-------------------------+---------------------------+
    | Accuracy          | High (multi-view        | Moderate (pairwise only)  |
    |                   | consensus)              |                           |
    +-------------------+-------------------------+---------------------------+
    | Robustness        | Robust to textureless   | Struggles with large      |
    |                   | regions (if enough      | viewpoint changes         |
    |                   | views)                  |                           |
    +-------------------+-------------------------+---------------------------+
    | Use Case          | High-quality            | Fast prototyping, preview |
    |                   | reconstruction          | reconstruction            |
    +-------------------+-------------------------+---------------------------+

Connection to Code
------------------

The algorithms described here are implemented in:

* :py:func:`aquamvs.dense.plane_sweep_stereo`: Main plane sweep function.
* :py:func:`aquamvs.dense.compute_ncc`: NCC cost computation.
* :py:func:`aquamvs.dense.extract_depth`: Winner-take-all depth selection and
  confidence estimation.
* :py:mod:`aquamvs.features.matching`: Sparse feature matching (SuperPoint + LightGlue).
* :py:mod:`aquamvs.dense.roma_depth`: Dense matching via RoMa v2.

For API details, see :doc:`/api/reconstruction`.

Next Steps
----------

Once depth maps are computed for all cameras, they must be fused into a
unified 3D representation. The next section covers :doc:`fusion`, which
describes multi-view depth fusion, geometric consistency filtering, and
surface reconstruction methods.
