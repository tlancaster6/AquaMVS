Multi-View Fusion and Surface Reconstruction
=============================================

This page explains the final stage of the AquaMVS pipeline: fusing depth
maps from multiple cameras into a unified point cloud, filtering geometric
inconsistencies, and reconstructing surface meshes.

Multi-View Depth Fusion
------------------------

Each camera produces an independent depth map (see :doc:`dense_stereo`).
These depth maps overlap spatially but may contain inconsistencies due to
occlusions, noise, or matching errors. Fusion combines multi-view information
to produce a single, cleaned point cloud.

**Problem Statement**
    Given :math:`N` cameras with depth maps :math:`\{D_i(u, v)\}_{i=1}^N`
    and confidence maps :math:`\{C_i(u, v)\}_{i=1}^N`, produce a fused point
    cloud :math:`\mathcal{P} = \{\mathbf{p}_j\}_{j=1}^M` where each point
    is geometrically consistent across multiple views.

**Geometric Consistency Filtering**
    For each pixel in reference camera :math:`R` with depth :math:`d_R(u, v)`:

    1. **Back-project to 3D**: Compute the 3D point using the refractive ray
       model (see :doc:`refractive_geometry`):

       .. math::

           \mathbf{p} = \mathbf{O}_R + d_R(u, v) \, \mathbf{d}_R(u, v)

       where :math:`\mathbf{O}_R` is the ray origin and :math:`\mathbf{d}_R`
       is the ray direction for pixel :math:`(u, v)`.

    2. **Project into source cameras**: For each source camera :math:`S_j`,
       project :math:`\mathbf{p}` to get pixel :math:`(u_j, v_j)`.

    3. **Compare depths**: Retrieve the depth :math:`d_j(u_j, v_j)` from the
       source depth map. Back-project the source pixel to 3D point :math:`\mathbf{p}_j`.

    4. **Compute 3D distance**:

       .. math::

           \Delta_j = \|\mathbf{p} - \mathbf{p}_j\|

    5. **Consistency check**: Mark as consistent if:

       .. math::

           \Delta_j < \tau_{\text{dist}}

       where :math:`\tau_{\text{dist}}` is a distance threshold (e.g., 0.01 m = 1 cm).

    6. **Count consistent views**: If the point is consistent with at least
       :math:`N_{\min}` source cameras (e.g., :math:`N_{\min} = 2`), retain it.
       Otherwise, discard as an outlier.

**Consistency Score**
    For visualization and analysis, each point is tagged with a consistency
    score:

    .. math::

        \text{score}(\mathbf{p}) = \frac{\text{\# consistent views}}{\text{total \# source cameras}}

    High scores indicate strong multi-view agreement.

**Fusion Pipeline Diagram**

.. mermaid::

   flowchart LR
     DM1["Depth Map 1<br/>(Camera A)"] --> GC["Geometric<br/>Consistency<br/>Filter"]
     DM2["Depth Map 2<br/>(Camera B)"] --> GC
     DMN["Depth Map N<br/>(Camera C)"] --> GC
     GC --> FPC["Fused<br/>Point Cloud"]
     FPC --> OR["Outlier<br/>Removal"]
     OR --> SR["Surface<br/>Reconstruction"]
     style GC fill:#ffb74d,stroke:#f57c00,stroke-width:2px
     style FPC fill:#81c784,stroke:#388e3c,stroke-width:2px

Point Cloud Generation
-----------------------

After geometric consistency filtering, valid depth pixels are converted to
3D points with color.

**Back-Projection**
    For each valid pixel :math:`(u, v)` in reference camera :math:`R` with
    depth :math:`d(u, v)`:

    .. math::

        \mathbf{p} = \mathbf{O}_R(u, v) + d(u, v) \, \mathbf{d}_R(u, v)

    where :math:`(\mathbf{O}_R, \mathbf{d}_R)` is the ray from the refractive
    projection model.

**Color Assignment**
    Point color is taken from the reference image at pixel :math:`(u, v)`:

    .. math::

        \mathbf{c} = I_R(u, v)

    For better color fidelity, colors can be averaged across consistent views
    (not currently implemented in AquaMVS).

**Statistical Outlier Removal**
    Even after consistency filtering, some outliers may remain. Statistical
    outlier removal cleans the point cloud:

    1. For each point :math:`\mathbf{p}_i`, find its :math:`k` nearest neighbors.

    2. Compute mean distance to neighbors: :math:`\bar{d}_i`.

    3. Compute global statistics across all points:

       .. math::

           \mu = \text{mean}(\{\bar{d}_i\}), \quad
           \sigma = \text{std}(\{\bar{d}_i\})

    4. Remove outliers where:

       .. math::

           \bar{d}_i > \mu + \lambda \sigma

       (e.g., :math:`k = 20`, :math:`\lambda = 2.0`).

    This filters isolated points far from the main surface.

**Merging Overlapping Regions**
    When multiple cameras view the same region, their point clouds overlap.
    AquaMVS currently retains all points (no explicit deduplication). For very
    dense clouds, voxel downsampling can reduce redundancy:

    .. math::

        \text{voxel\_size} = 0.001 \text{ m (1 mm)}

Surface Reconstruction
----------------------

Point clouds are useful but often a continuous surface mesh is desired for
visualization, physics simulation, or further processing. AquaMVS supports
three surface reconstruction methods.

Poisson Surface Reconstruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Overview**
    Poisson reconstruction solves for a smooth, watertight surface that best
    fits an oriented point cloud (points + normals). It's robust and produces
    high-quality meshes but may hallucinate geometry in regions without point
    coverage.

**Algorithm**
    Given point cloud :math:`\mathcal{P} = \{\mathbf{p}_i, \mathbf{n}_i\}_{i=1}^N`
    (points and normals):

    1. **Indicator Function**: Compute a volumetric indicator function
       :math:`\chi(\mathbf{x})` that is 1 inside the surface and 0 outside.
       The gradient of :math:`\chi` aligns with the normals:

       .. math::

           \nabla \chi(\mathbf{x}) \approx \mathbf{n}_i \text{ near } \mathbf{p}_i

    2. **Poisson Equation**: Solve the Poisson equation:

       .. math::

           \Delta \chi = \nabla \cdot \mathbf{V}

       where :math:`\mathbf{V}` is a vector field constructed from point normals.

    3. **Isosurface Extraction**: Extract the :math:`\chi = 0.5` isosurface
       using marching cubes to get the triangle mesh.

**Parameters**
    * **Depth**: Octree depth controlling mesh resolution (default: 9).
      Higher depth → finer mesh but slower computation.
    * **Density Filtering**: Poisson fills gaps, so low-density regions are
      trimmed using a density percentile threshold (e.g., 1st percentile).

**Pros**
    * Smooth, watertight mesh
    * Handles noise well
    * Good for general-purpose reconstruction

**Cons**
    * May fill holes with hallucinated geometry
    * Requires normal estimation (done automatically)

**Use Case**
    Best for smooth surfaces (e.g., sand beds, rock surfaces) where gaps
    should be interpolated.

Height-Field Reconstruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Overview**
    Height-field reconstruction projects the point cloud onto a regular XY
    grid and interpolates Z values. Assumes the surface is roughly planar and
    single-valued in Z (no overhangs).

**Algorithm**
    1. **Grid Definition**: Create a regular 2D grid in the XY plane:

       .. math::

           x_i \in [x_{\min}, x_{\max}], \quad
           y_j \in [y_{\min}, y_{\max}]

       with resolution :math:`\Delta` (e.g., 5 mm).

    2. **Interpolation**: For each grid cell :math:`(x_i, y_j)`, interpolate
       Z value from nearby points using linear interpolation:

       .. math::

           z_{ij} = \text{interp}\left(\{(x_k, y_k) \to z_k\}_{k=1}^N, (x_i, y_j)\right)

       (implemented via ``scipy.interpolate.griddata``).

    3. **Triangulation**: Connect neighboring grid cells with triangles to
       form a mesh. Each :math:`2 \times 2` grid cell produces 2 triangles.

    4. **Color Interpolation**: Interpolate point colors onto grid vertices
       similarly.

**Parameters**
    * **Grid Resolution**: Grid spacing in meters (default: 0.005 m = 5 mm).

**Pros**
    * Fast
    * Preserves fine detail
    * Simple and predictable

**Cons**
    * Cannot represent overhangs or vertical surfaces
    * Assumes planar geometry
    * Gaps in coverage → holes in mesh

**Use Case**
    Best for approximately planar surfaces viewed from above (e.g., water
    surface, sand bed in overhead view).

Ball Pivoting Algorithm (BPA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Overview**
    BPA "rolls" a virtual ball over the point cloud, creating triangles where
    the ball touches three points. Preserves detail and requires no normal
    orientation but is sensitive to sampling density.

**Algorithm**
    1. **Seed Triangle**: Find an initial triangle where a ball of radius
       :math:`r` rests on three points without containing other points.

    2. **Pivot**: For each triangle edge, pivot the ball around the edge until
       it touches a third point, forming a new triangle.

    3. **Grow**: Repeat pivoting until no new triangles can be formed.

**Parameters**
    * **Radii**: List of ball radii to try (e.g., ``[0.005, 0.01, 0.02]``).
      Multiple radii handle varying point density.

**Pros**
    * Preserves fine geometric detail
    * No normal estimation required
    * Non-watertight (doesn't fill gaps)

**Cons**
    * Sensitive to point density variations
    * May produce disconnected patches
    * Requires parameter tuning (ball radii)

**Use Case**
    Best for detailed surfaces with uniform point sampling where you want to
    avoid gap-filling (e.g., high-resolution scans of textured objects).

Mesh Simplification
-------------------

High-resolution meshes can have millions of faces, which is impractical for
visualization or downstream processing. Mesh simplification reduces face count
while preserving shape.

**Quadric Error Decimation**
    AquaMVS uses quadric error metrics to iteratively collapse edges:

    1. For each vertex, compute a quadric (a 4×4 matrix) representing the
       error of collapsing it to nearby positions.

    2. Iteratively collapse the edge with minimum error.

    3. Stop when target face count is reached.

**Parameters**
    * **Target Faces**: Desired number of triangles (e.g., 100,000).

**Example**
    A 2M-face Poisson mesh can be simplified to 100k faces with minimal
    visual difference.

Mesh Export Formats
-------------------

AquaMVS supports multiple mesh export formats:

* **PLY** (Polygon File Format): Simple, widely supported. Stores vertices,
  faces, and colors. Binary or ASCII.

* **OBJ** (Wavefront Object): Human-readable ASCII format. Stores geometry
  and can reference material files for textures.

* **STL** (Stereolithography): Used for 3D printing. Stores only triangle
  geometry (no color). Requires normals (auto-computed).

* **GLTF** (GL Transmission Format): Modern web-friendly format. Supports
  animation, materials, and embedded textures.

Connection to Code
------------------

The fusion and surface reconstruction algorithms are implemented in:

* :py:func:`aquamvs.fusion.filter_depth_map`: Geometric consistency filtering.
* :py:func:`aquamvs.fusion.fuse_point_clouds`: Multi-view point cloud merging.
* :py:func:`aquamvs.surface.reconstruct_poisson`: Poisson surface reconstruction.
* :py:func:`aquamvs.surface.reconstruct_heightfield`: Height-field reconstruction.
* :py:func:`aquamvs.surface.reconstruct_bpa`: Ball pivoting algorithm.

For API details, see :doc:`/api/reconstruction`.

Summary
-------

The fusion stage transforms per-camera depth maps into a unified 3D surface:

1. **Geometric Consistency Filtering**: Cross-view validation to remove outliers.
2. **Point Cloud Generation**: Back-projection with color assignment.
3. **Outlier Removal**: Statistical filtering to clean the point cloud.
4. **Surface Reconstruction**: Choose from Poisson (smooth), height-field
   (planar), or BPA (detailed).
5. **Mesh Simplification**: Reduce polygon count for practical use.
6. **Export**: Save as PLY, OBJ, STL, or GLTF.

This completes the AquaMVS reconstruction pipeline: from camera pixels, through
refractive ray tracing (:doc:`refractive_geometry`), dense stereo matching
(:doc:`dense_stereo`), to final 3D surface meshes.
