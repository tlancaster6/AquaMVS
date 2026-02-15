Refractive Geometry
===================

This page explains the mathematical foundations of refractive ray tracing
for underwater multi-view stereo. The key challenge is that cameras are in
air while targets are underwater, with a flat water surface between them.
Light rays refract at the air-water interface according to Snell's law,
requiring a modified projection model.

Coordinate System
-----------------

AquaMVS inherits the coordinate system convention from AquaCal:

**World Frame**
    The origin is at the optical center of the reference camera (whichever
    camera was designated as reference during AquaCal calibration). Camera
    names are hardware identifiers (e.g., ``e3v82e0``), not sequential labels.

    * **+X**: Right (when facing the scene)
    * **+Y**: Forward (optical axis of reference camera)
    * **+Z**: Down (into the water)
    * **Units**: Meters throughout

**Camera Frame**
    OpenCV convention for individual cameras:

    * **+X**: Right
    * **+Y**: Down
    * **+Z**: Forward (optical axis)

**Pixel Coordinates**
    Image coordinates use ``(u, v)`` where:

    * ``u``: Column (horizontal pixel index)
    * ``v``: Row (vertical pixel index)
    * Origin at top-left corner

**Extrinsics Convention**
    Transformation from world to camera frame:

    .. math::

        \mathbf{p}_{\text{cam}} = \mathbf{R} \mathbf{p}_{\text{world}} + \mathbf{t}

    where :math:`\mathbf{R}` is the rotation matrix and :math:`\mathbf{t}`
    is the translation vector.

**Typical Geometry**
    In a standard AquaMVS setup:

    * Cameras are positioned near :math:`Z \approx 0`
    * Water surface is at :math:`Z = z_{\text{water}} > 0` (e.g., 0.978 m)
    * Underwater targets are at :math:`Z > z_{\text{water}}`
    * Interface normal is :math:`\mathbf{n} = [0, 0, -1]` (points from water toward air)

Camera Model
------------

AquaMVS uses a standard pinhole camera model with radial and tangential
distortion. All images are undistorted in a preprocessing step, so the
projection model operates on rectified images.

**Intrinsic Matrix**
    The intrinsic matrix :math:`\mathbf{K}` maps 3D camera coordinates to
    pixel coordinates:

    .. math::

        \mathbf{K} = \begin{bmatrix}
        f_x & 0 & c_x \\
        0 & f_y & c_y \\
        0 & 0 & 1
        \end{bmatrix}

    * :math:`f_x, f_y`: Focal lengths in pixels
    * :math:`c_x, c_y`: Principal point (optical center) in pixels

**Standard Projection** (no refraction)
    For a 3D point :math:`\mathbf{p}_{\text{cam}} = [X, Y, Z]^T` in camera
    frame:

    .. math::

        \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} =
        \frac{1}{Z} \mathbf{K} \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}

    This standard model is **not valid** for underwater targets due to
    refraction.

Refractive Ray Casting
-----------------------

Ray casting is the inverse operation of projection: given a pixel coordinate,
compute the 3D ray that passes through it. For refractive scenarios, this
process has four steps.

**Step 1: Pinhole Back-Projection**
    Cast a ray from the camera optical center through the pixel. In camera
    frame, the normalized ray direction is:

    .. math::

        \mathbf{d}_{\text{cam}} = \mathbf{K}^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}

    Normalize to unit length: :math:`\mathbf{d}_{\text{cam}} \leftarrow \mathbf{d}_{\text{cam}} / \|\mathbf{d}_{\text{cam}}\|`

**Step 2: Transform to World Frame**
    Camera optical center in world frame:

    .. math::

        \mathbf{C} = -\mathbf{R}^T \mathbf{t}

    Ray direction in world frame:

    .. math::

        \mathbf{d}_{\text{air}} = \mathbf{R}^T \mathbf{d}_{\text{cam}}

**Step 3: Intersect Water Surface**
    Find the point where the ray crosses the plane :math:`Z = z_{\text{water}}`.
    Parametric ray equation:

    .. math::

        \mathbf{p}(t) = \mathbf{C} + t \mathbf{d}_{\text{air}}

    At the water surface:

    .. math::

        C_z + t \, d_{z,\text{air}} = z_{\text{water}}

    Solving for :math:`t`:

    .. math::

        t = \frac{z_{\text{water}} - C_z}{d_{z,\text{air}}}

    The intersection point is:

    .. math::

        \mathbf{P} = \mathbf{C} + t \mathbf{d}_{\text{air}}

**Step 4: Apply Snell's Law**
    Refract the ray direction at the interface. See the next section for
    the vector form of Snell's law.

.. mermaid::

   graph TD
     A["Camera in Air<br/>(optical center C)"] -->|"incident ray d_air"| B["Water Surface<br/>z = z_water"]
     B -->|"refracted ray d_water<br/>(Snell's law)"| C["Underwater Target<br/>(z > z_water)"]
     style B fill:#4fc3f7,stroke:#0288d1,stroke-width:2px

Snell's Law: Vector Form
-------------------------

Snell's law relates the incident and refracted ray directions at an interface
between two media with different refractive indices.

**Scalar Form**
    The traditional scalar formulation:

    .. math::

        n_1 \sin \theta_i = n_2 \sin \theta_t

    where:

    * :math:`n_1 = n_{\text{air}} \approx 1.0`: Refractive index of air
    * :math:`n_2 = n_{\text{water}} \approx 1.333`: Refractive index of water
    * :math:`\theta_i`: Angle of incidence (ray to normal)
    * :math:`\theta_t`: Angle of refraction (transmitted ray to normal)

**Vector Form Derivation**
    For computational implementation, we need a vector formulation. Given:

    * :math:`\mathbf{d}_i`: Incident ray direction (unit vector)
    * :math:`\mathbf{n}`: Interface normal (unit vector, pointing from medium 2 to medium 1)
    * :math:`\eta = n_1 / n_2`: Ratio of refractive indices

    The angle of incidence:

    .. math::

        \cos \theta_i = -\mathbf{d}_i \cdot \mathbf{n}

    (Note: The negative sign accounts for :math:`\mathbf{d}_i` pointing toward
    the interface while :math:`\mathbf{n}` points away from it.)

    From Snell's law:

    .. math::

        \sin^2 \theta_t = \eta^2 (1 - \cos^2 \theta_i) = \eta^2 \sin^2 \theta_i

    The refracted ray direction is:

    .. math::

        \mathbf{d}_t = \eta \mathbf{d}_i + \left( \eta \cos \theta_i - \sqrt{1 - \sin^2 \theta_t} \right) \mathbf{n}

**Total Internal Reflection**
    If :math:`\sin^2 \theta_t > 1`, total internal reflection occurs and there
    is no refracted ray. This happens when:

    .. math::

        \eta^2 (1 - \cos^2 \theta_i) > 1

    In practice, this is rare for air-to-water transitions at typical viewing
    angles but must be checked. AquaMVS returns a validity mask to flag these
    cases.

**Implementation in AquaMVS**
    The vector form is implemented in :py:class:`aquamvs.projection.RefractiveProjectionModel`.
    For air-to-water refraction:

    * :math:`\mathbf{d}_i = \mathbf{d}_{\text{air}}` (incident ray from Step 3)
    * :math:`\mathbf{n} = [0, 0, -1]` (points from water toward air)
    * :math:`\eta = n_{\text{air}} / n_{\text{water}} \approx 0.75`
    * :math:`\mathbf{d}_t = \mathbf{d}_{\text{water}}` (refracted ray into water)

    The refracted ray originates at the water surface intersection point
    :math:`\mathbf{P}` and points into the water.

Depth Parameterization
-----------------------

Depth maps in AquaMVS use **ray depth** (distance along the refracted ray)
rather than world Z-coordinate. This parameterization is natural for plane
sweep stereo and simplifies depth estimation.

**Ray Depth Definition**
    For a pixel at :math:`(u, v)`, ray casting returns:

    * :math:`\mathbf{O}`: Ray origin (water surface intersection point)
    * :math:`\mathbf{d}`: Ray direction (unit vector pointing into water)

    A 3D point at ray depth :math:`d` is:

    .. math::

        \mathbf{p}(d) = \mathbf{O} + d \, \mathbf{d}

**World Z Conversion**
    To convert ray depth to world Z-coordinate:

    .. math::

        Z = O_z + d \, d_z

    where :math:`O_z` is the Z-component of the ray origin and :math:`d_z`
    is the Z-component of the ray direction.

**Why Ray Depth?**
    Ray depth is preferred over world Z for several reasons:

    1. **Plane Sweep**: Depth hypotheses naturally correspond to distances
       along the ray. Sweeping in world Z would require non-uniform sampling.

    2. **Depth Priors**: Sparse triangulation produces 3D points. Projecting
       onto the ray gives the depth directly without Z-to-depth conversion.

    3. **Numerical Stability**: Ray depth is well-conditioned even for
       near-vertical rays, while Z-based parameterization can be ill-conditioned.

**Depth Range Specification**
    Configuration files specify depth ranges in ray depth units (meters):

    .. code-block:: yaml

        reconstruction:
          depth_min: 0.05  # meters along ray
          depth_max: 2.0   # meters along ray

    These are independent of the water surface Z-coordinate or ray direction.

Connection to Code
------------------

The mathematical concepts described here are implemented in:

* :py:class:`aquamvs.projection.RefractiveProjectionModel`: Ray casting
  (``cast_ray``) and refractive projection (``project``).

* :doc:`/api/calibration`: Calibration data structures providing camera
  parameters (intrinsics, extrinsics, water surface position, refractive indices).

For the underlying NumPy reference implementation (used during AquaCal
calibration), see ``aquacal.core.refractive_geometry`` in the AquaCal library.

Next Steps
----------

Now that we understand how rays are cast through the refractive interface,
we can use them for depth estimation. The next section covers :doc:`dense_stereo`,
which evaluates photometric similarity at discrete depth hypotheses to build
dense depth maps.
