Theory and Concepts
===================

This section provides a detailed walkthrough of the mathematics behind
AquaMVS's refractive multi-view stereo reconstruction pipeline. It covers
the complete path from camera pixels to 3D surface meshes, with emphasis
on the refractive ray model that distinguishes underwater reconstruction
from standard multi-view stereo.

The pipeline consists of three main stages:

1. **Refractive Geometry**: Ray casting through a flat air-water interface
   using Snell's law to model refraction effects.

2. **Dense Stereo**: Computing depth maps via plane sweep stereo or dense
   feature matching, accounting for refractive ray geometry.

3. **Fusion and Surface Reconstruction**: Merging multi-view depth maps
   into a unified point cloud and extracting surface meshes.

.. toctree::
   :maxdepth: 2

   refractive_geometry
   dense_stereo
   fusion
