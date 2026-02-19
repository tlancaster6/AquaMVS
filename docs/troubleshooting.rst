.. _troubleshooting:

Troubleshooting
===============

This page documents solutions to common issues encountered during installation,
pipeline execution, notebook usage, and configuration. If your issue is not
listed here, please open a GitHub issue at
`https://github.com/tlancaster6/AquaMVS/issues <https://github.com/tlancaster6/AquaMVS/issues>`_.

.. contents:: Contents
   :local:
   :depth: 2

Installation Issues
-------------------

PyTorch not found
   **Problem:** ``ModuleNotFoundError: No module named 'torch'`` when importing
   AquaMVS or running any CLI command.

   **Solution:** PyTorch must be installed manually before installing AquaMVS,
   as it is not declared as a pip dependency (to allow users to choose the
   appropriate CUDA version)::

       # CUDA 12.1 (recommended for most NVIDIA GPUs)
       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

       # CPU-only (for systems without a CUDA-capable GPU)
       pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

   See `pytorch.org/get-started <https://pytorch.org/get-started/locally/>`_ for
   all available combinations.

LightGlue or RoMa import errors
   **Problem:** ``ModuleNotFoundError: No module named 'lightglue'`` or
   ``No module named 'romav2'`` when running the pipeline.

   **Solution:** These packages are not available on PyPI and must be installed
   directly from their repositories::

       pip install git+https://github.com/cvg/LightGlue.git@edb2b83
       pip install git+https://github.com/tlancaster6/RoMaV2.git

   Both commands require git to be installed and network access to GitHub.

CUDA version mismatch
   **Problem:** ``RuntimeError: CUDA error: no kernel image is available for
   execution on the device`` or similar CUDA version mismatch errors at runtime.

   **Solution:** Reinstall PyTorch with the CUDA version that matches your
   installed CUDA toolkit. Check your CUDA version with ``nvidia-smi`` and
   select the corresponding wheel from `pytorch.org <https://pytorch.org/get-started/locally/>`_.
   If in doubt, the CPU-only build works on all hardware::

       pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Pipeline Errors
---------------

No cameras matched
   **Problem:** ``aquamvs init`` reports ``[WARNING] No cameras matched`` and
   generates an empty camera list.

   **Solution:** The ``--pattern`` regex must contain exactly one capture group
   that extracts the camera identifier from the filename. Test your pattern
   before using it::

       python -c "import re; print(re.match(r'^([a-z0-9]+)-', 'e3v82e0-cam1.mp4').group(1))"

   The output should be the camera name (e.g., ``e3v82e0``). Adjust the pattern
   until all camera filenames match. Common patterns:

   - ``^([a-z0-9]+)-``: matches ``e3v82e0`` from ``e3v82e0-cam1.mp4``
   - ``^cam(\d+)_``: matches ``01`` from ``cam01_scene.mp4``

CUDA out of memory
   **Problem:** The pipeline crashes with ``RuntimeError: CUDA out of memory``
   during plane sweep stereo.

   **Solution:** Reduce GPU memory usage by adjusting the following config keys::

       reconstruction:
         num_depth_hypotheses: 32   # Reduce from default (64)
         depth_batch_size: 4        # Reduce batch size

   Alternatively, run on CPU by passing ``--device cpu`` to ``aquamvs run``.
   CPU execution is slower but does not have memory constraints.

Depth maps mostly NaN
   **Problem:** The output depth maps are almost entirely NaN, and the fused
   point cloud is empty or very sparse.

   **Solution:** The specified depth range does not overlap with the actual
   scene geometry. Adjust ``depth_min`` and ``depth_max`` in the config to
   bracket your scene::

       reconstruction:
         depth_min: 0.5   # Minimum depth (meters, ray depth from camera)
         depth_max: 2.0   # Maximum depth (meters)

   To determine the correct range, run with a large range first (e.g., 0.1–5.0),
   then inspect the sparse cloud depth statistics::

       import open3d as o3d
       import numpy as np
       pcd = o3d.io.read_point_cloud("output/frame_000000/point_cloud/sparse.ply")
       pts = np.asarray(pcd.points)
       print(f"Z range: {pts[:,2].min():.3f} – {pts[:,2].max():.3f} m")

Empty point cloud after fusion
   **Problem:** ``point_cloud/fused.ply`` is empty or contains only a handful
   of points despite valid depth maps.

   **Solution:** The depth fusion step applies geometric consistency filtering,
   which may be too aggressive for your scene. Try increasing the consistency
   threshold in the config::

       reconstruction:
         consistency_threshold: 0.02   # Increase from default

   Also verify that the depth maps contain valid (non-NaN) data before fusion by
   checking the saved ``.npz`` files::

       import numpy as np
       d = np.load("output/frame_000000/depth_maps/{camera}.npz")
       valid = ~np.isnan(d["depth"])
       print(f"Valid pixels: {valid.sum()} / {valid.size}")

Notebook Issues
---------------

ModuleNotFoundError in Colab
   **Problem:** Notebook cells that import ``aquamvs`` or its dependencies fail
   with ``ModuleNotFoundError`` in Google Colab.

   **Solution:** Each Colab session starts from a clean environment. Add an
   installation cell at the top of the notebook::

       !pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
       !pip install git+https://github.com/cvg/LightGlue.git@edb2b83 -q
       !pip install git+https://github.com/tlancaster6/RoMaV2.git -q
       !pip install aquamvs -q

   After running the installation cell, restart the Colab runtime
   (*Runtime > Restart runtime*) and re-run all cells.

Dataset download fails
   **Problem:** The dataset download cell fails with a network error or the
   downloaded archive is corrupted.

   **Solution:** Verify network connectivity and try the download manually::

       wget https://zenodo.org/records/18702024/files/aquamvs-example-dataset.zip

   If the URL returns a 404 error, check the
   `GitHub Releases page <https://github.com/tlancaster6/AquaMVS/releases>`_
   for the current download URL — the release tag may have changed.

Open3D visualization fails headless
   **Problem:** Open3D ``draw_geometries`` raises an error or hangs in a
   headless environment (Colab, SSH session, CI).

   **Solution:** Use the ``OffscreenRenderer`` for non-interactive environments::

       import open3d as o3d

       def render_offscreen(geometry, width=800, height=600):
           renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
           renderer.scene.add_geometry("obj", geometry,
               o3d.visualization.rendering.MaterialRecord())
           img = renderer.render_to_image()
           return img

   In Colab, you can display the rendered image with ``IPython.display.Image``.

Configuration Issues
--------------------

Deprecated key warnings
   **Problem:** Running ``aquamvs run`` prints warnings such as::

       [WARNING] Deprecated key 'save_depth_maps' removed. Depth maps are now always saved.
       [WARNING] Deprecated key 'save_point_cloud' removed. Point clouds are now always saved.
       [WARNING] Deprecated key 'save_mesh' removed. Meshes are now always saved.

   **Solution:** Remove these keys from your ``config.yaml``. The corresponding
   outputs (depth maps, point clouds, meshes) are now unconditionally saved and
   cannot be disabled. The deprecated keys ``save_depth_maps``,
   ``save_point_cloud``, and ``save_mesh`` were removed in v1.1.

quality_preset in YAML has no effect
   **Problem:** Setting ``quality_preset: fast`` (or ``balanced`` / ``quality``)
   in ``config.yaml`` does not change reconstruction behavior.

   **Solution:** Quality presets are applied at configuration generation time,
   not at runtime. Use the ``--preset`` flag with ``aquamvs init`` to bake the
   preset into your config::

       aquamvs init --input-dir ... --preset fast      # Fewer depth planes, larger batches
       aquamvs init --input-dir ... --preset balanced   # Default tradeoffs (default)
       aquamvs init --input-dir ... --preset quality    # Maximum accuracy

   The generated ``config.yaml`` will contain the appropriate parameter values.
   Do not add ``quality_preset`` manually to an existing config file.
