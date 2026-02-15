Tutorials
=========

This section provides hands-on tutorials for AquaMVS reconstruction workflows.

End-to-End Reconstruction
-------------------------

Walk through a complete reconstruction from synchronized camera images to a 3D surface mesh using the Python API.

:download:`Download Jupyter Notebook <notebook.ipynb>`

The notebook demonstrates:

- Loading and inspecting a pipeline configuration
- Running the reconstruction pipeline with the ``Pipeline`` class
- Examining intermediate outputs (depth maps, consistency maps)
- Visualizing the fused point cloud
- Exporting meshes to various formats (OBJ, STL, GLB)

See also the :doc:`CLI Guide </cli_guide>` for command-line workflow.

Quick Start
-----------

For a minimal working example:

.. code-block:: python

   from aquamvs import Pipeline, PipelineConfig

   # Load configuration from YAML
   config = PipelineConfig.from_yaml("config.yaml")

   # Run reconstruction
   pipeline = Pipeline(config)
   pipeline.run()

   # Outputs saved to config.output_dir

Configuration
-------------

The :class:`~aquamvs.config.PipelineConfig` class provides extensive customization:

- **Matcher selection**: ``sparse_matching.matcher_type`` (``"lightglue"`` or ``"roma"``)
- **Pipeline mode**: ``reconstruction.pipeline_mode`` (``"full"`` for dense stereo, ``"sparse"`` for sparse reconstruction)
- **Depth range**: ``reconstruction.depth_min`` and ``depth_max``
- **Device**: ``runtime.device`` (``"cpu"`` or ``"cuda"``)
- **Output control**: ``runtime.save_*`` flags to enable/disable intermediate outputs

See :doc:`API Reference </api/config>` for full configuration options.
