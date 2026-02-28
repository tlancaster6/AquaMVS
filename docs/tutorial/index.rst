Tutorials
=========

AquaMVS provides two ways to run reconstruction: a command-line interface (CLI) for
batch processing and scripted workflows, and a Python API for programmatic control
and integration into custom pipelines.

Choose Your Workflow
--------------------

**Command-Line Interface (CLI)**
   Best for: Running reconstructions on new datasets, batch processing, quick experiments.
   Start with the :doc:`CLI Guide </cli_guide>`.

   .. code-block:: bash

      aquamvs init --input-dir ./videos --calibration calibration.json --output-dir ./output
      aquamvs run config.yaml

**Python API**
   Best for: Custom workflows, programmatic access, integration with other tools.
   Start with the :doc:`End-to-End Tutorial <notebook>`.

   .. code-block:: python

      from aquamvs import Pipeline, PipelineConfig
      config = PipelineConfig.from_yaml("config.yaml")
      Pipeline(config).run()

.. toctree::
   :maxdepth: 2
   :hidden:

   notebook
