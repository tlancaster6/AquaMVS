Command-Line Interface
======================

AquaMVS provides a command-line interface for common operations.

.. code-block:: bash

   aquamvs --help

Commands
--------

``aquamvs init``
~~~~~~~~~~~~~~~~
Generate a pipeline configuration from a video directory and calibration file.

.. code-block:: bash

   aquamvs init --video-dir /path/to/videos --pattern "^([a-z0-9]+)-" \
                --calibration /path/to/calibration.json \
                --output-dir /path/to/output

``aquamvs run``
~~~~~~~~~~~~~~~
Run the reconstruction pipeline.

.. code-block:: bash

   aquamvs run config.yaml

``aquamvs export-refs``
~~~~~~~~~~~~~~~~~~~~~~~
Export undistorted reference images for ROI mask creation.

.. code-block:: bash

   aquamvs export-refs config.yaml --frame 0

``aquamvs benchmark``
~~~~~~~~~~~~~~~~~~~~~
Run benchmark comparison across extractor configurations.

.. code-block:: bash

   aquamvs benchmark config.yaml --frame 0
