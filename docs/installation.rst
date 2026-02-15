Installation
============

This guide covers installing AquaMVS on Windows, Linux, and macOS.

Prerequisites
-------------

- **Python**: 3.10 or later
- **pip**: Latest version (upgrade with ``pip install --upgrade pip``)
- **git**: For installing git-based prerequisites

Install PyTorch
---------------

AquaMVS requires PyTorch. Visit the `PyTorch installation page <https://pytorch.org/get-started/locally/>`_
and use their configuration selector to get the correct install command for your system.

**GPU (CUDA 12.1) examples:**

.. code-block:: bash

   # Windows or Linux with NVIDIA GPU
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

**CPU-only examples:**

.. code-block:: bash

   # Windows, Linux, or macOS (CPU only)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Choose the command matching your OS and GPU from pytorch.org. For other CUDA versions
or ROCm (AMD GPU), consult the PyTorch website.

Install Git Prerequisites
--------------------------

AquaMVS depends on two libraries that are not available on PyPI and must be installed from git:

**Quick method** (recommended):

.. code-block:: bash

   pip install -r requirements-prereqs.txt

**Manual method:**

.. code-block:: bash

   pip install git+https://github.com/cvg/LightGlue.git@edb2b83
   pip install git+https://github.com/tlancaster6/RoMaV2.git@3862b19d5880cd7d690b544d27f30bb88e7d8fa4

**Why git dependencies?**

- **LightGlue**: Not yet published to PyPI by upstream maintainers
- **RoMa v2**: Uses a fork with a dataclasses metadata bugfix (PR submitted upstream)

Install AquaMVS
---------------

**From PyPI** (recommended for users):

.. code-block:: bash

   pip install aquamvs

**From source** (for development):

.. code-block:: bash

   git clone https://github.com/tlancaster6/AquaMVS.git
   cd AquaMVS
   pip install -e ".[dev]"

Platform-Specific Notes
------------------------

Windows
^^^^^^^

If you encounter build errors during Open3D installation, you may need to install
`Visual C++ Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_.
Select "Desktop development with C++" during installation.

Linux
^^^^^

Open3D requires OpenGL libraries for visualization. On Ubuntu/Debian:

.. code-block:: bash

   sudo apt install libgl1-mesa-glx

On headless servers or CI environments, Open3D's OffscreenRenderer may be unavailable.
AquaMVS degrades gracefully, skipping visualization steps when rendering is unavailable.

macOS
^^^^^

On Apple Silicon (M1/M2/M3), PyTorch supports the MPS (Metal Performance Shaders) backend
for GPU acceleration. Use the standard CPU/MPS install command from pytorch.org:

.. code-block:: bash

   pip install torch torchvision

Verify Installation
--------------------

Check that AquaMVS installed correctly:

.. code-block:: bash

   python -c "import aquamvs; print(aquamvs.__version__)"
   aquamvs --help

You should see version information and the CLI help text.

Troubleshooting
---------------

**"No module named 'torch'"**
   PyTorch must be installed before AquaMVS. See `Install PyTorch`_ above.

**"No module named 'lightglue'" or "No module named 'romav2'"**
   Git prerequisites must be installed before AquaMVS. See `Install Git Prerequisites`_ above.

**CUDA version mismatch**
   Your installed PyTorch CUDA version must match your NVIDIA driver. Check compatibility
   at https://pytorch.org/get-started/locally/. To check your installed PyTorch:

   .. code-block:: bash

      python -c "import torch; print(torch.__version__)"

   The output shows the CUDA version (e.g., ``2.1.0+cu121`` = CUDA 12.1).

**Open3D visualization errors on headless Linux**
   This is expected. AquaMVS automatically disables visualization when OffscreenRenderer
   is unavailable. Reconstruction still works; only debug visualizations are skipped.

**ImportError on Windows (DLL load failed)**
   This usually indicates missing Visual C++ runtime libraries. Install the
   `Visual C++ Redistributable <https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist>`_.
