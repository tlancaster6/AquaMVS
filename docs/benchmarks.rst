Benchmark Results
=================

Overview
--------

AquaMVS includes a comprehensive benchmark suite for evaluating reconstruction
accuracy, performance, and regression detection. The benchmark system supports:

- **Synthetic scenes** with analytic ground truth for accuracy validation
- **Real-world scenes** with measured ground truth (if available)
- **Quality presets** (FAST, BALANCED, QUALITY) comparison
- **Feature extractor** comparison (SuperPoint, LightGlue, RoMa, ALIKED, DISK)
- **Regression detection** with configurable thresholds

The benchmark suite is designed for both:

1. **Local development**: Full accuracy benchmarks with visualization
2. **CI/CD**: Fast synthetic-only regression tests (< 1 minute)

Running Benchmarks
------------------

Local Benchmarks
~~~~~~~~~~~~~~~~

Create a benchmark configuration YAML:

.. code-block:: yaml

    output_dir: ".benchmarks/my_run"

    tests:
      - name: "flat_plane"
        scene:
          type: "flat_plane"
          depth_z: 1.2
          bounds: [-0.3, 0.3, -0.3, 0.3]
          resolution: 0.005

    extractor_configs:
      - name: "lightglue_sparse"
        sparse_matching:
          extractor: "superpoint"
          matcher: "lightglue"
        reconstruction:
          dense_matching: false

      - name: "roma_dense"
        sparse_matching:
          extractor: "superpoint"
          matcher: "lightglue"
        reconstruction:
          dense_matching: true
          dense_matcher: "roma"

Run the benchmark:

.. code-block:: bash

    aquamvs benchmark benchmark_config.yaml

Generate visualization plots:

.. code-block:: bash

    aquamvs benchmark benchmark_config.yaml --visualize

Compare two runs:

.. code-block:: bash

    aquamvs benchmark benchmark_config.yaml --compare .benchmarks/run1 .benchmarks/run2

CI Benchmarks
~~~~~~~~~~~~~

The CI benchmark suite runs automatically on every commit. It uses:

- Small synthetic scenes (20cm × 20cm patches)
- Reduced image sizes (64×64 or 128×128)
- Few depth hypotheses (16-32)
- Unit tests for core functions

Target runtime: **< 60 seconds** for entire suite.

Run locally:

.. code-block:: bash

    pytest tests/benchmarks/ -m benchmark -v

Interpreting Results
--------------------

Accuracy Metrics
~~~~~~~~~~~~~~~~

**Completeness** (%)
    Percentage of ground truth surface covered by reconstruction.

    - **> 90%**: Excellent coverage
    - **60-90%**: Good coverage with some gaps
    - **< 60%**: Poor coverage, significant missing regions

**Geometric Error** (mm)
    Distance between reconstructed points and ground truth surface.

    - **Median < 2mm**: High accuracy
    - **Median 2-5mm**: Good accuracy
    - **Median > 5mm**: Poor accuracy or systematic bias

**Mean vs Median Error**
    Median is more robust to outliers. Large mean/median ratio indicates
    outlier presence (common at scene boundaries).

Performance Metrics
~~~~~~~~~~~~~~~~~~~

**Runtime** (seconds)
    Total wall-clock time for reconstruction pipeline.

**Frames per second** (fps)
    Throughput metric (higher is better).

**Peak memory** (MB)
    Maximum GPU memory usage during reconstruction.

Quality Presets
---------------

AquaMVS provides three quality presets that trade off speed and accuracy:

FAST Preset
~~~~~~~~~~~

**Target use case**: Real-time or near-real-time reconstruction

**Parameters:**

- ``num_depths: 64``
- ``window_size: 7``
- ``depth_batch_size: 8``
- ``max_keypoints: 1024``
- ``voxel_size: 0.002`` (2mm)
- ``poisson_depth: 8``

**Expected performance:**

- Speed: ~5-10 fps (hardware dependent)
- Accuracy: Median error ~3-5mm
- Completeness: ~70-80%

BALANCED Preset (Default)
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Target use case**: General-purpose reconstruction with good speed/accuracy tradeoff

**Parameters:**

- ``num_depths: 128``
- ``window_size: 11``
- ``depth_batch_size: 4``
- ``max_keypoints: 2048``
- ``voxel_size: 0.001`` (1mm)
- ``poisson_depth: 9``

**Expected performance:**

- Speed: ~2-5 fps
- Accuracy: Median error ~2-3mm
- Completeness: ~80-90%

QUALITY Preset
~~~~~~~~~~~~~~

**Target use case**: High-accuracy offline reconstruction

**Parameters:**

- ``num_depths: 256``
- ``window_size: 15``
- ``depth_batch_size: 1``
- ``max_keypoints: 4096``
- ``voxel_size: 0.0005`` (0.5mm)
- ``poisson_depth: 10``

**Expected performance:**

- Speed: ~0.5-2 fps
- Accuracy: Median error ~1-2mm
- Completeness: ~85-95%

**Configuration:**

.. code-block:: yaml

    quality_preset: "quality"  # or "fast" or "balanced"

Profiling
---------

Identify pipeline bottlenecks using the profiling command:

.. code-block:: bash

    aquamvs profile config.yaml --frame 0

Sample output:

.. code-block:: text

    Profile Report (device: cuda)
    Total time: 1234.56 ms
    Peak memory: 512.34 MB

    +-----------------------+----------+-----------+------------+-------------+
    | Stage                 | CPU (ms) | CUDA (ms) | Total (ms) | Memory (MB) |
    +=======================+==========+===========+============+=============+
    | dense_matching        | 12.34    | 456.78    | 469.12     | 256.45      |
    | depth_estimation      | 8.90     | 234.56    | 243.46     | 128.23      |
    | sparse_matching       | 45.67    | 89.12     | 134.79     | 64.12       |
    +-----------------------+----------+-----------+------------+-------------+

    Top 3 Bottlenecks:
      1. dense_matching: 469.12 ms, 256.45 MB
      2. depth_estimation: 243.46 ms, 128.23 MB
      3. sparse_matching: 134.79 ms, 64.12 MB

Export Chrome trace for visualization in chrome://tracing:

.. code-block:: bash

    aquamvs profile config.yaml --output-dir ./profiling_results

**Note:** Full profiling integration with the pipeline is pending. Current
implementation demonstrates the profiling infrastructure and API.

Baseline Results
----------------

**Placeholder for published baseline results**

This section will be populated with baseline performance metrics from the
reference hardware configuration after the initial benchmark runs are published.

Expected baseline results table format:

.. list-table:: Baseline Performance (Reference Hardware)
   :header-rows: 1
   :widths: 20 15 15 15 15 20

   * - Configuration
     - Completeness (%)
     - Median Error (mm)
     - Runtime (s)
     - FPS
     - Peak Memory (MB)
   * - FAST + SuperPoint
     - TBD
     - TBD
     - TBD
     - TBD
     - TBD
   * - BALANCED + LightGlue
     - TBD
     - TBD
     - TBD
     - TBD
     - TBD
   * - QUALITY + RoMa
     - TBD
     - TBD
     - TBD
     - TBD
     - TBD

**Reference Hardware:**

- GPU: NVIDIA RTX 3090 (24GB)
- CPU: Intel Core i9-12900K
- RAM: 64GB DDR5
- PyTorch: 2.0+
- CUDA: 11.8+

Regression Detection
--------------------

The benchmark system supports automated regression detection with configurable
thresholds:

**Default Thresholds:**

- **Accuracy metrics** (completeness, error): ±5% change triggers warning
- **Performance metrics** (runtime, memory): +10% degradation triggers warning

**Comparison Workflow:**

1. Establish baseline:

   .. code-block:: bash

       aquamvs benchmark config.yaml
       # Results saved to .benchmarks/run_20260215_143022/

2. Make changes to code

3. Run benchmark again:

   .. code-block:: bash

       aquamvs benchmark config.yaml
       # Results saved to .benchmarks/run_20260215_150134/

4. Compare runs:

   .. code-block:: bash

       aquamvs benchmark config.yaml --compare \
           .benchmarks/run_20260215_143022 \
           .benchmarks/run_20260215_150134

Sample comparison output:

.. code-block:: text

    Benchmark Comparison
    ====================

    Baseline: .benchmarks/run_20260215_143022
    Current:  .benchmarks/run_20260215_150134

    +-----------------------+----------+---------+--------+------------+
    | Metric                | Baseline | Current | Delta  | Change (%) |
    +=======================+==========+=========+========+============+
    | completeness          | 0.85     | 0.83    | -0.02  | -2.4%      |
    | median_error_mm       | 2.34     | 2.45    | +0.11  | +4.7%      |
    | runtime_s             | 12.5     | 14.2    | +1.7   | +13.6% ⚠   |
    +-----------------------+----------+---------+--------+------------+

    Regressions Detected: 1
      - runtime_s: +13.6% (threshold: 10%)

**Integration with CI:**

The CI benchmark suite runs on every commit as an advisory check
(``continue-on-error: true``). Failures do not block merges but should be
investigated.

Storage
-------

Benchmark results are stored in ``.benchmarks/`` directory (gitignored):

.. code-block:: text

    .benchmarks/
    ├── run_20260215_143022/
    │   ├── summary.json
    │   ├── config.yaml
    │   ├── plots/
    │   │   ├── accuracy_bars.png
    │   │   ├── timing_bars.png
    │   │   └── error_heatmap.png
    │   └── point_clouds/
    └── run_20260215_150134/
        └── ...

**Summary JSON** contains all metrics and metadata for comparison.

**Plots** are generated with ``--visualize`` flag.

**Point clouds** are saved for visual inspection (when enabled in config).
