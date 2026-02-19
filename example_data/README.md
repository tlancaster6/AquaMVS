# AquaMVS Example Dataset

This directory contains an example dataset for testing AquaMVS without preparing your own data.

## Contents

The example dataset includes:
- **1 frame per camera** from a 13-camera ring rig
- **Pre-made calibration JSON** generated using [AquaCal](https://github.com/tlancaster6/AquaCal)
- **Ready-to-run configuration file** with paths pre-configured

## Download

The example dataset is available for download from:

**[https://doi.org/10.5281/zenodo.18702024](https://doi.org/10.5281/zenodo.18702024)**

Download size: ~361 MB (13 cameras x 5 frames x 2 variants + calibration data)

## Installation

After downloading, extract to this directory. The structure should be:

```
example_data/
├── README.md              (this file)
├── config.yaml            (pre-configured pipeline config)
├── calibration.json       (AquaCal output)
└── images/
    ├── e3v82e0/
    │   └── frame_000000.png
    ├── e3v8213/
    │   └── frame_000000.png
    ├── e3v8220/
    │   └── frame_000000.png
    ├── e3v8223/
    │   └── frame_000000.png
    ├── e3v8224/
    │   └── frame_000000.png
    ├── e3v8227/
    │   └── frame_000000.png
    ├── e3v8229/
    │   └── frame_000000.png
    ├── e3v822b/
    │   └── frame_000000.png
    ├── e3v822d/
    │   └── frame_000000.png
    ├── e3v822e/
    │   └── frame_000000.png
    ├── e3v822f/
    │   └── frame_000000.png
    ├── e3v8232/
    │   └── frame_000000.png
    └── e3v8237/           (center auxiliary camera)
        └── frame_000000.png
```

## Usage

Once extracted, run the example pipeline:

```bash
cd example_data
aquamvs run config.yaml
```

Or from Python:

```python
from aquamvs import Pipeline

pipeline = Pipeline("example_data/config.yaml")
pipeline.run()
```

## Dataset Details

- **Cameras**: 12 standard-lens ring cameras + 1 auxiliary fisheye center camera
- **Ring radius**: 0.635 m
- **Water surface height**: ~0.978 m (Z-coordinate in world frame)
- **Refractive index**: 1.333 (fresh water at 20°C)
- **Image resolution**: Varies by camera model (typically 1920x1080 or similar)
- **Calibration method**: AquaCal refractive multi-camera calibration

## Calibration Source

The calibration was performed using [AquaCal](https://github.com/tlancaster6/AquaCal), which provides refractive multi-camera calibration with Snell's law modeling at the air-water interface. The `calibration.json` file contains:
- Intrinsic parameters (focal length, principal point, distortion)
- Extrinsic parameters (camera positions and orientations)
- Refractive parameters (water surface height, refractive index)
- Undistortion maps for each camera

See the [AquaCal documentation](https://github.com/tlancaster6/AquaCal) for details on the calibration process.

## License

This example dataset is provided under the [MIT License](../LICENSE) for demonstration and testing purposes.

**Note**: The underwater subject in these images may be subject to separate copyright or usage restrictions. This dataset is intended for academic and non-commercial testing of the AquaMVS pipeline only.
