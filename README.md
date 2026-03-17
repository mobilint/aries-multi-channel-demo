# ARIES Multi-Channel Demo

ARIES Multi-Channel Demo is a sample application for running multi-channel video inference workflows on MLA100-based systems.

## 1) Overview

This project supports:

- Linux and Windows build/run workflow
- MLA100-based hardware inference
- Multiple feeder types (camera/RTSP/YouTube/video file)
- Multiple model types (SSD / STYLENET / FACE / POSE / OBJECT / SEGMENTATION)
- Layout and worker configuration through YAML files

## 2) Prerequisites (single source)

### Linux

- Ubuntu 20.04 LTS or newer
- Git and Git LFS
- `build-essential`, `cmake`
- `libopencv-dev`
- Mobilint runtime packages:
  - `mobilint-aries-driver`
  - `mobilint-qb-runtime`
  - `mobilint-cli`
- Optional: `python3`, `pip`, and `yt-dlp` (required for YouTube feeder)

### Windows

- Windows 10/11
- Visual Studio 2022
- CMake
- OpenCV 4.x (installed in your environment)
- MLA100 runtime and driver support

## 3) Linux setup and run flow

1. Install required system packages:

```bash
sudo apt update
sudo apt install -y libopencv-dev mobilint-qb-runtime python3-pip
```

2. Get LFS assets (`.mxq` and sample `.mp4`):

```bash
git lfs install
git lfs pull
```

3. Build/update and register desktop shortcut:

```bash
./update.sh
```

4. Run the demo:

```bash
./run.sh
```

or directly:

```bash
cd build && ./src/demo/demo
```

If `run.sh` prints `Demo binary not found`, run `./update.sh` first.

## 4) Build options

### Build manually (Linux)

```bash
mkdir build
cd build
cmake [-DQBRUNTIME_PATH=<path>] ..
make -j
./src/demo/demo
```

`cmake` and other build dependencies are required. If CMake cannot be installed via package manager, install it from kitware:

https://apt.kitware.com/

### Packaging

```bash
./package/package.sh aries2-v4 aries2
```

## 5) Required model files in this repository

The current example configuration uses the following `mxq` files:

- `../mxq/ssd.mxq`
- `../mxq/yolo11s-face.mxq`
- `../mxq/yolo26s-seg.mxq`
- `../mxq/yolo26s-pose.mxq`
- `../mxq/yolo26s.mxq`

All `.mxq` and sample `.mp4` assets are managed by Git LFS.

## 6) Feeder setting (`rc/FeederSetting.yaml`)

`FEEDER_TYPE: { CAMERA | IPCAMERA | YOUTUBE | VIDEO }`

```yaml
- feeder_type: FEEDER_TYPE
  src_path:
    - <path or url>
- feeder_type: ...
```

| feeder_type | src_path                   | Description                                  |
|-------------|----------------------------|----------------------------------------------|
| CAMERA      | Camera index               | Enter camera index as a number                |
| IPCAMERA    | RTSP URL                   | Enter RTSP URL                               |
| YOUTUBE     | YouTube URLs               | Enter one or more URLs                        |
| VIDEO       | Video file paths           | Enter one or more local video file paths       |

## 7) Model setting (`rc/ModelSetting.yaml`)

`MODEL_TYPE: { SSD | STYLENET | FACE | POSE | OBJECT | SEGMENTATION }`

`CLUSTER: { Cluster0 | Cluster1 }`

`CORE: { Core0 | Core1 | Core2 | Core3 }`

`subconfig` is required for `FACE`, `POSE`, `OBJECT`, and `SEGMENTATION`.

```yaml
- model_type: MODEL_TYPE
  mxq_path: <path to mxq>
  dev_no: <board index>
  core_id:
    - cluster: CLUSTER
      core: CORE
  # optional: num_core: INTEGER
- ...
```

For `SSD` and `STYLENET`, `subconfig` is not required.

## 8) Layout setting (`rc/LayoutSetting*.yaml`)

```yaml
image_layout:
  - path: /path/to/image
    roi: [x, y, w, h]
feeder_layout:
  - [x, y, w, h]
worker_layout:
  - {feeder_index: <index>, model_index: <index>, roi: [x, y, w, h]}
```

## 9) Keyboard shortcuts

| Key | Action            |
| --- | ----------------- |
| D, d | Display FPS       |
| T, t | Display Time      |
| M, m | Maximize Screen   |
| C, c | Clear All Workers |
| F, f | Fill All Workers  |
| Q, q | Quit              |
| Esc  | Quit              |

## Windows

### OpenCV installation

Download OpenCV 4.x installer from official releases:

https://github.com/opencv/opencv/releases

### Build commands

```bash
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -DCMAKE_PREFIX_PATH="C:\path\to\opencv\build" -DQBRUNTIME_PATH="C:\path\to\qbruntime" ..
```

Then open the generated `.sln` file and build.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

## Third-Party Notices

This project depends on external components such as OpenCV, Mobilint runtime/CLI packages, and optional `yt-dlp`.  
See [NOTICE](NOTICE.md) for dependency licenses and notices.
