# Mobilint Demo Packaging Guide

This package script builds required third-party dependencies and creates a distributable archive.

## 1) Prerequisites

- Linux environment
- `make`, `cmake`, `gcc/g++`
- OpenCV installed in your environment
- Root privileges (for required package operations)

## 2) Build the package

```bash
./package/package.sh aries2-v4 aries2
```

The first argument is `PRODUCT`, and the second is `DRIVER_TYPE`.

## 3) Include assets

The package includes:

- `mxq` files (LFS-managed)
- `rc` configuration files
- `src`
- Package metadata/README

Make sure required LFS assets are pulled before packaging:

```bash
git lfs install
git lfs pull
```

## 4) Configuration notes

### Feeder setting (`rc/FeederSetting.yaml`)

`FEEDER_TYPE: { CAMERA | VIDEO | IPCAMERA | YOUTUBE }`

### Model setting (`rc/ModelSetting.yaml`)

`MODEL_TYPE: { SSD | STYLENET | FACE | POSE | OBJECT | SEGMENTATION }`

## 5) Keyboard shortcuts (runtime)

| Shortcut | Description          |
| ---      | ---                  |
| D, d     | Display FPS          |
| M, m     | Maximize Screen      |
| C, c     | Clear All Worker     |
| F, f     | Fill All Worker      |
| T, t     | Display Running Time |
| Q, q     | Quit                 |
