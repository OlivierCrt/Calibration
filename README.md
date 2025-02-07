# Perception Calibration

## Description

This repository contains code for performing monocular and stereo camera calibration using Python and OpenCV. It includes tasks such as detecting calibration patterns, calculating intrinsic and extrinsic camera parameters, and rectifying images for stereo vision. The goal is to enable 3D reconstruction from a pair of stereo images.

## Features

- **Monocular Calibration**: Calibrate a single camera, detect calibration patterns, and calculate intrinsic parameters.
- **Stereo Calibration**: Calibrate two cameras together, calculate their relative position, and rectify images to align epipolar lines.
- **3D Reconstruction**: Use disparity maps to create depth maps and 3D point clouds from stereo images.

## Setup

1. Clone the repository:
    ```bash
    git clone <https://github.com/OlivierCrt/Calibration>
    ```

2. Create and activate a Python environment:
    ```bash
    conda create -n Calibration python=3.7
    conda activate Calibration
    ```

3. Install dependencies:
    ```bash
    conda install scikit-learn numpy matplotlib
    conda install -c conda-forge opencv
    ```

4. Run the test script to check setup:
    ```bash
    python test.py
    ```

## Usage

- Run `CalibrationMono.py` for monocular calibration.
- Run `CalibrationStereo.py` for stereo calibration and rectification.

