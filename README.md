# Monocular Visual SLAM with Dynamic Object Removal and Bundle Adjustment

This project implements a monocular Visual SLAM (Simultaneous Localization and Mapping) system that uses ORB features for feature extraction, YOLO for dynamic object feature keypoint removal and bundle adjustment to refine camera poses and 3D points. In addition, loop closure detection is integrated using a bag-of-words (BoW) approach based on a pre-trained vocabulary.

## Overview

The SLAM pipeline in this project processes a sequence of images to:
- **Extract Features:** Detect ORB features in each frame.
- **Dynamic Keypoint Removal:** Remove keypoints from regions with dynamic objects using a YOLO-based detector.
- **Feature Matching & Pose Estimation:** Match features between consecutive frames and estimate relative camera motion using the essential matrix.
- **Triangulation:** Compute 3D points from matched features.
- **Loop Closure Detection:** Identify revisited areas via a BoW approach to reduce drift.
- **Bundle Adjustment:** Optimize the estimated camera poses and 3D points by minimizing the reprojection error.

## Features

- **ORB Feature Detection:** Extracts robust keypoints and descriptors.
- **YOLO-Based Dynamic Removal:** Eliminates features on potentially dynamic objects (classes such as person or car) to improve SLAM accuracy.
- **Pose Estimation:** Uses RANSAC and essential matrix recovery to compute relative rotations and translations.
- **3D Point Triangulation:** Recovers scene structure from matched keypoints.
- **Loop Closure:** Detects loop closures by comparing current descriptors with those stored in a BoW representation.
- **Bundle Adjustment:** Refines camera poses and 3D reconstructions using non-linear least squares optimization.

## Requirements

- **OpenCV:** For image processing and computer vision algorithms.
- **NumPy:** For numerical computations.
- **SciPy:** For optimization routines and linear algebra.
- **PyTorch:** For running the YOLO model.
- **Ultralytics YOLO:** For object detection.
- **dbow:** For bag-of-words representation and loop closure detection.
  
Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

_Note:_ The `dbow` package and the vocabulary file (`vocab.pkl`) must be available. Please generate a bow vocabulary based on the dataset for use in the code. If you do not have this vocabulary, please disable loop closure in the visual_slam class. The code will function more like visual odometry.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Dhrumil-Kotadia/Monocular_Visual_SLAM.git
   cd Monocular_Visual_SLAM
   ```

2. **Place Required Files:**

   - Ensure that `vocab.pkl` (the vocabulary for loop closure) is located in the repository root or update the file path in the code.
   - Download the YOLO model weights (e.g., `yolov8n.pt`) and place them in the same directory as the script (or adjust the path in the code). Ultralytics can also download the checkpoint if it is not present.
   - The `wrapper.py` script expects a sequence of images in PNG format stored in the folder `../image_0/` relative to the script location. Adjust the `Path` variable in the `main()` function if needed.

## Usage

Run the pipeline by executing the `wrapper.py` file:

```bash
python3 wrapper.py
```

The script will:
- Load the images from the specified directory.
- Process each frame through the SLAM pipeline.
- Save the estimated camera poses and 3D points in JSON files:
  - `Poses_temp.json` and `Points3D_temp.json` for initial estimates.
  - `Poses_ba.json` and `Points3D_ba.json` for the results after bundle adjustment.

## Pipeline Details

1. **Image Loading:**
   - The `Load_Images` function reads and sorts all PNG images from a given directory.

2. **Visual SLAM Class (`visual_slam`):**
   - **Initialization:** Sets up camera intrinsics, ORB detector, BFMatcher, and loads the YOLO model.
   - **Feature Extraction:** Uses ORB to extract keypoints and descriptors, then filters out dynamic keypoints based on YOLO detections.
   - **Feature Matching:** Matches features between consecutive frames using Hamming distance.
   - **Pose Estimation:** Computes the essential matrix and recovers the relative camera pose.
   - **Triangulation:** Converts 2D correspondences to 3D points.
   - **Loop Closure Detection:** Uses a bag-of-words approach (via `dbow`) to detect if the camera revisits a previously seen area.
   - **Bundle Adjustment:** Optimizes the estimated camera poses and 3D points by minimizing reprojection errors with a sparse least squares solver.

3. **Output:**
   - Final camera poses and 3D points are saved to JSON files for further analysis or visualization.

## Customization

- **Dynamic Object Removal:** You can toggle the removal of dynamic features by modifying the `remove_dynamic` flag in the `visual_slam` class.
- **Loop Closure Parameters:** Adjust thresholds in the `check_loop_closure` method to suit your environment and dataset.
- **Camera Intrinsics:** Update the intrinsic matrix `K` in the `main()` function according to your camera's calibration.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project draws inspiration from multiple open-source SLAM implementations and research in monocular visual SLAM. Special thanks to the developers and researchers in the computer vision community.
