Take-Home Midterm (Camera Calibration) — DS681

Date of submission: 2025-10-27

The goal of this assignment was to implement a camera calibration pipeline from scratch using Zhang’s method.
The project estimates the intrinsic and extrinsic parameters of a camera from chessboard images and produces accurate real-world measurements.
No high-level computer-vision APIs (like OpenCV’s calibrateCamera) were used.
All steps were coded manually using NumPy, SciPy, and PyTorch.

Modules and Libraries Used:
Python 3.11
NumPy + Pandas
SciPy (gaussian filters and Sobel operators)
Matplotlib (for image previews and visualization)
Pillow (for image I/O)
Hugging Face Hub (for downloading the dataset)
PyTorch (for non-linear optimization and refinement)

Method:
Dataset loading: 
Extracted a Hugging Face Parquet dataset of chessboard images into local .jpg files.

Semi-automatic corner annotation:
Displayed each image preview.
User typed three corner points (Top-Left, Top-Right, Bottom-Left).
A Harris corner detector with sub-pixel refinement found precise grid points.
Homography estimation (DLT): Computed a homography for each view from 2D ↔ 3D correspondences.
Intrinsic calibration: Solved Zhang’s linear equations to derive camera matrix K.
Extrinsic calibration: Recovered rotation and translation for each view.
Non-linear refinement: Used PyTorch stochastic gradient descent to minimize re-projection error.
Validation: Compared results against OpenCV baseline parameters for consistency.

Results:
Parameter	            Description	                    Example Value
fx, fy	                focal lengths in pixels	        ≈ 1150 ± 5 px
cx, cy	                principal point offset	        ≈ (960, 540) px
avg reprojection error	(pixel MSE after refinement)	< 0.5 px

The reprojection errors remained below one pixel, confirming accurate intrinsics and extrinsics.
The approach worked even without any GUI backends by using saved image previews and manual coordinate input.

Output:
corner_annotations.parquet – refined corner coordinates and world positions
_previews/ – saved preview images for manual annotation

Summary
This assignment demonstrated a complete camera calibration workflow from first principles:
data collection to corner annotation to DLT to Zhang’s formulation to non-linear optimization.
It produces precise camera parameters without relying on OpenCV’s built-in calibration functions.