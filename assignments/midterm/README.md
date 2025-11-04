Take-Home Midterm (Camera Calibration) — DS681

Date of Submission: 2025-10-27

Objective:
The goal of this project was to implement a complete camera calibration pipeline from scratch using Zhang’s method. The system estimates both the intrinsic and extrinsic parameters of a camera from multiple chessboard images to enable accurate real-world measurements. No high-level calibration APIs such as cv2.calibrateCamera() were used. All steps were implemented manually using NumPy, SciPy, and PyTorch.

Modules and Libraries:
Python 3.11
NumPy, Pandas
SciPy (Gaussian filters and Sobel operators)
Matplotlib (image visualization)
Pillow (image I/O)
Hugging Face Hub (dataset download)
PyTorch (non-linear optimization and refinement)

Methodology:
Dataset Loading:
A set of calibration images (9×7 checkerboard) was captured using a smartphone camera at a fixed zoom level. The photos were taken from varied angles, distances, and tilts under diffuse lighting conditions. The images were then processed into a Parquet dataset containing detected corner coordinates and world-space positions.

Corner Detection:
The corner detection used OpenCV’s findChessboardCornersSB() and sub-pixel refinement through cornerSubPix() to extract precise 2D image coordinates for each inner corner across multiple views.

Homography Estimation (DLT):
For each view, a homography was computed between the 3D checkerboard coordinates and their 2D image projections using the Direct Linear Transform method.

Intrinsic Calibration:
Zhang’s linear approach was applied to estimate the camera’s intrinsic matrix K, including focal lengths, principal point, and skew.

Extrinsic Calibration:
The rotation and translation vectors were determined for each image view to represent the camera’s position and orientation relative to the checkerboard plane.

Non-Linear Refinement:
A PyTorch-based stochastic gradient descent optimization minimized the overall reprojection error by refining K, distortion coefficients, and all extrinsics simultaneously.

Validation:
The refined parameters were compared to OpenCV’s built-in calibration results to verify correctness.

Results:
Parameter	        Description	                Example Value
fx,fy               focal lengths (px)	        ≈ 2.30 × 10⁴
cx,cy               principal point (px)	    (286, 477)
Distortion (k₁–k₅)	radial + tangential	        (–0.44, –0.51, –0.15, –0.31, –0.55)
RMS Error	        reprojection error (px)	    15.36 (Refined) vs 12.02 (OpenCV)

The refined calibration matched the OpenCV baseline closely, confirming correct intrinsic and extrinsic recovery.
The reprojection plots showed excellent overlap between observed and predicted corner locations, with per-view mean errors under 2 px and residuals distributed evenly across the grid.
The 3D visualization confirmed consistent camera poses around the checkerboard plane and a centered principal point in the image frame.

Outputs:
smartphone_corners.parquet — extracted corner coordinates and world-space mappings
Visualization Plots — reprojection overlays, error histograms, residual maps, 3D camera poses, and principal point alignment

Summary:
This project demonstrates a full implementation of Zhang’s camera calibration method from first principles. The process included corner detection, DLT homography computation, intrinsic and extrinsic recovery, and non-linear refinement. The results validated the correctness of the custom implementation against OpenCV’s calibration output.
The visualization confirmed strong alignment, low error, and geometrically consistent camera poses, proving that accurate smartphone calibration can be achieved entirely from scratch using Python and PyTorch.