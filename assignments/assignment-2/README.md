Assignment 2 — DS681

Date of submission: 2025-10-13

The goal of this assignment was to test if explainable AI can help make a car damage-detection system fairer.
Hertz uses AI to find scratches and dents on returned rental cars.
Customers said the system sometimes marked small marks as real damage.
We used an anomaly-detection model to check if heatmaps could help people understand why the AI made each decision.

Modules and Libraries Used:
Python 3.11.3
PyTorch Lightning
Anomalib library
Jupyter Notebook for training and testing
PGVector + PostgreSQL for similarity search

Method
Loaded the MVTec AD dataset using Anomalib.
Trained the PatchCore model on CPU.
Measured AUROC scores to check how well it finds defects.
Created heatmaps to show where the model found problems.
Stored features in PGVector to find similar images.

Results
Category    Image AUROC	Pixel AUROC
Tile	    0.98	    0.95
Leather	    0.97	    0.93
Grid	    0.96	    0.91
Average	    0.97	    0.93

The model correctly found most defects.
The heatmaps made its decisions easier to explain.