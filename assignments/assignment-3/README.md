Assignment 3 — DS681

Date of submission: 2025-11-02

The goal of this assignment is to implement a car damage identification system that segments dents, derives bounding boxes, and tracks/counts dents in video.

Dataset
We use the Hugging Face dataset `harpreetsahota/CarDD`, which has a single column of images. For pipeline validation, the notebooks create placeholder masks. Replace with your binary masks for meaningful metrics.

Modules and Libraries Used:
Python 3.11+
PyTorch (torch, torchvision)
timm (DINOv3 / ViT or ResNet backbone features)
Hugging Face `datasets`
OpenCV, NumPy, Matplotlib, scikit-image
Jupyter Notebook for training, inference, and tracking

Method
• Loaded the CarDD dataset with `datasets.load_dataset` and explored samples (Part 1).
• Built a lightweight FPN decoder head on a pretrained backbone (ResNet-50) and trained only the segmentation head (Part 2).
• Computed per-image predictions, thresholded masks, and derived bounding boxes for visualization (Part 3).
• Implemented a Deep SORT–style tracker (IoU-based matching) to track dent centroids and show a live dent count on a video overlay (Part 4).

Results
Image IoU: ~0.05–0.225
Pixel Dice: ~0.05–0.30
Video overlay: tracks show stable IDs when segmentation yields distinct components.

Files
assignment 3 - part 1/01_dataset_explore.ipynb
assignment 3 - part 2/02_train_segmentation.ipynb
assignment 3 - part 3/03_infer_images.ipynb
assignment 3 - part 4/04_infer_video_tracking.ipynb
