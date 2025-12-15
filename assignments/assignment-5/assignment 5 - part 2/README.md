Assignment 5 — DS681

Date of submission: 12/14/2025
Task Completed: Task 2 — Video-Based Player Analysis

The goal of Task 2 is to verify basketball player actions by directly analyzing video footage. Unlike Task 1, which relies on commentary text, this task uses visual evidence from the broadcast to identify plays, track players and the ball, and confirm player actions such as shooting, passing, dribbling, rebounding, and defending.

Dataset
For this assignment, we use the full broadcast video of:
Cleveland Cavaliers vs. Golden State Warriors – 2016 NBA Finals, Game 7
Video Link: https://www.youtube.com/watch?v=EoVTttvKfRs

A short clip is extracted from the full game to reduce processing time while preserving meaningful gameplay. The video provides visual evidence for detecting players, tracking the basketball, and verifying actions within each play.

Modules and Libraries Used:
Python 3.11+
Jupyter Notebook (VS Code)
OpenCV
NumPy
Pandas
Matplotlib
Ultralytics YOLO (YOLOv8)
FFmpeg (video clipping)
yt-dlp (video download)

Method:
Task 2.1 — Chunking the Video into Plays
    - The basketball is detected in each frame using a pretrained YOLOv8 model.
    - Consecutive timestamps where the ball is visible are grouped together.
    - Gaps in ball visibility are used as boundaries between plays.
    - Each resulting time window represents a distinct play segment.

Task 2.2 — Analyzing Specific Player Actions
    - Players are detected and tracked using centroid-based tracking.
    - The player closest to the ball at each timestamp is inferred to be in possession.
    - Player actions are inferred using simple heuristics:
        - Dribbling / Ball Control: Player remains closest to the ball over consecutive frames
        - Passing: Change in ball ownership between tracked players.
        - Shooting: Ball leaves the vicinity of the player near the basket area.
        - Assisting: Rapid ball movement followed by a shot within the same play.
            - Each identified action is logged with a timestamp and associated player track ID.

Bird’s-Eye View Generation
    - A homography transform is computed using manually selected court corner points.
    - Player and ball positions are projected onto a top-down court representation.
    - Bird’s-eye view images are saved as visual evidence for each detected action.

Results
    - The system successfully segments the video into individual plays using ball activity.
    - Player actions such as dribbling, passing, and shooting are detected within each play.
    - Bird’s-eye visualizations clearly show player spacing and ball location at key moments.
    - The generated action table provides timestamped evidence supporting commentary-based analysis from Task 1.

Files
Task2_Video_Action_Verification.ipynb
    - Complete notebook for video-based analysis, play chunking, player tracking, and action detection.

Output Files:
game7.mp4 / clip.mp4
    - Full game video and extracted analysis clip.

task2_actions_with_birdseye.csv
    - Table of detected player actions with timestamps and bird’s-eye view image paths.

birdseye/
    - Folder containing top-down court visualizations for selected action timestamps.


**Note**
Because basketball detection from broadcast footage is sparse, initial play segmentation produced very short intervals. To obtain meaningful plays, nearby detections were merged and expanded into short temporal windows. This preserves video-based evidence while enabling reliable action verification.