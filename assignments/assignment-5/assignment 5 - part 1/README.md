Assignment 4 — DS681

Date of submission: 11/26/2025
Task Completed: Task 1 — Commentary-Based Player Analysis

The goal of Task 1 is to build a text-based analysis system that can read basketball commentary and answer user questions about player performance. Instead of using computer vision, this task focuses on Natural Language Processing (NLP) to detect scoring events, identify players, and extract meaningful insights from the transcript of a real NBA game.

Dataset
For this assignment, we use the full broadcast transcript of:
Cleveland Cavaliers vs. Golden State Warriors – 2016 NBA Finals, Game 7
Video Link: https://www.youtube.com/watch?v=EoVTttvKfRs

The commentary transcript was exported to CSV and includes:
Timestamp (MM:SS)
Start time in seconds
Duration in seconds
Text commentary

Only the timestamp and commentary text columns are used in this task.

Modules and Libraries Used:
Python 3.11+
Jupyter Notebook (VS Code)
csv (for loading the transcript)
re (regular expressions)
dataclasses
typing
No machine learning libraries are required for Task 1

Method:
Transcript Loading
• Loaded the Game 7 commentary from a local CSV file.
• Extracted the timestamp and text fields into a structured TRANSCRIPT list.

Player Roster Setup
• Defined full player rosters for the 2016 Cavaliers and Warriors.
• Implemented case-insensitive matching so players can be detected by first name, last name, or full name.

Text Parsing and Scoring Detection
• Used regular expressions to detect common scoring actions (three-pointers, layups, dunks, free throws).
• Identified explicit commentary lines like “Curry has 12 points” to sync running totals.
• Logged events with timestamps for each player.

Player Analysis Engine
• Built a lightweight analysis system that:
    Maintains per-player scoring timelines
    Detects first field goals
    Captures notable commentary such as double-teams or “MVP” mentions
    Determines the leading scorer in the chosen segment

5. Query Interface
• Implemented a small question-answering function:
    “Analyze the player that scored the most in this game”
    “Analyze the player LeBron James”
• Returned summaries modeled after Google-style responses, including timestamps and contextual notes.

Results
• The system successfully identifies scoring actions from commentary text and tracks player totals throughout the chosen segment of Game 7.
• Generated clear natural-language summaries such as:
    “Stephen Curry is the top scorer in this segment with X points… First field goal at (TIMESTAMP)… Hits back-to-back threes….”
• The analysis reflects commentary, not official game statistics, which aligns with the assignment instructions.

Files
Task1_Basketball_Analyzer.ipynb — Complete notebook containing:
Transcript loader
Player roster
Text-parsing pipeline
Scoring detection logic
QA interface
Example queries and outputs:

[FULL GAME] Cavaliers vs Warriors 2016 Finals Game 7.csv
Local transcript file used for analysis.
