Computer Using Agent (CUA) — DS681 Project

Date of submission: 12/11/2025
Task Completed: Full CUA System — Vision + LLM Reasoning + WebRTC Screen Streaming + MongoDB Logging

The goal of this project is to build a Computer Using Agent (CUA) capable of watching a user’s screen, interpreting on-screen academic papers, and answering questions about highlighted text, tables, figures, and diagrams. Unlike typical NLP tasks, this system relies entirely on visual understanding. The CUA reads PDFs only through Vision-Language Models (VLMs) and uses a local Large Language Model (LLM) for reasoning. All interactions are logged into a database for retrieval and analysis.

---------------------------------------------------------------------------------------------

Dataset / Input Source

There is no traditional dataset.
Instead, the user streams their live screen using the browser’s getDisplayMedia API. The agent processes captured frames to extract text, tables, diagrams, and highlighted sections.

The main reference paper used during development and testing was:
EfficientViT: Enhanced Vision Transformers for High-Resolution Recognition Tasks
arXiv PDF: https://arxiv.org/pdf/2205.14756

Since the assignment constraints prohibit local text extraction, the PDF is viewed on-screen and processed strictly through images.

----------------------------------------------------------------------------------------------

System Architecture Overview

The CUA is composed of the following components:

1. Web Browser (User Interface)
Displays an academic paper from arXiv.
Allows screen sharing via WebRTC (navigator.mediaDevices.getDisplayMedia()).
Streams screen content to the agent.

2. WebRTC / HTTP Frame Transport
Two transport modes were implemented:
    - WebRTC PeerConnection (required by assignment; may fail depending on VS Code port forwarding)
    - HTTP-based Frame Upload Loop (robust fallback)

Every 1 second:
    - The browser draws the screen to a <canvas>
    - Sends frame.png to the agent's /frame endpoint
This produces a live stream of frames/latest.png for the VLM.

3. Vision-Language Model (VLM)

Model: DeepSeek OCR / Qwen-VL (local)

Responsibilities:
    - Extract page text from the screenshot
    - Identify tables, figures, references, equations
    - Detect highlighted text (yellow or purple)
    - Provide structured descriptions of the visible paper content

4. Reasoning LLM
Model: Qwen2.5 (running locally via Ollama)

Receives:
    - OCR output
    - Web search metadata
    - User question

Provides:
    - Tutorials on highlighted text
    - Explanations of figures and tables
    - Identification of important sections
    - Ablation analysis

5. MongoDB Logging
A local MongoDB container stores:
    - Extracted text from each frame
    - Highlighted regions
    - User questions
    - Agent answers
    - Referenced papers and citations

This forms a full "interaction memory" for the CUA.

----------------------------------------------------------------------------------------

Modules and Libraries Used:
Python Environment
Python 3.11+
VS Code Dev Container + Docker
Jupyter Notebooks (.ipynb) for experimentation

Key Libraries:
    - fastapi (web server)
    - uvicorn (ASGI runtime)
    - aiortc (WebRTC signaling)
    - opencv-python (frame decoding)
    - pillow (image manipulation)
    - transformers (VLM + tokenizer interfaces)
    - pydantic-ai (agent framework)
    - requests (web search APIs + LLM calls)
    - pymongo (MongoDB driver)
    - gradio (demo UI)

External Local Services:
    - MongoDB (Docker container)
    - Ollama running Qwen2.5 LLM
    - Local VLM weights (DeepSeek / Qwen-VL)

------------------------------------------------------------------------------

Method
1. Frame Capture Pipeline:
    - Browser streams the PDF page.
    - Canvas captures frames.
    - HTTP POST uploads frames to /frame.
    - Server writes to project/frames/latest.png.

This file is always the most recent view of the user’s screen.

2. OCR + VLM Processing

The notebook loads:
FRAME_PATH = "/workspaces/eng-ai-agents/project/frames/latest.png"


Steps:
    - Read the PNG.
    - Preprocess and crop if needed.
    - Pass through the VLM.
    
VLM outputs:
    - Plain text
    - Detected highlights
    - Table regions
    - Figure captions
    - Layout structure

3. Reasoning LLM (Qwen2.5)

The OCR output and user question are merged into a structured prompt. Qwen2.5 is queried locally through:
POST http://host.docker.internal:11434/api/chat

The LLM produces high-level explanations, tutorials, and reasoning.

4. Web Search Integration

Using Semantic Scholar API:
    - Identify referenced authors
    - Retrieve metadata for citations like "[Yuan et al., 2022]"
    - Provide summarized related-work insights

5. MongoDB Storage

Each event stores:
    - Timestamp
    - Raw OCR text
    - Cleaned text
    - User question
    - LLM answer
    - Figures or tables detected
    - Highlights detected (yellow vs purple)

This allows replayable interactions and searchable logs.

6. Gradio User Interface

The demo app provides:
    - Upload or live-frame viewing
    - “Explain highlight”
    - “Auto-mark important sections”
    - “Explain this table”
    - “Identify key ablation findings”
    - “Generate Mermaid diagram”

-----------------------------------------------------------------------------------

Results

The CUA system successfully captures live screen images, extracts text and structural information using a Vision-Language Model, and generates clear explanations through a local reasoning LLM. It identifies highlighted regions, interprets tables and figures, and produces concise tutorial-style explanations for selected text. The agent also recognizes important sections of the paper and explains why they matter, including summarizing ablation studies and identifying the most impactful component removals. Finally, the system stores each interaction (frames, OCR output, questions, and answers) in MongoDB, producing a complete and searchable log of the user’s session.

-----------------------------------------------------------------------------------------------------

Files
project/webrtc_receiver.py

FastAPI server handling:
    - WebRTC connection
    - Frame uploads
    - HTML interface
    - Saved frame path

project/frames/latest.png

Continuously updated screenshot of the user’s screen.

Notebooks:
    - 00_environment.ipynb
    - 01_single_image_prototype.ipynb
    - 02_gradio_single_image.ipynb
    - 03_webrtc_bridge.ipynb
    - 04_gradio_app.ipynb

Makefile Includes:
make resume — recreate environment
make webrtc — start the screen receiver
make mongo-start — start the MongoDB container