from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Set

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole

FRAME_PATH = Path("/workspaces/eng-ai-agents/project/frames/latest.png")
FRAME_PATH.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI()
pcs: Set[RTCPeerConnection] = set()


class Offer(BaseModel):
    sdp: str
    type: str


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8" />
    <title>CUA Screen Share</title>
    <style>
      body { font-family: sans-serif; padding: 1rem; }
      button { padding: 0.5rem 1rem; margin-top: 1rem; }
      video { margin-top: 1rem; max-width: 100%; border: 1px solid #ccc; }
      #status { margin-top: 1rem; font-size: 0.9rem; }
    </style>
  </head>
  <body>
    <h2>Computer Using Agent – Screen Share</h2>
    <p>
      Click the button below, choose "Your Entire Screen" or a window with your paper,
      and the stream will be sent to the Python WebRTC receiver.
    </p>

    <button id="startBtn">Start Screen Sharing</button>

    <div id="status">Status: idle</div>

    <video id="preview" autoplay playsinline></video>

    <script>
      const startBtn = document.getElementById("startBtn");
      const statusEl = document.getElementById("status");
      const preview = document.getElementById("preview");

      let pc = null;

      let canvas = document.createElement("canvas");
      let ctx = canvas.getContext("2d");
      let frameIntervalId = null;

      function logStatus(msg) {
        console.log(msg);
        statusEl.textContent = "Status: " + msg;
      }

      async function startScreenShare() {
        try {
          if (pc) {
            logStatus("Peer connection already active.");
            return;
          }

          logStatus("Requesting screen capture...");
          const stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });

          // Show preview locally (optional)
          preview.srcObject = stream;

          // Set canvas size to match video
          const track = stream.getVideoTracks()[0];
          const settings = track.getSettings();
          const width = settings.width || 1280;
          const height = settings.height || 720;
          canvas.width = width;
          canvas.height = height;

          // Start periodic frame upload via HTTP
          if (frameIntervalId) {
            clearInterval(frameIntervalId);
          }
          frameIntervalId = setInterval(() => {
            if (!preview.srcObject) {
              return;
            }

            // Draw current video frame onto canvas
            ctx.drawImage(preview, 0, 0, canvas.width, canvas.height);

            canvas.toBlob(async (blob) => {
              if (!blob) return;
              try {
                const formData = new FormData();
                formData.append("file", blob, "frame.png");
                await fetch("/frame", {
                  method: "POST",
                  body: formData,
                });
                statusEl.textContent = "Status: screen sharing active (HTTP upload)";
              } catch (err) {
                console.error("Error uploading frame:", err);
              }
            }, "image/png");
          }, 1000); // one frame per second


          pc = new RTCPeerConnection();

          // Add all tracks from the screen stream
          stream.getTracks().forEach(track => pc.addTrack(track, stream));

          pc.onconnectionstatechange = () => {
            logStatus("Connection state: " + pc.connectionState);
          };

          pc.oniceconnectionstatechange = () => {
            console.log("ICE state:", pc.iceConnectionState);
          };

          // We'll send the offer once ICE gathering is complete
          pc.onicegatheringstatechange = async () => {
            if (pc.iceGatheringState === "complete") {
              logStatus("ICE gathering complete, sending offer to server...");
              const offer = pc.localDescription;
              try {
                const response = await fetch("/offer", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({
                    sdp: offer.sdp,
                    type: offer.type
                  })
                });
                const answer = await response.json();
                logStatus("Received answer from server, setting remote description...");
                await pc.setRemoteDescription(answer);
                logStatus("Screen sharing active (check Python logs for frames).");
              } catch (err) {
                console.error("Error sending offer to /offer:", err);
                logStatus("Error sending offer to server, see console.");
              }
            }
          };

          logStatus("Creating offer...");
          const offer = await pc.createOffer();
          await pc.setLocalDescription(offer);
          logStatus("Local description set, gathering ICE candidates...");

        } catch (err) {
          console.error("Error starting screen share:", err);
          logStatus("Error starting screen share: " + err.message);
        }
      }

      startBtn.addEventListener("click", () => {
        startScreenShare();
      });
    </script>
  </body>
</html>
    """


@app.post("/frame")
async def upload_frame(file: UploadFile = File(...)):
    """
    Accept a single image file (PNG/JPEG) and store it as the latest frame
    for the CUA pipeline.
    """
    contents = await file.read()

    FRAME_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FRAME_PATH, "wb") as f:
        f.write(contents)

    return {"status": "ok"}


@app.post("/offer")
async def offer_endpoint(offer: Offer):
    """
    Receive an SDP offer from the browser, create an RTCPeerConnection,
    hook the video track, and return an SDP answer.
    """
    pc = RTCPeerConnection()
    pcs.add(pc)
    print("Created RTCPeerConnection, total PCs:", len(pcs))

    media_blackhole = MediaBlackhole()

    @pc.on("track")
    async def on_track(track):
        print("Track received:", track.kind)
        if track.kind == "video":
            asyncio.create_task(save_frames_loop(track))
        else:
            media_blackhole.addTrack(track)

        @track.on("ended")
        async def on_ended():
            print("Track ended:", track.kind)
            await media_blackhole.stop()

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=offer.sdp, type=offer.type)
    )

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


async def save_frames_loop(track):
    """
    Continuously receive frames from the WebRTC video track and
    periodically save them to FRAME_PATH as latest.png.
    """
    print("Starting save_frames_loop for track:", track.kind)
    frame_count = 0

    while True:
        frame = await track.recv()
        img = frame.to_ndarray(format="bgr24")

        frame_count += 1
        if frame_count % 5 == 0:
            cv2.imwrite(str(FRAME_PATH), img)
            if frame_count % 50 == 0:
                print(f"Saved frame {frame_count} to {FRAME_PATH}")


@app.on_event("shutdown")
async def on_shutdown():
    """
    Close all peer connections on shutdown.
    """
    print("Shutting down, closing peer connections...")
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
