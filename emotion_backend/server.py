# server.py  -  the emotion backend: send it webcam frames, it tracks valence / arousal from the face (emote.py).
# Run from inside emotion_backend (emote.py opens ./facial_expression_analysis/models/... by relative path):
#   cd emotion_backend
#   uvicorn server:app --port 8000
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")           # one face per frame is small work: leave the CPU cores to the music generator
import cv2
cv2.setNumThreads(1)
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import emote

app = FastAPI(title="Emotion backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/frame")
async def frame(file: UploadFile = File(...)):
    # one webcam image (jpg / png) -> the smoothed emotion after this frame
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not read the image")
    return emote.uploaded_frame(img)


@app.get("/emotion")
def emotion():
    # the current smoothed mood, plus the last 10 s average if one is ready (the music backend reads this)
    last = emote.emo_batches[-1] if emote.emo_batches else None
    return {"valence": emote.get_valence(), "arousal": emote.get_arousal(), "batch": last}


@app.get("/batches")
def batches():
    return emote.emo_batches
