import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    #one face per frame
    os.environ.setdefault(v, "1")          
import cv2
cv2.setNumThreads(1)
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import emote

app = FastAPI(title="Emotes to E-Notes emotion backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/frame")
def frame(file: UploadFile = File(...)):
    img = cv2.imdecode(np.frombuffer(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not read the image")
    return emote.uploaded_frame(img)


@app.get("/emotion")
def emotion():
    # the current smoothed mood, plus the last 10 s average if one is ready , read in 
    last = emote.emo_batches[-1] if emote.emo_batches else None
    return {"valence": emote.get_valence(), "arousal": emote.get_arousal(), "batch": last}


@app.get("/batches")
def batches():
    return emote.emo_batches
