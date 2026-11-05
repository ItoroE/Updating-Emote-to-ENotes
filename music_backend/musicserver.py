
import base64
import json
import os
import urllib.request

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from music_backend import generate_music as M

EMOTION_URL = os.environ.get("EMOTION_URL", "http://localhost:8000")
#html js files for both browsers
PAGES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pages")      

app = FastAPI(title="Emotes to E-Notes music backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["X-Music-Info"])
app.mount("/pages", StaticFiles(directory=PAGES), name="pages")

# load the models once
@app.on_event("startup")
def startup():
    M.load()                               


def wav_response(wav, info):
    return Response(wav, media_type="audio/wav", headers={"X-Music-Info": json.dumps(info)})


@app.get("/generate")
def generate(valence: float = 0.0, arousal: float = 0.0, face_scale: bool = True, candidates: int = 3,
             seed: int | None = None):
    wav, info = M.generate_wav(valence, arousal, face_scale, candidates, seed)
    return wav_response(wav, info)


def current_emotion():
    try:
        with urllib.request.urlopen(f"{EMOTION_URL}/emotion", timeout=5) as r:
            e = json.load(r)
    except Exception as ex:
        raise HTTPException(503, f"emotion backend not reachable at {EMOTION_URL}: {ex}")
    # prefer the 10 s average (steadier) over the frame-by-frame value
    src = e["batch"] if e.get("batch") else e
    return float(src["valence"]), float(src["arousal"])


@app.get("/from_emotion")
def from_emotion(candidates: int = 3):
    v, a = current_emotion()
    wav, info = M.generate_wav(v, a, True, candidates)
    return wav_response(wav, info)


@app.get("/from_emotion.json")
def from_emotion_json(candidates: int = 3):
    # same as /from_emotion, but JSON with the wav inside (base64), for apps that want the numbers too
    v, a = current_emotion()
    wav, info = M.generate_wav(v, a, True, candidates)
    return {**info, "wav_base64": base64.b64encode(wav).decode()}


@app.get("/live/next")
def live_next(valence: float = 0.0, arousal: float = 0.0, session: str = "default", reset: bool = False,
              face_scale: bool = True, candidates: int = 1):
    # the next 3 s of one continuous piece, continuing from the last chunk of this session


    # Play the chunks back to back, they are already crossfaded. candidates=1 keeps it faster than real time on CPU.
    wav, info = M.live_chunk(session, valence, arousal, face_scale, candidates, reset)
    return wav_response(wav, info)


@app.get("/")
def live_page():
    return FileResponse(os.path.join(PAGES, "live.html"))


@app.get("/single")
def page():
    return FileResponse(os.path.join(PAGES, "single.html"))
