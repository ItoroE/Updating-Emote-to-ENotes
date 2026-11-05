#  valence / arousal in, a 4 s wav out, using the trained models vocoder.pt, transformerv4.pt, vqvae.pt
# The emotion backend's face model gives valence and arousal on its own scale (about -1 to 1).
#  The music labels the transformer was trained on cover a smaller range (about -0.4 to 0.6), so face values are mapped onto that range
import io
import os
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                            

import generate as G                    

# one id at a time is small work not alot so 2 threads was fastest                              
torch.set_num_threads(2)                  

SR = 22050

# face values run from - to +
FACE_RANGE = 1.0                          

# the sampling settings tuned for transformer v4 
SETTINGS = SimpleNamespace(temperature=1.1, top_k=300, mood_scale=1.0, guidance=2.5, rounds=G.ROUNDS, candidates=3)

_models = None
_lock = threading.Lock()                   


# loads the training labels once
def load():
    global _models
    if _models is None:
        _models = G.load_models()
        G.mood_stats()                    
    return _models


def face_to_label(valence, arousal):
    # face scale (-1..1) and  music label scale, separately for each side of the average
    mu, lo, hi, _ = G.mood_stats()
    out = []
    for i, x in enumerate((valence, arousal)):
        x = max(-1.0, min(1.0, float(x) / FACE_RANGE))
        out.append(float(mu[i] + x * ((hi[i] - mu[i]) if x >= 0 else (mu[i] - lo[i]))))
    return out


def generate_wav(valence, arousal, face_scale=True, candidates=None, seed=None):
    # returns (wav bytes, info dict) so  face_scale=False takes valence / arousal already in music label units.
    network, vq, transformer = load()
    mood = face_to_label(valence, arousal) if face_scale else [float(valence), float(arousal)]
    args = SimpleNamespace(**vars(SETTINGS))
    if candidates is not None:
        args.candidates = max(1, int(candidates))
    with _lock:
        if seed is not None:
            torch.manual_seed(int(seed))
        t = time.time()
        _, db = G.generate_best(transformer, vq, network, mood, args)
        # uses neural vocoder 
        audio = G.mel_to_wav(db)            
        secs = time.time() - t
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV")
    info = {"valence_in": float(valence), "arousal_in": float(arousal), "face_scale": face_scale,
            "music_valence": round(mood[0], 3), "music_arousal": round(mood[1], 3),
            "musicality": round(float(G.musicality(db)), 2), "seconds": round(secs, 1)}
    return buf.getvalue(), info


#live one continuous piece, chunk by chunk each new 4 s clip starts with the ids of 11 time columns (about 1 s) from the end of the previous clip
COL_IDS = G.ROWS * G.ROUNDS                
COL_SAMPLES = 4 * 512                       
CONTEXT_COLS = 11
CLIP = 4 * SR            
 # each chunk moves this far from the last mood toward the new one                  
FADE = 2048  
LIVE_SMOOTH = 0.75                         
                                         


class LiveSession:
    def __init__(self):
        self.ids, self.tail, self.mood = None, None, None

    def next_chunk(self, valence, arousal, face_scale=True, candidates=1):
        network, vq, transformer = load()
        target = face_to_label(valence, arousal) if face_scale else [float(valence), float(arousal)]
        self.mood = target if self.mood is None else \
            [m + LIVE_SMOOTH * (t - m) for m, t in zip(self.mood, target)]
        args = SimpleNamespace(**vars(SETTINGS))
        args.candidates = max(1, int(candidates))
        with _lock:
            t = time.time()
            if self.ids is None:
                prefix, join = None, 0
            else:
                first = G.COLS - 1 - CONTEXT_COLS                          
                prefix = self.ids[0, first * COL_IDS:(G.COLS - 1) * COL_IDS]
                 # where the previous clip's audio ended
                join = CLIP - first * COL_SAMPLES                         
            ids, db = G.generate_best(transformer, vq, network, self.mood, args, prefix=prefix)
            audio = G.mel_to_wav(db)
            secs = time.time() - t
        if self.tail is None:
            chunk = audio[:CLIP - FADE]
        else:
            ramp = np.linspace(0, 1, FADE, dtype=np.float32)
            head = self.tail * (1 - ramp) + audio[join - FADE:join] * ramp
            chunk = np.concatenate([head, audio[join:CLIP - FADE]])
        self.ids, self.tail = ids, audio[CLIP - FADE:CLIP].astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, chunk, SR, format="WAV")
        info = {"valence_in": float(valence), "arousal_in": float(arousal),
                "music_valence": round(self.mood[0], 3), "music_arousal": round(self.mood[1], 3),
                "seconds_audio": round(len(chunk) / SR, 2), "seconds_to_make": round(secs, 1),
                "musicality": round(float(G.musicality(db)), 2)}
        return buf.getvalue(), info


_sessions = {}


def live_chunk(session, valence, arousal, face_scale=True, candidates=1, reset=False):
    if reset or session not in _sessions:
        _sessions[session] = LiveSession()
    return _sessions[session].next_chunk(valence, arousal, face_scale, candidates)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--valence", type=float, default=0.0)
    ap.add_argument("--arousal", type=float, default=0.0)
    ap.add_argument("--music_scale", action="store_true")    
     # values already in music label units 
    ap.add_argument("--out", type=str, default="music_backend_test.wav")
    a = ap.parse_args()
    wav, info = generate_wav(a.valence, a.arousal, face_scale=not a.music_scale)
    with open(a.out, "wb") as f:
        f.write(wav)
    print("saved", a.out, info)
