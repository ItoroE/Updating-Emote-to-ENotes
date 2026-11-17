# Updating Emote to E-Notes

Emotion-conditioned audio generation using a transformer, quantized audio tokens, and a neural vocoder. Includes webcam-based emotion tracking and browser playback of sequential WAV chunks.

## Overview

The system accepts **valence** and **arousal** values either from a webcam-based facial-expression pipeline or from manual sliders. It maps those values to the music model's training-label range, generates quantized audio representations with a transformer, decodes them to a spectrogram, and synthesizes WAV audio. A live-session pipeline carries context from one generated chunk into the next and crossfades boundaries.

## Architecture

The application separates emotion recognition, audio generation, and playback into distinct components. Users can provide emotion values through webcam analysis or manual controls in the browser.

| Component | Location | Description |
| --- | --- | --- |
| Browser interface | `music_backend/pages/` | Captures webcam frames, provides manual emotion sliders, and schedules generated audio for playback. |
| Emotion service | `emotion_backend/` | Detects facial landmarks, estimates valence and arousal, and maintains smoothed emotion readings. |
| Music API | `music_backend/musicserver.py` | Exposes endpoints for single-clip generation and continuous live sessions. |
| Audio generation | `music_backend/generate_music.py` | Converts emotion values to model conditioning labels and coordinates audio generation. |
| Model pipeline | `transformer.py`, `generate.py`, `neural_vocoder.py` | Samples quantized audio tokens, reconstructs audio representations, and synthesizes waveforms. |

The frontend sends emotion values and a session identifier to the music API. The generation pipeline produces WAV audio and metadata, which the browser receives and plays through the Web Audio API. For live sessions, the backend preserves context between successive clips and crossfades their boundaries.


## Features

- **Facial emotion tracking:** OpenCV/dlib-based facial-expression analysis, smoothing, and 10-second aggregated emotion windows.
- **Manual emotion controls:** Browser sliders allow testing without webcam access.
- **Conditional music generation:** A transformer samples audio tokens conditioned on two-dimensional valence/arousal labels.
- **Quantized audio representation:** Tokenization, vector quantization, spectrogram reconstruction, and vocoder-related code are separated into modules.
- **WAV endpoints:** Generate an audio clip directly from supplied emotion values or from the emotion backend's latest readings.
- **Stateful live continuation:** Live sessions reuse approximately one second of token context from the preceding clip, smooth changes in mood, and crossfade adjacent audio chunks.
- **Playback diagnostics:** API response headers report the selected music emotion, generation time, and a musicality metric; the browser reports generation latency and potential playback gaps.

## Tech stack

**Modeling:** Python, PyTorch, NumPy, transformer-based sequence modeling, vector quantization, spectrogram processing, neural vocoding.

**Backend:** FastAPI, OpenCV, dlib, SoundFile.

**Frontend:** HTML, JavaScript, browser webcam APIs, Web Audio API.

## Project structure

| File | Role |
| --- | --- |
| `transformer.py` | Emotion-conditioned transformer model |
| `train.py` | Model training logic |
| `dataset.py`, `data.py` | Dataset preparation and loading |
| `tokenization.py` | Audio tokenization utilities |
| `generate.py` | Model loading, token sampling, and audio generation |
| `neural_vocoder.py`, `vocoder.py` | Audio waveform reconstruction components |
| `validation.py` | Validation/diagnostic routines |
| `emotion_backend/emote.py` | Facial-expression inference and emotion aggregation |
| `emotion_backend/server.py` | `/frame`, `/emotion`, and `/batches` endpoints |
| `music_backend/generate_music.py` | Audio-generation wrapper, emotion-scale mapping, and live-session continuation |
| `music_backend/musicserver.py` | WAV-serving API and HTML pages |
| `music_backend/pages/live.*` | Live webcam/manual controls and sequential audio playback |
| `music_backend/pages/single.*` | Single-clip generation UI |

## Setup

### Prerequisites

- Python environment with the required dependencies
- Compatible transformer, VQ-VAE, and vocoder checkpoints
- Facial-landmark and emotion-model assets

### Configuration

1. Create a Python environment and install the dependencies used by the modules, including PyTorch, FastAPI, Uvicorn, OpenCV, dlib, NumPy, SoundFile, and the model-specific audio libraries imported in the source.
2. Configure the checkpoint files expected by the generation code (including the trained transformer, VQ-VAE, and vocoder weights).
3. Configure the facial model assets expected by `emotion_backend/emote.py`.
4. Start each service in a separate terminal:

   - **Emotion backend:** Open a terminal in `emotion_backend/` and run Bash: `uvicorn server:app --port 8000`
   - **Music backend:** Open another terminal in the repository root and run Bash: `uvicorn music_backend.musicserver:app --port 8001`

5. Open `http://localhost:8001/` for the live interface or `http://localhost:8001/single` for single-clip generation.

### Frontend routing

The live frontend requests the emotion service through `/emotion-api`. Configure a reverse proxy for this path or change the frontend API URL. Webcam access requires localhost or HTTPS in most browsers.

## API endpoints

| Endpoint | Behavior |
| --- | --- |
| `GET /generate?valence=0&arousal=0` | Return generated WAV audio with metadata headers |
| `GET /from_emotion` | Generate from the emotion backend's current/aggregated reading |
| `GET /from_emotion.json` | Return generation metadata and base64-encoded WAV |
| `GET /live/next?valence=0&arousal=0&session=demo` | Continue a session and return its next WAV chunk |
| `GET /` | Live interface |
| `GET /single` | Single-clip interface |

The `/live/next` endpoint accepts `reset=true` to start a fresh session. The emotion backend additionally exposes `POST /frame`, `GET /emotion`, and `GET /batches`.

## Live audio continuation

The live generator stores the previous chunk's sampled token IDs, its audio tail, and its current mood. On the next request it uses the prior token context as a prefix, moves the current mood toward the requested target, generates another clip, and crossfades the overlap. The browser queues WAV chunks with the Web Audio API and monitors whether generation is keeping up with playback. Audio is generated in chunks; playback continuity depends on generation latency and hardware performance.

## Known limitations

- Trained checkpoints and facial-analysis assets must be supplied separately.
- The `/emotion-api` route requires proxy configuration or a frontend URL change.
- Live sessions are held in memory and reset when the server restarts.
- Generation latency depends on hardware and sampling settings.
- No reproducible deployment configuration is included.

## Architecture comparison

| Original Emote to E-Notes | Updated version |
| --- | --- |
| GRU conditional VAE | Transformer with quantized audio representation |
| MIDI note events | Generated WAV audio |
| MIDI playback service | Browser Web Audio playback |
| Emotion-triggered batches | Stateful, context-aware chunk continuation |
| MIDI sequence assembly | Spectrogram reconstruction, vocoder, and crossfading |

