import numpy as np
import cv2
import dlib
import matplotlib.pyplot as plt
import json
import os
import time
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from facial_expression_analysis.source.emotions_dlib import EmotionsDlib, plot_landmarks
import facial_expression_analysis.source.emotions_dlib as emotions_dlib
import pls_numpy

# read the emotion model with numpy instead of scikit-learn (blocked by Smart App Control on this PC, see pls_numpy.py)
emotions_dlib.load = pls_numpy.load


EXP_AVG = 0.7
EMO_BATCH_SEC  = 10.0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./facial_expression_analysis/models/shape_predictor_68_face_landmarks.dat")

emotion_estimator = EmotionsDlib(
    file_emotion_model='./facial_expression_analysis/models/model_emotion_pls=30_fullfeatures=False.joblib', 
    file_frontalization_model='./facial_expression_analysis/models/model_frontalization.npy'
    )


disp_arousal = 0.0
disp_valence = 0.0
disp_intensity = 0.0

emotion_buffer = []
emo_batches = []
last_emosaved = time.time()
fr = 0

def get_arousal():
    return disp_arousal

def get_intensity():
    return disp_intensity

def get_valence():
    return disp_valence

def get_emotion_name(emotion_name):
    return emotion_name


def get_emotion_values(frame):
    """Detects the largest face in the frame and returns
    arousal, valence, intensity, emotion_name."""
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(grayscale)

    if len(faces) == 0:
        return None

    # choose largest face
    face_size = 0
    idx_largest_face = 0
    if len(faces) > 1:
        for i, face in enumerate(faces):
            current_size = ((face.bottom() - face.top()) * (face.right() - face.left()))
            if face_size < current_size:
                face_size = current_size
                idx_largest_face = i

    face = faces[idx_largest_face]
    landmarks_object = predictor(grayscale, face)
    dict_emotions = emotion_estimator.get_emotions(landmarks_object)

    arousal = dict_emotions['emotions']['arousal']
    valence = dict_emotions['emotions']['valence']
    intensity = dict_emotions['emotions']['intensity']
    emotion_name = dict_emotions['emotions']['name']

    arousal = float(np.asarray(arousal).squeeze())
    valence = float(np.asarray(valence).squeeze())
    intensity = float(np.asarray(intensity).squeeze())

    return (arousal, valence, intensity, emotion_name)


def uploaded_frame(frame):
    global disp_arousal, disp_valence, disp_intensity
    global emotion_buffer, emo_batches, last_emosaved, fr

    result = get_emotion_values(frame)

    if result is None:
        fr += 1
        return {
            "detected": False,
            "frame": fr
        }

    arousal, valence, intensity, emotion_name = result

    # exponential smoothing
    disp_arousal = round(
        EXP_AVG * arousal + (1 - EXP_AVG) * disp_arousal, 2
    )
    disp_valence = round(
        EXP_AVG * valence + (1 - EXP_AVG) * disp_valence, 2
    )
    disp_intensity = round(
        EXP_AVG * intensity + (1 - EXP_AVG) * disp_intensity, 2
    )

    emotion_buffer.append({
        "frame": fr,
        "timestamp": time.time(),
        "arousal": disp_arousal,
        "valence": disp_valence,
        "intensity": disp_intensity,
        "emotion_name": emotion_name
    })

    current_time = time.time()
    batch_data = None

    if current_time - last_emosaved >= EMO_BATCH_SEC and len(emotion_buffer) > 0:
        sum_arousal = 0
        sum_valence = 0
        sum_intensity = 0

        for e in emotion_buffer:
            sum_arousal += e["arousal"]
            sum_valence += e["valence"]
            sum_intensity += e["intensity"]

        avg_arousal = float(sum_arousal / len(emotion_buffer))
        avg_valence = float(sum_valence / len(emotion_buffer))
        #avg_intensity = float(sum_intensity / len(emotion_buffer))
        latest_name = emotion_buffer[-1]["emotion_name"]

        emo_batch = {
            "batch_id": len(emo_batches) +1,
            "start_batch": last_emosaved,
            "end_batch": current_time,
            "arousal": avg_arousal,
            "valence": avg_valence,
            #"intensity": avg_intensity,
            "emotion_name": latest_name
        }

        emo_batches.append(emo_batch)
        batch_data = emo_batch

        print("new batch:", emo_batch)

        with open("emotion_batches.json", "w") as f:
            json.dump(emo_batches, f, indent=2)

        emotion_buffer = []
        last_emosaved = current_time

    response = {
        "detected": True,
        "frame": fr,
        "arousal": disp_arousal,
        "valence": disp_valence,
        #"intensity": disp_intensity,
        "emotion_name": emotion_name,
        "batch": batch_data
    }

    fr += 1
    return response


    