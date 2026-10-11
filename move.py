import librosa
import librosa.display
from  pathlib import Path
import shutil
import soundfile as sf
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd          
import re
import sys
import gc                    
#seperate files into diff directories


folder_path = Path('music_data_separate')

instrumental = (folder_path/ 'Instrumental')
vocal = (folder_path/'Vocal')

#parents true m kdir makes file if it doesnt exist and if it exists the true flag cuases return nothing 
instrumental.mkdir(parents =True, exist_ok=True)
vocal.mkdir(parents=True, exist_ok = True)

ifile = list((folder_path/ 'Instrumental').iterdir())
vfile = list((folder_path/'Vocal').iterdir())

#parents true m kdir makes file if it doesnt exist and if it exists the true flag cuases return nothing 
instrumental.mkdir(parents =True, exist_ok=True)
vocal.mkdir(parents=True, exist_ok = True)
'''
for file in ifile:
    if file.is_file():
        if ('(Instrumental)_UVR-MDX-NET-Inst_HQ_3.mp3' in file.name):
            shutil.move(file, instrumental)

        else:{
            shutil.move(file, vocal)
        }

for file in vfile:
    if file.is_file():
        if ('(Instrumental)_UVR-MDX-NET-Inst_HQ_3.mp3' in file.name):
            shutil.move(file, instrumental)

        else:{
            shutil.move(file, vocal)
        }
'''
N_FFT, HOP, N_MELS, FMAX = 2048, 512, 128, 8000

# new: load the annotations once, keyed by song id
ann_folder = Path("music_data/DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)")

val = pd.read_csv(ann_folder / "valence.csv")
aro = pd.read_csv(ann_folder / "arousal.csv")

val.columns = val.columns.str.strip()      # remove stray spaces in headers
aro.columns = aro.columns.str.strip()

val = val.set_index("song_id")             # rows are looked up by song id
aro = aro.set_index("song_id")


def col_times(df):
    times = []
    for c in df.columns:                   # "sample_15000ms"
        ms = int(c.split("_")[1][:-2])     # 15000
        times.append(ms / 1000)            # 15.0 seconds
    return np.array(times)


val_t = col_times(val)                     # each file keeps its own time axis
aro_t = col_times(aro)


def window_label(sid, t0, t1):
    out = []
    for df, t in ((val, val_t), (aro, aro_t)):
        row = df.loc[sid].values
        in_window = (t >= t0) & (t < t1)
        v = row[in_window]
        v = v[~np.isnan(v)]
        if len(v) > 0:
            out.append(v.mean())
        else:
            out.append(np.nan)             # no annotation in this window
    return np.array(out, dtype=np.float32) # [valence, arousal]


# Get audio files from instrumental folder
audio_path = list(instrumental.iterdir())

# PROCESS_LIMIT: set to None for all files, or a number to limit for testing
PROCESS_LIMIT = None  # Change to 2 to test quickly

if PROCESS_LIMIT:
    audio_path = audio_path[:PROCESS_LIMIT]

#start , stop, step

for idx, audio_file in enumerate(audio_path):
    print(f"Processing {idx+1}/{len(audio_path)}: {audio_file.name}")
    if audio_file.is_file():
        sid = int(re.match(r"\d+", audio_file.stem).group())
        if sid not in val.index or sid not in aro.index:
            continue                       # skip songs missing from either file

        y, sr = librosa.load(audio_file, sr=22050)
        audio_window = 4*sr
        hop_window = 2*sr
        split_count = 1
        split_folder = instrumental/audio_file.stem
        split_folder.mkdir(parents=True, exist_ok = True)
        
        # Copy original file to folder
        original_dest = split_folder/audio_file.name
        shutil.copy2(audio_file, original_dest)
        
        for start in range(0, len(y)-audio_window, hop_window):
            audio_slice = y[start: start + audio_window]
            S = librosa.feature.melspectrogram(y=audio_slice, sr=sr, n_fft=N_FFT,
                                               hop_length=HOP, n_mels=N_MELS, fmax=FMAX)
            spectrogram_db = librosa.power_to_db(S, ref=np.max)

            label = window_label(sid, start / sr, start / sr + 4)

            split_name = audio_file.stem + '_' + str(split_count)
            audio_route = split_folder/f"{split_name}.wav"
            sf.write(audio_route, audio_slice, sr)
            np.savez(split_folder/f"{split_name}.npz",
                     spec=spectrogram_db.astype(np.float32),
                     label=label,
                     start=start / sr)
            split_count +=1
        
        # Cleanup to free memory
        del y, S, spectrogram_db, audio_slice
        import gc
        gc.collect()

print("Done processing all files")
sys.exit(0)