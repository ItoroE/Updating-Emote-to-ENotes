import IPython.display as ip
import matplotlib.pyplot as plt
import numpy as np
from  pathlib import Path
from audio_separator.separator import Separator
separator = Separator(
    output_format='MP3',
    output_dir=r'music_data\DEAM_vocal_instrumental_separation',
    model_file_dir='music_data'
)
separator.load_model('UVR-MDX-NET-Inst_HQ_3.onnx')

folder_path = Path('music_data') / 'DEAM_audio' / 'MEMD_audio'

print("Current working directory:", Path.cwd(), flush=True)
print("Looking inside:", folder_path.resolve(), flush=True)
print("Folder exists:", folder_path.exists(), flush=True)

#creates path object for every 

files = list(folder_path.iterdir())

print("Number of items found:", len(files), flush=True)
print("Contents:", files, flush=True)

for audio_file in files:
    print("Found item:", audio_file, flush=True)
    print("Is file:", audio_file.is_file(), flush=True)

    if audio_file.is_file():
        print("Processing:", audio_file, flush=True)

        output_files = separator.separate(str(audio_file))

        print("Output:", output_files, flush=True)
