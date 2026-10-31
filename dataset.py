import re
import librosa
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

#folder has some intial unused/failed attempts so goes through folder and only picks up numbered npz files
CLIP_RE = re.compile(r"_(\d+)\.npz$")   

# get 173 frames from each spectrogram (each spec has 173 columns) 
# add extra padding so 4 pool is evenly divided  though
FRAMES = 176                              


def list_songs(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root.resolve()}")
    songs = []
    for d in root.iterdir():
        if d.is_dir():
            songs.append(d.name)
    songs.sort()
    return songs


def clip_number(path):
    #group 1  (/d+) 
    return int(CLIP_RE.search(path.name).group(1))


class ClipDataset(Dataset):
    # with_linear=True also returns the full STFT of the clip's .wav (1025 bins up to 11 kHz), the post-net's target
    def __init__(self, root, songs, with_linear=False):
        root = Path(root)
        self.with_linear = with_linear
        self.clips = []

        for s in songs:
            files = []
            for f in (root / s).glob("*.npz"):
                if CLIP_RE.search(f.name):
                    files.append(f)
            # add sorrted clips of all song.... glob may not return back in numeric order so this ensure it does
            # sort will manually call the clip function
            files.sort(key=clip_number)      
            self.clips.extend(files)

        # later for the conditional transformer, need to easily get the labeled arrays
        

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        d = np.load(self.clips[i])

        # rn now spec values range from -80 dB to 0 dB
        # want to normalize this cuz model train better on values centered around 0
        x = (d["spec"] + 40) / 40
        #adds padding columns of what isnt filled in by frames for pooling
        x = np.pad(x, ((0, 0), (0, FRAMES - x.shape[1])), constant_values=-1.0)

        # 128 frequences with 176 frames, altered the dimension to fit CNN
        x = torch.from_numpy(x).float().unsqueeze(0)    
        label = torch.from_numpy(d["label"]).float() 
        has_label = not torch.isnan(label).any().item()

        # keep NaN out of the loss
        label = torch.nan_to_num(label, nan=0.0)                   

        start = float(d["start"])
        if not self.with_linear:
            return x, label, has_label, start

        # same settings as the mel in move.py (n_fft 2048, hop 512), but every frequency bin, not 128 mel bands
        audio, _ = sf.read(str(self.clips[i]).replace(".npz", ".wav"), dtype="float32")
        lin = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=2048, hop_length=512)), ref=np.max, top_db=80)
        lin = np.pad((lin + 40) / 40, ((0, 0), (0, FRAMES - lin.shape[1])), constant_values=-1.0)
        return x, torch.from_numpy(lin).float().unsqueeze(0)

'''
if __name__ == "__main__":
    root = "music_data_separate/Instrumental"

    songs = list_songs(root)
    n_val = max(1, len(songs) // 10)
    val_songs = songs[:n_val]
    train_songs = songs[n_val:]
    assert set(train_songs).isdisjoint(val_songs)       # no song in both splits

    train_ds = ClipDataset(root, train_songs)
    val_ds = ClipDataset(root, val_songs)
    print(len(train_ds), "train clips,", len(val_ds), "val clips")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    x, label, has_label, start = next(iter(train_loader))
    print(x.shape, label.shape, has_label.shape, start.shape)
    # expect: (16, 1, 128, 176)  (16, 2)  (16,)  (16,)
    '''
