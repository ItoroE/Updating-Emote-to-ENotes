
#train on Kaggle with kaggle_job/train_vocoder.ipynb, put the downloaded neural_vocoder.pt in this folder
import argparse
import glob
import math
import os
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import librosa

SR, MEL_HOP = 22050, 512
VOCODER_PATH = "neural_vocoder.pt"


# generator
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, hidden, scale):
        super().__init__()
        # looks 3 frames each way in time
        self.dw = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)   
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, hidden)
        self.pw2 = nn.Linear(hidden, dim)
        # each block starts as a small change
        self.gamma = nn.Parameter(torch.full((dim,), scale))                   
    def forward(self, x):                                                       
        h = self.pw2(F.gelu(self.pw1(self.norm(self.dw(x).transpose(1, 2)))))
        return x + (self.gamma * h).transpose(1, 2)


class NeuralVocoder(nn.Module):
    N_FFT, HOP, UP = 1024, 256, 2

    def __init__(self, dim=512, hidden=1536, layers=8):
        super().__init__()
        self.inp = nn.Conv1d(128, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([ConvNeXtBlock(dim, hidden, 1 / layers) for _ in range(layers)])
        self.final = nn.LayerNorm(dim)
        self.bins = self.N_FFT // 2 + 1
        # magnitude and phase, per output frame
        self.head = nn.Linear(dim, self.UP * 2 * self.bins)                    
        self.register_buffer("window", torch.hann_window(self.N_FFT))

    @classmethod
    def samples(cls, frames):                                                    
        return cls.HOP * (frames * cls.UP - 1)

    def forward(self, mel):
        # mel: (B, 128, T) on the (dB + 40) / 40 scale is (B, samples)
        x = self.norm(self.inp(mel).transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        h = self.head(self.final(x.transpose(1, 2)))                             
        B, T, _ = h.shape
        h = h.reshape(B, T * self.UP, 2 * self.bins).transpose(1, 2)            
        log_mag, phase = h.float().chunk(2, dim=1)
        spec = torch.polar(torch.exp(log_mag).clamp(max=100.0), phase)
        return torch.istft(spec, self.N_FFT, self.HOP, window=self.window, center=True,
                           length=self.samples(T))



_VOCODER = [None, False]


def load_vocoder(device="cpu"):
    if not _VOCODER[1]:
        _VOCODER[1] = True
        if os.path.exists(VOCODER_PATH):
            net = NeuralVocoder()
            net.load_state_dict(torch.load(VOCODER_PATH, map_location="cpu")["generator"])
            _VOCODER[0] = net.to(device).eval()
            print(f"vocoder: using {VOCODER_PATH} (trained neural vocoder)", flush=True)
    return _VOCODER[0]


@torch.no_grad()
def neural_mel_to_wav(db, length=4 * SR):
    # db (128, T) mel in dB, -80..0 -> audio, peak 0.9
    net = load_vocoder()
    x = torch.from_numpy((np.asarray(db, dtype=np.float32) + 40) / 40)[None]
    audio = net(x)[0].numpy()
    audio = np.pad(audio, (0, max(0, length - len(audio))))[:length]
    return audio / (np.abs(audio).max() + 1e-8) * 0.9


#Discrimanator
class PeriodDisc(nn.Module):
    # looks at the audio folded into rows of period samples and catches buzz and wrong periodic structure
    def __init__(self, period):
        super().__init__()
        self.period = period
        ch = [1, 32, 128, 512, 1024]
        self.convs = nn.ModuleList([weight_norm(nn.Conv2d(ch[i], ch[i + 1], (5, 1), (3, 1), padding=(2, 0)))
                                    for i in range(4)] + [weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0)))])
        self.out = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, y):
        B, L = y.shape
        if L % self.period:
            y = F.pad(y, (0, self.period - L % self.period), mode="reflect")
        x = y.view(B, 1, -1, self.period)
        feats = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            feats.append(x)
        x = self.out(x)
        feats.append(x)
        return x.flatten(1), feats


class ResolutionDisc(nn.Module):
    # looks at a spectrogram of the audio and  catches smeared harmonics and noise between them
    def __init__(self, n_fft):
        super().__init__()
        self.n_fft = n_fft
        self.convs = nn.ModuleList([weight_norm(nn.Conv2d(1, 32, (3, 9), padding=(1, 4)))] +
                                   [weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))) for _ in range(3)] +
                                   [weight_norm(nn.Conv2d(32, 32, (3, 3), padding=(1, 1)))])
        self.out = weight_norm(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, y):
        win = torch.hann_window(self.n_fft, device=y.device)
        x = torch.stft(y, self.n_fft, self.n_fft // 4, window=win, return_complex=True).abs()[:, None]
        x = x.transpose(2, 3)                                                    
        feats = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            feats.append(x)
        x = self.out(x)
        feats.append(x)
        return x.flatten(1), feats


class Discriminators(nn.Module):
    def __init__(self):
        super().__init__()
        self.discs = nn.ModuleList([PeriodDisc(p) for p in (2, 3, 5, 7, 11)] +
                                   [ResolutionDisc(n) for n in (512, 1024, 2048)])

    def forward(self, y):
        return [d(y) for d in self.discs]


#losses
class MelLoss(nn.Module):
    # L1 between log mel spectrograms at three resolutions, full band (0 - 11 kHz)
    def __init__(self):
        super().__init__()
        self.cfg = [(512, 128, 64), (1024, 256, 128), (2048, 512, 128)]
        for i, (n_fft, hop, mels) in enumerate(self.cfg):
            self.register_buffer(f"fb{i}", torch.from_numpy(librosa.filters.mel(sr=SR, n_fft=n_fft, n_mels=mels)).float())
            self.register_buffer(f"win{i}", torch.hann_window(n_fft))

    def forward(self, fake, real):
        loss = 0.0
        for i, (n_fft, hop, _) in enumerate(self.cfg):
            fb, win = getattr(self, f"fb{i}"), getattr(self, f"win{i}")
            a = fb @ torch.stft(fake, n_fft, hop, window=win, return_complex=True).abs()
            b = fb @ torch.stft(real, n_fft, hop, window=win, return_complex=True).abs()
            loss = loss + F.l1_loss(torch.log(a.clamp(min=1e-5)), torch.log(b.clamp(min=1e-5)))
        return loss / len(self.cfg)

# hinge loss
def d_loss(real_out, fake_out):                                                  
    return sum(F.relu(1 - r).mean() + F.relu(1 + f).mean() for (r, _), (f, _) in zip(real_out, fake_out))


def g_losses(real_out, fake_out):
    adv = sum(F.relu(1 - f).mean() for f, _ in fake_out)
    fm = sum(F.l1_loss(ff, rf.detach()) for (_, rfs), (_, ffs) in zip(real_out, fake_out) for rf, ff in zip(rfs, ffs))
    return adv, fm / len(fake_out)


# pack data
def pack(args):
    # Clips for Kaggle, exactly the pairs the vocoder must learn
    from dataset import list_songs
    root = "music_data_separate/Instrumental"
    songs = list_songs(root)
    val_num = max(1, int(len(songs) / 10))                                     
    pick = []
    for part, ss, n in (("train", songs[val_num:], args.clips), ("val", songs[:val_num], args.val_clips)):
        files = []
        for s in ss:
            files += [f for f in glob.glob(os.path.join(root, s, "*_[0-9]*.npz"))
                      if os.path.exists(f.replace(".npz", ".wav"))]
        random.Random(0).shuffle(files)
        pick += [(f, part == "val") for f in files[:n]]

    import soundfile as sf
    N = len(pick)
    audio = np.zeros((N, 4 * SR), dtype=np.int16)
    specs = np.zeros((N, 128, 173), dtype=np.float16)
    scale = np.zeros(N, dtype=np.float32)
    for i, (f, _) in enumerate(pick):
        a, _ = sf.read(f.replace(".npz", ".wav"), dtype="int16")
        audio[i, :len(a)] = a[:4 * SR]
        specs[i] = np.load(f)["spec"]
        y = a.astype(np.float32) / 32768
        S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=2048, hop_length=MEL_HOP, n_mels=128, fmax=8000)
        scale[i] = 1.0 / np.sqrt(S.max() + 1e-12)
        if (i + 1) % 500 == 0:
            print(f"  packed {i + 1}/{N}", flush=True)

    out = args.data
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "voc_audio.npy"), audio)
    np.save(os.path.join(out, "voc_specs.npy"), specs)
    np.save(os.path.join(out, "voc_scale.npy"), scale)
    np.save(os.path.join(out, "voc_is_val.npy"), np.array([v for _, v in pick]))
    import shutil
    shutil.copy2("vqvae.pt", os.path.join(out, "vqvae.pt"))
    mb = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)) / 1e6
    print(f"packed {N} clips ({sum(v for _, v in pick)} validation) into {out}/ ({mb:.0f} MB): upload this folder to Kaggle")


#training
def train(args):
    from train import CNNNet, CodebookQuant, ROUNDS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = args.data
    audio = torch.from_numpy(np.load(os.path.join(d, "voc_audio.npy")).astype(np.float32) / 32768)
    audio = audio * torch.from_numpy(np.load(os.path.join(d, "voc_scale.npy")))[:, None]
    specs = torch.from_numpy(np.load(os.path.join(d, "voc_specs.npy")).astype(np.float32))
    is_val = torch.from_numpy(np.load(os.path.join(d, "voc_is_val.npy")))
    tr, va = torch.where(~is_val)[0], torch.where(is_val)[0]
    print(f"{len(tr)} train clips, {len(va)} validation clips, device {device}", flush=True)

    network, vq = CNNNet().to(device), CodebookQuant(512, 128, ROUNDS).to(device)   # frozen, for blurry inputs
    ck = torch.load(os.path.join(d, "vqvae.pt"), map_location=device)
    network.load_state_dict(ck["network"])
    vq.load_state_dict(ck["vq"])
    network.eval()
    vq.eval()

    @torch.no_grad()
    def mel_input(idx, use_vq):
        x = (specs[idx].to(device) + 40) / 40                                
        if use_vq.any():
            xp = F.pad(x, (0, 3), value=-1.0)[:, None]
            q, _, _ = vq(network(xp[use_vq]))
            x[use_vq] = network.decode(q, (128, 176)).clamp(-1, 1)[:, 0, :, :173]
        return x

    gen, disc, mel_loss = NeuralVocoder().to(device), Discriminators().to(device), MelLoss().to(device)
    opt_g = torch.optim.AdamW(gen.parameters(), lr=args.lr, betas=(0.8, 0.9))
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr, betas=(0.8, 0.9))
    step = 0
    # continue a run from a previous session for kaggle
    if args.resume and os.path.exists(args.resume):                             
        ck = torch.load(args.resume, map_location=device)
        gen.load_state_dict(ck["generator"])
        disc.load_state_dict(ck["discriminators"])
        opt_g.load_state_dict(ck["opt_g"])
        opt_d.load_state_dict(ck["opt_d"])
        step = ck["step"]
        print(f"resumed from {args.resume} at step {step}", flush=True)
    print(f"generator {sum(p.numel() for p in gen.parameters()) / 1e6:.1f}M weights, "
          f"discriminators {sum(p.numel() for p in disc.parameters()) / 1e6:.1f}M", flush=True)

    def lr_at(s, hours):
        # slow start, then cosine decay. Progress is steps or time whichever is further along, so the decay also
        # finishes when the run is cut by the Kaggle CPU limit
        progress = min(1.0, max(s / args.steps, hours / args.hours))
        return args.lr * min(1.0, (s + 1) / 1000) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    def save(path, full):
        ck = {"generator": gen.state_dict(), "step": step}
        if full:
            ck.update(discriminators=disc.state_dict(), opt_g=opt_g.state_dict(), opt_d=opt_d.state_dict())
        torch.save(ck, path)

    @torch.no_grad()
    def validate():
        gen.eval()
        tot = {"real mel": 0.0, "VQ-VAE mel": 0.0}
        for k, use in (("real mel", False), ("VQ-VAE mel", True)):
            for i in range(0, len(va), 16):
                idx = va[i:i + 16]
                fake = gen(mel_input(idx, torch.full((len(idx),), use, device=device)))[:, :audio.shape[1]]
                tot[k] += mel_loss(fake, audio[idx, :fake.shape[1]].to(device)).item() * len(idx)
        gen.train()
        return {k: v / len(va) for k, v in tot.items()}

    T = args.frames
    seg = NeuralVocoder.samples(T)
    t_start, t_last_save, best = time.time(), time.time(), float("inf")
    gen.train()
    disc.train()
    while step < args.steps:
        idx = tr[torch.randint(len(tr), (args.batch,))]
        use_vq = torch.rand(args.batch, device=device) < args.vq_frac
        mel = mel_input(idx, use_vq)
        t0 = torch.randint(0, min(173 - T, (4 * SR - seg) // MEL_HOP) + 1, (args.batch,))
        mel = torch.stack([mel[b, :, t0[b]:t0[b] + T] for b in range(args.batch)])
        real = torch.stack([audio[i, s * MEL_HOP:s * MEL_HOP + seg] for i, s in zip(idx.tolist(), t0.tolist())]).to(device)
        for g in opt_g.param_groups + opt_d.param_groups:
            g["lr"] = lr_at(step, (time.time() - t_start) / 3600)

        fake = gen(mel)
        # a few mel-only steps first allow for a stabler start
        gan_on = step >= args.gan_start                                        
        if gan_on:
            ld = d_loss(disc(real), disc(fake.detach()))
            opt_d.zero_grad()
            ld.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 10.0)
            opt_d.step()
        lm = mel_loss(fake, real)
        loss = 45 * lm
        if gan_on:
            adv, fm = g_losses(disc(real), disc(fake))
            loss = loss + adv + 2 * fm
        opt_g.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 10.0)
        opt_g.step()
        step += 1

        hours = (time.time() - t_start) / 3600
        if step % 200 == 0:
            msg = f"step {step}  mel {lm.item():.4f}"
            if gan_on:
                msg += f"  disc {ld.item():.3f}  adv {adv.item():.3f}  feat {fm.item():.3f}"
            print(msg + f"  ({step / max(time.time() - t_start, 1):.1f} steps/s, {hours:.2f} h)", flush=True)
        if step % args.val_every == 0 or step == args.steps or hours > args.hours:
            v = validate()
            print(f"step {step}  VALIDATION mel error  from real mel {v['real mel']:.4f} | "
                  f"from VQ-VAE mel {v['VQ-VAE mel']:.4f}", flush=True)
            score = v["real mel"] + v["VQ-VAE mel"]
            if score < best:
                best = score
                save(VOCODER_PATH, full=False)
                print(f"saved {VOCODER_PATH}", flush=True)
        if time.time() - t_last_save > 1800 or step == args.steps or hours > args.hours:
            save("vocoder_ckpt.pt", full=True)                                   
            t_last_save = time.time()
        if hours > args.hours:
            print(f"time limit of {args.hours} h reached at step {step}: saved vocoder_ckpt.pt to continue later", flush=True)
            break

    
    gen.eval()                                                                 
    import soundfile as sf
    gen.load_state_dict(torch.load(VOCODER_PATH, map_location=device)["generator"])
    with torch.no_grad():
        for k, i in enumerate(va[:3].tolist()):
            for tag, use in (("real_mel", False), ("vqvae_mel", True)):
                y = gen(mel_input(torch.tensor([i]), torch.tensor([use], device=device)))[0].cpu().numpy()
                sf.write(f"example{k}_{tag}.wav", y / (np.abs(y).max() + 1e-8) * 0.9, SR)
            a = audio[i].numpy()
            sf.write(f"example{k}_original.wav", a / (np.abs(a).max() + 1e-8) * 0.9, SR)
    print("wrote example*.wav", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", action="store_true")
    # training clips to pack 4s each defualt
    ap.add_argument("--clips", type=int, default=4000)            
    ap.add_argument("--val_clips", type=int, default=64)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--data", default="kaggle_job/vocoder_data")
    ap.add_argument("--steps", type=int, default=200000)

    # stop and save before Kaggle's 12 h session limit
    ap.add_argument("--hours", type=float, default=11.0)        
    ap.add_argument("--batch", type=int, default=16)

    # mel frames per training piece 0.75 s
    ap.add_argument("--frames", type=int, default=32)            
    ap.add_argument("--lr", type=float, default=5e-4)

    # share of inputs that go through the VQ-VAE first
    ap.add_argument("--vq_frac", type=float, default=0.5)        
    ap.add_argument("--gan_start", type=int, default=2000)
    ap.add_argument("--val_every", type=int, default=5000)
    ap.add_argument("--resume", default="")
    args = ap.parse_args()
    if args.pack:
        pack(args)
    if args.train:
        train(args)
