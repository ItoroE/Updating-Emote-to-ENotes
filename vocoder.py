
import argparse
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")            
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import librosa

SR, N_FFT, HOP, N_LIN = 22050, 2048, 512, 1025
POSTNET_PATH = "postnet.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"


class PostNet(nn.Module):
    def __init__(self, hidden=384):
        super().__init__()
        B = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=128, fmax=8000).T    
        #each bin average of its bands        
        spread = B / np.maximum(B.sum(1, keepdims=True), 1e-8)                         
        self.register_buffer("spread", torch.from_numpy(spread).float())
        self.register_buffer("empty", torch.from_numpy(B.sum(1) == 0)[:, None])        
        self.inp = nn.Conv1d(128, hidden, kernel_size=5, padding=2)
        self.blocks = nn.ModuleList([nn.Sequential(nn.GELU(), nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
                                                   nn.GELU(), nn.Conv1d(hidden, hidden, kernel_size=5, padding=2))
                                     for _ in range(2)])
        self.out = nn.Conv1d(hidden, N_LIN, kernel_size=1)
        nn.init.zeros_(self.out.weight)                          # start exactly at the spread
        nn.init.zeros_(self.out.bias)

    def forward(self, mel):
        m = mel[:, 0]
        base = torch.where(self.empty, torch.full_like(self.spread[:, :1], -1.0), self.spread @ m)
        h = self.inp(m)
        for b in self.blocks:
            #each block only adds details
            h = h + b(h)                                     
        return (base + self.out(F.gelu(h))).unsqueeze(1)


def postnet_losses(pred, target, frames=173):
    # take off the 3 extra padded columns 
    pred, target = pred[..., :frames], target[..., :frames]      
    l1 = F.l1_loss(pred, target)
    # residual: each block only adds detail
    fl1 = F.l1_loss(pred[:, :, 1:] - pred[:, :, :-1], target[:, :, 1:] - target[:, :, :-1])  
    return l1, fl1


#  mel -> audio
_POSTNET = [None, False]                                        


def load_postnet():
    if not _POSTNET[1]:
        _POSTNET[1] = True
        if os.path.exists(POSTNET_PATH):
            net = PostNet().to(device)
            net.load_state_dict(torch.load(POSTNET_PATH, map_location=device)["postnet"])
            net.eval()
            _POSTNET[0] = net
            print(f"vocoder: using {POSTNET_PATH} (mel -> full spectrum) + Griffin-Lim", flush=True)
        else:
            print(f"vocoder: {POSTNET_PATH} not found, falling back to librosa mel_to_audio", flush=True)
    return _POSTNET[0]


@torch.no_grad()
def mel_to_wav(db, n_iter=100):
    from neural_vocoder import load_vocoder, neural_mel_to_wav
    if load_vocoder() is not None:
        return neural_mel_to_wav(db)
    net = load_postnet()
    if net is None:
        audio = librosa.feature.inverse.mel_to_audio(librosa.db_to_power(db), sr=SR, n_fft=N_FFT, hop_length=HOP,
                                                     fmax=8000, length=4 * SR, n_iter=n_iter)
    else:
        x = np.pad((db + 40) / 40, ((0, 0), (0, 3)), constant_values=-1.0)
        lin = net(torch.from_numpy(x).float()[None, None].to(device))[0, 0, :, :db.shape[1]]
        mag = librosa.db_to_amplitude(lin.clamp(-1, 1).cpu().numpy() * 40 - 40)
        audio = librosa.griffinlim(mag, n_iter=n_iter, hop_length=HOP, length=4 * SR)
        # normalize the volume
    return audio / (np.abs(audio).max() + 1e-8) * 0.9                       


#training
def train(args):
    from dataset import list_songs, ClipDataset
    from generate import VQ_PATH
    from train import CNNNet, CodebookQuant, ROUNDS

    root = "music_data_separate/Instrumental"
    songs = list_songs(root)
    val_num = max(1, int(len(songs) / 10))                        
    train_data = ClipDataset(root, songs[val_num:], with_linear=True)
    valid_data = ClipDataset(root, songs[:val_num], with_linear=True)
    g = torch.Generator().manual_seed(0)
    if args.clips:        
         # a random part of the clips jsut for a quicker run                                        
        train_data = Subset(train_data, torch.randperm(len(train_data), generator=g)[:args.clips].tolist())
    valid_data = Subset(valid_data, torch.randperm(len(valid_data), generator=g)[:args.val_clips].tolist())
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, drop_last=True,
                              num_workers=args.workers, persistent_workers=args.workers > 0)
    valid_loader = DataLoader(valid_data, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    print(f"{len(train_data)} train clips, {len(valid_data)} validation clips, device {device}", flush=True)

    # the trained VQ-VAE, frozen: only used to make the blurry inputs the post-net will see at generation time
    network, vq = CNNNet().to(device), CodebookQuant(512, 128, ROUNDS).to(device)
    ckpt = torch.load(VQ_PATH, map_location=device)
    network.load_state_dict(ckpt["network"])
    vq.load_state_dict(ckpt["vq"])
    network.eval()
    vq.eval()

    @torch.no_grad()
    def through_vqvae(x):
        q, _, _ = vq(network(x))
        return network.decode(q, x.shape[-2:]).clamp(-1, 1)

    postnet = PostNet().to(device)
    optimizer = optim.Adam(postnet.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    best = float("inf")
    for epoch in range(args.epochs):
        postnet.train()
        for step, (x, lin) in enumerate(train_loader):
            x, lin = x.to(device), lin.to(device)
            use_vq = torch.rand(x.shape[0], 1, 1, 1, device=device) < args.vq_frac
             # half real mels, half VQ-VAE reconstructions
            inp = torch.where(use_vq, through_vqvae(x), x)         
            l1, fl1 = postnet_losses(postnet(inp), lin)
            loss = l1 + args.freq_w * fl1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step % 50 == 0:
                print(f"epoch {epoch}  step {step}/{len(train_loader)}  l1 {l1.item():.4f}  freq-diff l1 {fl1.item():.4f}",
                      flush=True)

        postnet.eval()
        tot = {"real": 0.0, "vq": 0.0}
        n = 0
        with torch.no_grad():
            for x, lin in valid_loader:
                x, lin = x.to(device), lin.to(device)
                for k, inp in (("real", x), ("vq", through_vqvae(x))):
                    l1, fl1 = postnet_losses(postnet(inp), lin)
                    tot[k] += (l1 + args.freq_w * fl1).item() * x.shape[0]
                n += x.shape[0]
        real, vqv = tot["real"] / n, tot["vq"] / n
        print(f"epoch {epoch}  VALIDATION  from real mel {real:.4f} | from VQ-VAE mel {vqv:.4f}", flush=True)
        if real + vqv < best:
            best = real + vqv
            torch.save({"epoch": epoch, "postnet": postnet.state_dict()}, POSTNET_PATH)
            print(f"saved {POSTNET_PATH}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clips", type=int, default=0)               
    ap.add_argument("--val_clips", type=int, default=400)

    # share of inputs that go through the VQ-VAE first
    ap.add_argument("--vq_frac", type=float, default=0.5)  

    # weight of the harmonic-sharpness loss       
    ap.add_argument("--freq_w", type=float, default=1.0)     

    # each worker loads wavs and runs an STFT     
    ap.add_argument("--workers", type=int, default=2)             
    args = ap.parse_args()
    if args.train:
        train(args)
