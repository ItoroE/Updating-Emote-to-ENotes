import argparse
import os

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import list_songs, ClipDataset

# how many residual codes there end up being 512 represented per round
ROUNDS = 2                                  

# token grid from 128,176 (frequency rows, time columns), end poooling output
GRID = (8, 44)                                


class CNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder: frequency is pooled 4 times (128 -> 8), while to time value i pool twice (toggled around with this... in hopes of learning more rhythm)
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)))    
        # pooling shrinks it to64x88
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)))    
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 1)))    
        #final shrinking finally to 8,44
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                   nn.MaxPool2d(kernel_size=(2, 1)))              

      
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),   
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),    
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),                   
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),                 
        )

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

    def decode(self, q, size=None):
        out = self.dec(q)
        # only resize if sizing doesnt math, cause images to blur
        '''if size is not None and tuple(out.shape[-2:]) != tuple(size):    
            out = F.interpolate(out, size=size, mode="bilinear", align_corners=False)'''
        return out

# each round make ones token for every cell , 2 rounds makes 704 tokens plus the 1 start token
class CodebookQuant(nn.Module):
    def __init__(self, num_embeddings, dim, num_quantizers=1, decay=0.99, dead_threshold=0.5):
        super().__init__()
        self.codebook = nn.Parameter(torch.zeros(num_quantizers, num_embeddings, dim))
        #whether or not value has been saved , initted checks whteher codebook has real data yet and still needs to be trained
        self.register_buffer("initted", torch.tensor(False))
        # count of how often each code is picked
        self.register_buffer("usage", torch.ones(num_quantizers, num_embeddings))   # decaying pick counts
        # decay allows for weight on more current steps, old coutns are mutikplied by decay before new ones are pciked
        self.decay = decay

        # replaces any code whose count ever falls below 0.5 
        self.dead_threshold = dead_threshold

    def forward(self, x):
        B, C, H, W = x.shape
        z = x.permute(0, 2, 3, 1).reshape(-1, C)

        #runs once to fill codebook
        if self.training and not self.initted:
            with torch.no_grad():
                leftover = z.detach()
                for r in range(self.codebook.shape[0]):
                    pick = torch.randperm(leftover.shape[0], device=z.device)[: self.codebook.shape[1]]
                   
                    self.codebook.data[r] = leftover[pick]
                    nearest = torch.cdist(leftover, self.codebook[r]).argmin(dim=1)
                    leftover = leftover - self.codebook[r][nearest]
            self.initted.fill_(True)

        residual = z
        output = torch.zeros_like(z)
        all_idx = []
        # what each round was asked to quantize
        round_inputs = []                   

        for r in range(self.codebook.shape[0]):
            c = self.codebook[r]
            round_inputs.append(residual.detach())
            with torch.no_grad():
                idx = torch.cdist(residual.detach(), c.detach()).argmin(dim=1)
            q = c[idx]
            if self.training:
                with torch.no_grad():
                    counts = torch.bincount(idx, minlength=c.shape[0]).float()
                    self.usage[r].mul_(self.decay).add_(counts)
            residual = residual - q.detach()
            output = output + q
            all_idx.append(idx)

        codebook_loss = F.mse_loss(output, z.detach())
        commit_loss = F.mse_loss(z, output.detach())
        weighted_loss = codebook_loss + 0.25 * commit_loss

        #reset entries noone picks
        if self.training:                                     
            with torch.no_grad():
                for r in range(self.codebook.shape[0]):
                    dead = self.usage[r] < self.dead_threshold
                    n_dead = int(dead.sum().item())
                    if n_dead > 0:
                        src = round_inputs[r]
                        pick = torch.randint(0, src.shape[0], (n_dead,), device=src.device)
                        self.codebook.data[r][dead] = src[pick]
                        self.usage[r][dead] = 1.0

        out = z + (output - z).detach()
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)
        idx = torch.stack(all_idx, dim=-1).view(B, H, W, -1)
        return out, idx, weighted_loss


def recon_losses(recon, x):
    # need more than regular loss between cells, so svae dl1 and fl1 for more specific about what changes
    l1 = F.l1_loss(recon, x) 

    # saves frame  to frame (how each band changes moment to moment)                                                     
    dl1 = F.l1_loss(recon[..., 1:] - recon[..., :-1], x[..., 1:] - x[..., :-1])     
    # band-to-band hild loudness changes
    fl1 = F.l1_loss(recon[:, :, 1:] - recon[:, :, :-1], x[:, :, 1:] - x[:, :, :-1])
    return l1, dl1, fl1


@torch.no_grad()
def validate(network, vq, loader, device):
    network.eval()
    vq.eval()
    tot = {"mse": 0.0, "l1": 0.0, "dl1": 0.0, "fl1": 0.0, "vq": 0.0}
    diff_recon, diff_real, n = 0.0, 0.0, 0
    fdiff_recon, fdiff_real = 0.0, 0.0
    seen = [set() for _ in range(ROUNDS)]
    for batch in loader:
        x = batch[0].to(device)
        q, idx, vq_loss = vq(network(x))
        recon = network.decode(q, x.shape[-2:])
        l1, dl1, fl1 = recon_losses(recon, x)
        b = x.shape[0]
        tot["mse"] += F.mse_loss(recon, x).item() * b
        tot["l1"] += l1.item() * b
        tot["dl1"] += dl1.item() * b
        tot["fl1"] += fl1.item() * b
        tot["vq"] += vq_loss.item() * b
        diff_recon += (recon[..., 1:] - recon[..., :-1]).abs().mean().item() * b
        diff_real += (x[..., 1:] - x[..., :-1]).abs().mean().item() * b
        fdiff_recon += (recon[:, :, 1:] - recon[:, :, :-1]).abs().mean().item() * b
        fdiff_real += (x[:, :, 1:] - x[:, :, :-1]).abs().mean().item() * b
        n += b
        for r in range(ROUNDS):
            seen[r].update(idx[..., r].unique().tolist())
    out = {k: v / n for k, v in tot.items()}
    
    # how much frame-to-frame changes survives 1 is all 0 none
    out["detail"] = diff_recon / diff_real     

    # how much band-to-band  detail survives
    out["harmonic"] = fdiff_recon / fdiff_real            
    out["codes"] = [len(s) for s in seen]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)

     # weight of the frame-to-frame loss
    ap.add_argument("--time_w", type=float, default=1.0)  

    # weight of the band-to-band loss 
    ap.add_argument("--freq_w", type=float, default=1.0)  

    #replaces existing vqvae.pt
    ap.add_argument("--overwrite", action="store_true")   
    args = ap.parse_args()

    if os.path.exists("vqvae.pt") and not args.overwrite:
        raise SystemExit("vqvae.pt already exists. Rename it")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = "music_data_separate/Instrumental"
    songs = list_songs(root)
    val_num = max(1, int(len(songs) / 10))
    valid_songs = songs[:val_num]
    train_songs = songs[val_num:]
    # make sure no song is in both splits
    assert set(train_songs).isdisjoint(valid_songs)        

    train_data = ClipDataset(root, train_songs)
    valid_data = ClipDataset(root, valid_songs)
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch, shuffle=False)
    print(f"{len(train_data)} train clips, {len(valid_data)} validation clips, device {device}")

    network = CNNNet().to(device)
    vq = CodebookQuant(512, 128, ROUNDS).to(device)

    
    with torch.no_grad():
        dummy = torch.zeros(2, 1, 128, 176, device=device)
        feat = network(dummy)
        assert tuple(feat.shape[-2:]) == GRID, f"token grid is {tuple(feat.shape[-2:])}, expected {GRID}"
        assert tuple(network.dec(feat).shape[-2:]) == (128, 176), "decoder output does not match the input size"
    print(f"token grid {GRID}, {ROUNDS} rounds -> {GRID[0] * GRID[1] * ROUNDS} ids per clip")

    optimizer = optim.Adam(list(network.parameters()) + list(vq.parameters()), lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    best = float("inf")
    for epoch in range(args.epochs):
        network.train()
        vq.train()
        for step, (x, label, has_label, start) in enumerate(train_loader):
            x = x.to(device)
            feat = network(x)
            q, idx, vq_loss = vq(feat)
            recon = network.decode(q, x.shape[-2:])

            l1, dl1, fl1 = recon_losses(recon, x)
            loss = l1 + args.time_w * dl1 + args.freq_w * fl1 + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 100 == 0:
                usage = [idx[..., r].unique().numel() for r in range(idx.shape[-1])]
                print(f"epoch {epoch}  step {step}/{len(train_loader)}  total {loss.item():.4f}  "
                      f"l1 {l1.item():.4f}  diff-l1 {dl1.item():.4f}  freq-l1 {fl1.item():.4f}  vq {vq_loss.item():.4f}  "
                      f"feat std {feat.std().item():.3f}  codes used {usage}", flush=True)

        v = validate(network, vq, valid_loader, device)
        print(f"epoch {epoch}  VALIDATION  mse {v['mse']:.4f}  l1 {v['l1']:.4f}  diff-l1 {v['dl1']:.4f}  "
              f"freq-l1 {v['fl1']:.4f}  vq {v['vq']:.4f}  frame-to-frame detail kept {v['detail']:.0%}  "
              f"harmonic detail kept {v['harmonic']:.0%}  codes used {v['codes']}", flush=True)

        ckpt = {"epoch": epoch, "network": network.state_dict(), "vq": vq.state_dict(),
                "optimizer": optimizer.state_dict(), "rounds": ROUNDS, "grid": GRID}
        torch.save(ckpt, "vqvae.pt")                                  # the latest epoch
        score = v["l1"] + args.time_w * v["dl1"] + args.freq_w * v["fl1"]
        if score < best:
            best = score
             # the best validation epoch
            torch.save(ckpt, "vqvae_best.pt")                        
            print(f"saved vqvae_best.pt (validation l1 + diff-l1 = {score:.4f})", flush=True)
        print(f"saved checkpoint after epoch {epoch}", flush=True)