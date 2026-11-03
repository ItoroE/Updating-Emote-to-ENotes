import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from dataset import list_songs, ClipDataset
from train import CNNNet, CodebookQuant, ROUNDS 

device = "cuda" if torch.cuda.is_available() else "cpu"

#same split as training 
root = "music_data_separate/Instrumental"
songs = list_songs(root)
val_num = max(1, int(len(songs) / 10))
valid_songs = songs[:val_num]
train_songs = songs[val_num:]
assert set(train_songs).isdisjoint(valid_songs)

valid_data = ClipDataset(root, valid_songs)
train_data = ClipDataset(root, train_songs)

# a random slice of training clips, the same size as the validation set
g = torch.Generator().manual_seed(0)
n = min(len(valid_data), len(train_data))
train_subset = Subset(train_data, torch.randperm(len(train_data), generator=g)[:n].tolist())

valid_loader = DataLoader(valid_data, batch_size=16, shuffle=False)
train_loader = DataLoader(train_subset, batch_size=16, shuffle=False)

#load saved weights back into model to test
network = CNNNet().to(device)
vq = CodebookQuant(512, 128, ROUNDS).to(device)
ckpt = torch.load("vqvae.pt", map_location=device)
network.load_state_dict(ckpt["network"])
vq.load_state_dict(ckpt["vq"])
network.eval()
# turns off the codebook init and dead-entry restart
vq.eval()                                    


@torch.no_grad()
def evaluate(loader):
    total_recon, total_vq, count = 0.0, 0.0, 0
    # distinct codes used per round, over the whole set
    seen = [set() for _ in range(ROUNDS)]     
    for x, label, has_label, start in loader:
        x = x.to(device)
        feat = network(x)
        q, idx, vq_loss = vq(feat)
        recon = network.decode(q, x.shape[-2:])
        # weight by batch size so a smaller last batch counts for same amount of weights
        b = x.shape[0]                       
        total_recon += F.mse_loss(recon, x).item() * b
        total_vq += vq_loss.item() * b
        count += b
        for r in range(idx.shape[-1]):
            seen[r].update(idx[..., r].unique().tolist())
    return total_recon / count, total_vq / count, [len(s) for s in seen], count


@torch.no_grad()
def baselines():
    # prediction scores on the validation clips without crossreferencing clips
    prof_sum, n = torch.zeros(1, 1, 128, 1), 0
    for x, _, _, _ in train_loader:
        prof_sum += x.mean(dim=(0, 3), keepdim=True) * x.shape[0]
        n += x.shape[0]

    # average loudness per frequency band
    profile = prof_sum / n                  
    const = profile.mean().item()             

    b_const, b_prof, count = 0.0, 0.0, 0
    for x, _, _, _ in valid_loader:
        b = x.shape[0]
        b_const += F.mse_loss(torch.full_like(x, const), x).item() * b
        b_prof += F.mse_loss(profile.expand_as(x), x).item() * b
        count += b
    return b_const / count, b_prof / count


tr = evaluate(train_loader)
va = evaluate(valid_loader)
b_const, b_prof = baselines()

print(f"train subset ({tr[3]} clips):  recon {tr[0]:.4f}  vq {tr[1]:.4f}  codes used {tr[2]}")
print(f"validation   ({va[3]} clips):  recon {va[0]:.4f}  vq {va[1]:.4f}  codes used {va[2]}")
print(f"baseline, one constant:       {b_const:.4f}")
print(f"baseline, average spectrum:   {b_prof:.4f}")


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

x, _, _, _ = next(iter(valid_loader))
x = x.to(device)
with torch.no_grad():
    q, idx, _ = vq(network(x))
    recon = network.decode(q, x.shape[-2:])

fig, ax = plt.subplots(4, 2, figsize=(10, 12))
for i in range(4):
    ax[i, 0].imshow(x[i, 0].cpu(), origin="lower", aspect="auto", vmin=-1, vmax=1)
    ax[i, 0].set_title("input")
    ax[i, 1].imshow(recon[i, 0].cpu(), origin="lower", aspect="auto", vmin=-1, vmax=1)
    ax[i, 1].set_title("reconstruction")
plt.tight_layout()
plt.savefig("val_recon.png")
print("saved val_recon.png")