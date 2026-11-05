import torch 
from torch.utils.data import DataLoader
from dataset import list_songs, ClipDataset
from train import CNNNet, CodebookQuant


root = "music_data_separate/Instrumental"
songs = list_songs(root)
val_num = max(1, int((len(songs)/10)))
valid_songs = songs[:val_num]
train_songs = songs[val_num:]

network = CNNNet()
ckpt = torch.load("vqvae.pt")
network.load_state_dict(ckpt["network"])

# Extract num_codebooks from checkpoint to avoid shape mismatch
num_codebooks = ckpt["vq"]["codebook"].shape[0]
vq = CodebookQuant(512, 128, num_codebooks)
vq.load_state_dict(ckpt["vq"])
network.eval()
vq.eval()


@torch.no_grad()
def tokenize(songs, output):
    data = ClipDataset(root, songs)
    loader = DataLoader(data, batch_size = 32, shuffle = False)
    tot_idx, tot_label, tot_haslabel, tot_start = [], [], [], []
    
    for x, label, has_label, start in loader:
        _, idx, _ = vq(network(x))
        tot_idx.append(idx)
        tot_label.append(label)
        tot_haslabel.append(has_label)
        tot_start.append(start)
    torch.save({
        "idx": torch.cat(tot_idx),             
        "label": torch.cat(tot_label),         
        "has_label": torch.cat(tot_haslabel),       
        "start": torch.cat(tot_start),         
    }, output)
    print(output, len(data), "clips")

tokenize(train_songs, "tokens_train.pt")
tokenize(valid_songs, "tokens_val.pt")
