import argparse
import glob
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")                       
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from train import CNNNet, CodebookQuant, ROUNDS, GRID
from transformer import MusicTransformer, MusicTransformerV4, load_tokens
from vocoder import mel_to_wav

device = "cuda" if torch.cuda.is_available() else "cpu"

# 8 frequency rows, 44 time columns
ROWS, COLS = GRID    

# 704 ids per clip
N_TOK = ROWS * COLS * ROUNDS                
VQ_PATH = "vqvae.pt"
TRANSFORMER_PATH = "transformer.pt"


def _read_conventions():
    # just makes sure the shape is the same as training
    sd = torch.load(TRANSFORMER_PATH, map_location="cpu")
    tok_key = next(k for k in sd if k.startswith("tok"))
    pos_key = next(k for k in sd if k.startswith("pos"))
    vocab, max_len = sd[tok_key].shape[0], sd[pos_key].shape[0]
    start = int(load_tokens("tokens_val.pt").tensors[0][0, 0])
    return vocab, max_len, start


VOCAB, MAX_LEN, START_ID = _read_conventions()
print(f"transformer conventions: vocabulary {VOCAB}, positions {MAX_LEN}, start id {START_ID}, "
      f"{N_TOK} ids per clip", flush=True)

# song labels cover within this range for the songs used as training data, fall between normal boounds 
# of what the generator can produced
MOODS = {
    "upbeat":  (0.40, 0.45),
    "intense": (-0.10, 0.50),
    "calm":    (0.20, -0.20),
    "somber":  (-0.30, -0.20),
    "neutral": (0.10, 0.15),
}


#loading model in 
def load_models():
    network = CNNNet().to(device)
    vq = CodebookQuant(512, 128, ROUNDS).to(device)
    ckpt = torch.load(VQ_PATH, map_location=device)
    if ckpt.get("rounds", 4) != ROUNDS:
        raise SystemExit(f"{VQ_PATH} was trained with {ckpt.get('rounds', 4)} rounds but this code expects {ROUNDS}.")
    network.load_state_dict(ckpt["network"])
    vq.load_state_dict(ckpt["vq"])
    network.eval()
    vq.eval()

    sd = torch.load(TRANSFORMER_PATH, map_location=device)
    #trained with mood hidden sometimes, allowance for classifier free guidance
    guided = "null_mood" in sd                                           
    d = sd["token.weight"].shape[1]
    if "blocks.0.qkv.weight" in sd:                                        
        layers = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
        transformer = MusicTransformerV4(vocab=VOCAB, d=d, heads=int(sd["n_heads"]), max_len=MAX_LEN,
                                         layers=layers).to(device)
        print(f"transformer: version 4, {layers} layers", flush=True)
    else:
        transformer = MusicTransformer(vocab=VOCAB, d=d, heads=4, max_len=MAX_LEN, guided=guided).to(device)
    transformer.load_state_dict(sd)
    if not guided:
        print("transformer: older model without mood guidance, --guidance has no effect", flush=True)
    transformer.eval()
    return network, vq, transformer


# mood regulators


_STATS = [None]


def mood_stats():
    # holds training nonNone labels, returns mean of them 
    if _STATS[0] is None:
        d = torch.load("tokens_train.pt")
        lab = d["label"][d["has_label"].bool()].float()
        _STATS[0] = (lab.mean(0), torch.quantile(lab, 0.01, dim=0), torch.quantile(lab, 0.99, dim=0), lab)
    return _STATS[0]


def scale_mood(mood, scale):
    # push mood farther from the average, just toggled with it seems to applify mood's affect on every token
    mu, lo, hi, _ = mood_stats()
    req = torch.minimum(torch.maximum(mu + scale * (torch.as_tensor(mood, dtype=torch.float32) - mu), lo), hi)
    return [float(req[0]), float(req[1])]



@torch.no_grad()
def generate_ids(transformer, mood, temperature=0.8, top_k=50, mood_scale=1.0, guidance=1.0,
                 fine_temperature=None, top_p=1.0, prefix=None):
   #based on specific mood writes the one clips 704 tokens 
    mood = scale_mood(mood, mood_scale) if mood_scale != 1.0 else list(mood) 

    # (1, 2): valence, arousal  
    mood_t = torch.tensor([mood], dtype=torch.float32, device=device)    
    tokens = torch.tensor([[START_ID]], device=device)                   

    # use guidance if TRUE and the model supports it
    guided = guidance != 1.0 and getattr(transformer, "guided", False)     
    B = 2 if guided else 1                                                 
    labels = mood_t.expand(B, -1)                

    # hides the mood on row 1 only                        
    hide = torch.tensor([False, True], device=device) if guided else None  
   # remembers previous version of transformer
    cache = transformer.new_cache() if hasattr(transformer, "new_cache") else None
    n = 0                                                               
    if prefix is not None:                                                
        prefix = torch.as_tensor(prefix, device=device).reshape(1, -1) 

        # the loop skips these positions  
        n = prefix.shape[1]                                                
        tokens = torch.cat([tokens, prefix], dim=1)                      
        if cache is not None and n > 0:                                  
            transformer(tokens[:, :-1].expand(B, -1), labels, hide, cache=cache)   

    # p, which clip token we are writing
    for p in range(n, N_TOK):                                         
        r = p % ROUNDS                                                    
        # coarse ids use temperature; fine ids use fine_temperature if one was given
        temp = temperature if r == 0 or fine_temperature is None else fine_temperature
        inp = tokens[:, -1:] if cache is not None else tokens              
        # run the transformer and keep the scores for the last position
        out = transformer(inp.expand(B, -1), labels, hide, **({"cache": cache} if cache is not None else {}))[:, -1]
        if guided:
             # no-mood + guidance to check what the mood adds
            logits = (out[1:] + guidance * (out[:1] - out[1:])) / temp    
        else:
             # scores for the next id
            logits = out / temp                                           

        # start with every id banned
        allowed = torch.full_like(logits, float("-inf")) 

         # only this round's id range is legal and banned ids drop to -inf               
        allowed[:, r * 512:(r + 1) * 512] = 0                             
        logits = logits + allowed                                        

        # keep only the top_k best choices
        kth = torch.topk(logits, top_k).values[:, -1:]      

        # ban everything below them               
        logits[logits < kth] = float("-inf")                               
        if top_p < 1.0:                 
            #keep ids that pss top-p probability and let go of all other ids                                  
            sorted_logits, order = torch.sort(logits, descending=True)     
            cume = F.softmax(sorted_logits, dim=-1).cumsum(-1)              
            cut = cume - F.softmax(sorted_logits, dim=-1) > top_p           
            logits[0, order[0, cut[0]]] = float("-inf")                    
        # samples one id from proability tehn feed it back in
        probs = F.softmax(logits, dim=-1)                                 
        nxt = torch.multinomial(probs, 1)                                  
        tokens = torch.cat([tokens, nxt], dim=1)                           

    return tokens[:, 1:]                                                   


@torch.no_grad()
def ids_to_spectrogram(ids, vq, network, rounds=ROUNDS):
     # undoing the time-major flatten and perround offsets
    ids = ids.reshape(1, COLS, ROWS, ROUNDS).permute(0, 2, 1, 3)          
    ids = ids - torch.arange(ROUNDS, device=ids.device) * 512              
    q = 0
    for r in range(rounds):                                                
        q = q + vq.codebook[r][ids[..., r]]                                
    q = q.permute(0, 3, 1, 2)                                              
    return network.decode(q, (128, 176))                                  


def musicality(db):
    # higher values show clearer notes except when sound breaks, flicker, burst out.
    db = np.maximum(db - db.max(), -80)
    tonal = np.abs(np.diff(db, axis=0)).mean()
    flicker = np.abs(np.diff(db, axis=1)).mean()
    dropouts = (db.max(axis=0) < -40).mean()
    return tonal - 0.5 * flicker - 10 * dropouts


@torch.no_grad()
def generate_best(transformer, vq, network, mood, args, prefix=None):
    # generates args.candidates clips and keeps the one with the clearest notes. About half of single samples are
    # noise bursts or a wash
    best = None
    for c in range(max(1, args.candidates)):
        ids = generate_ids(transformer, mood, args.temperature, args.top_k, args.mood_scale, args.guidance,
                           prefix=prefix)
        db = to_db(ids_to_spectrogram(ids, vq, network, args.rounds))
        score = musicality(db)
        # candidates ranked by musicality ranked my musicality
        if best is None or score > best[0]:
            best = (score, ids, db)
    print(f"best of {args.candidates}: musicality {best[0]:.2f} (real clips: median about 3.2)", flush=True)
    return best[1], best[2]


# decoder output , dB, drop the 3 padded columns that make frame evenly divided

def to_db(spec):                                                           
    return spec[0, 0, :, :173].clamp(-1, 1).cpu().numpy() * 40 - 40

#help turn spec into a wav file 
def db_to_wav(db, path, n_iter=100):
    audio = mel_to_wav(db, n_iter)                                       
    sf.write(path, audio, 22050)
    return audio


#number and pics for every clip
def report(db, name, ids=None):
    low, mid, high = db[:43].mean(), db[43:86].mean(), db[86:].mean()
    print(f"[{name}] mean {db.mean():.1f} dB | low/mid/high bands {low:.1f} / {mid:.1f} / {high:.1f} dB")
    print(f"[{name}] change over time {db.std(axis=1).mean():.1f} dB | loud cells (> -40 dB): {(db > -40).mean():.0%}")

    if ids is not None:                                                   
        g = ids.reshape(COLS, ROWS, ROUNDS).cpu()                        
        unique = [len(g[..., r].unique()) for r in range(ROUNDS)]
        same_as_prev = (g[1:] == g[:-1]).float().mean().item()
        print(f"[{name}] distinct ids per round (of {ROWS * COLS}): {unique} | same as previous time step: {same_as_prev:.0%}")

    plt.figure(figsize=(8, 3))
    plt.imshow(db, origin="lower", aspect="auto", vmin=-80, vmax=0)
    plt.title(name)
    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.close()


#  a real clip through the ids path, round by round
@torch.no_grad()
def real_ids_test(npz_path, network, vq, args):
    # a real clip's ids pushed through the generation path: checks the undo steps
    real_db = np.load(npz_path)["spec"]                                 
    x = np.pad((real_db + 40) / 40, ((0, 0), (0, 3)), constant_values=-1.0)
    x = torch.from_numpy(x).float()[None, None].to(device)               

    q, idx, _ = vq(network(x))                                            
    direct = network.decode(q, (128, 176))                               

    offset = torch.arange(ROUNDS, device=device) * 512
    seq = (idx + offset).permute(0, 2, 1, 3).reshape(1, -1)               

    outs = {}
    for r in range(1, ROUNDS + 1):
        outs[r] = ids_to_spectrogram(seq, vq, network, rounds=r)
        db_to_wav(to_db(outs[r]), f"R{r}_real_ids.wav", args.n_iter)

    diff = (outs[ROUNDS] - direct).abs().max().item()
    print(f"max difference, ids path vs direct decode ({ROUNDS} rounds): {diff:.6f}")

    panels = [("real", real_db), ("direct VQ-VAE decode", to_db(direct))]
    panels += [(f"from real ids, {r} round(s)", to_db(outs[r])) for r in range(1, ROUNDS + 1)]
    fig, ax = plt.subplots(len(panels), 1, figsize=(10, 2.4 * len(panels)))
    for a, (title, db) in zip(ax, panels):
        a.imshow(db, origin="lower", aspect="auto", vmin=-80, vmax=0)
        a.set_title(title)
    plt.tight_layout()
    plt.savefig("real_ids.png")
    plt.close()
    print("saved R*_real_ids.wav and real_ids.png")


# -accuracy of transformer test
@torch.no_grad()
def accuracy_test(transformer):
    val = load_tokens("tokens_val.pt")
    train = load_tokens("tokens_train.pt")
    loader = DataLoader(val, batch_size=64, shuffle=False)

    top1, top5 = torch.zeros(ROUNDS), torch.zeros(ROUNDS)
    nll, nll_shuf = torch.zeros(ROUNDS), torch.zeros(ROUNDS)
    count = 0

    def masked(logits, r):                                  
        lg = logits[:, r::ROUNDS].clone()                    
        lg[..., :r * 512] = float("-inf")
        lg[..., (r + 1) * 512:] = float("-inf")
        return lg

    for tokens, label in loader:
        tokens, label = tokens.to(device), label.to(device)
        # the model sees the real sequence , learns to adjust itself correctly
        inp, tgt = tokens[:, :-1], tokens[:, 1:]             
        logits = transformer(inp, label)
        shuffled = label[torch.randperm(label.shape[0], device=device)]
          # same tokens, wrong moods
        logits_shuf = transformer(inp, shuffled)           

        for r in range(ROUNDS):
            lg, y = masked(logits, r), tgt[:, r::ROUNDS]
            top1[r] += (lg.argmax(-1) == y).sum().item()
            top5[r] += (lg.topk(5, dim=-1).indices == y.unsqueeze(-1)).any(-1).sum().item()
            nll[r] += F.cross_entropy(lg.reshape(-1, VOCAB), y.reshape(-1), reduction="sum").item()
            nll_shuf[r] += F.cross_entropy(masked(logits_shuf, r).reshape(-1, VOCAB),
                                           y.reshape(-1), reduction="sum").item()
        count += tgt.shape[0] * ROWS * COLS                  # positions per round per clip
    tr_seq, va_seq = train.tensors[0][:, 1:], val.tensors[0][:, 1:]
    print(f"{len(val)} validation clips | uniform-guess loss per round: {math.log(512):.2f}")
    for r in range(ROUNDS):
        common = torch.bincount(tr_seq[:, r::ROUNDS].flatten(), minlength=VOCAB).argmax()
        base = (va_seq[:, r::ROUNDS] == common).float().mean().item()
        print(f"round {r + 1}: top-1 {top1[r] / count:.1%} (always-guess-common: {base:.1%}) | "
              f"top-5 {top5[r] / count:.1%} | loss {nll[r] / count:.2f} | loss with wrong moods {nll_shuf[r] / count:.2f}")


#make sure mood is being taken into account
def label_check():
    d = torch.load("tokens_train.pt")
    lab = d["label"][d["has_label"].bool()].float()
    for i, name in enumerate(["valence", "arousal"]):
        q = torch.quantile(lab[:, i], torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
        print(f"{name}: mean {lab[:, i].mean():+.3f}  std {lab[:, i].std():.3f}  "
              f"5/25/50/75/95% = {[round(v, 2) for v in q.tolist()]}")
    print("valence-arousal correlation:", round(torch.corrcoef(lab.T)[0, 1].item(), 2))


def _real_rows(n):
    files = glob.glob("music_data_separate/Instrumental/*/*.npz")
    random.Random(0).shuffle(files)
    rows = []
    for f in files[:n]:
        d = np.load(f)
        if np.isnan(d["label"]).any():
            continue
        db = d["spec"]
        rows.append([d["label"][0], d["label"][1], db.mean(),
                     db[86:].mean() - db[:43].mean(), db.std(axis=1).mean()])
    return np.array(rows)


def real_correlation(n=3000):
    #  checking if the labels move the same way in the real audio clips like I have it occuring in the generated one
    r = _real_rows(n)
    print(len(r), "labeled clips checked")
    for j, name in enumerate(["mean loudness", "brightness (high minus low bands)", "change over time"]):
        print(f"{name}: corr with valence {np.corrcoef(r[:, 0], r[:, 2 + j])[0, 1]:+.2f}, "
              f"with arousal {np.corrcoef(r[:, 1], r[:, 2 + j])[0, 1]:+.2f}")


def real_by_mood(n=3000):
    # what REAL clips look like at low / high valence and arousal
    r = _real_rows(n)
    for i, name in enumerate(["valence", "arousal"]):
        lo, hi = np.quantile(r[:, i], [0.1, 0.9])
        for tag, sel in (("low", r[:, i] <= lo), ("high", r[:, i] >= hi)):
            print(f"real clips, {tag:4s} {name:7s}: mean dB {r[sel, 2].mean():.1f} | "
                  f"brightness {r[sel, 3].mean():.1f} | change {r[sel, 4].mean():.1f}")


@torch.no_grad()
def reference_clips(transformer, network, vq, args):
    # What each preset mood is SUPPOSED to sound like and  the real clips whose labels are closest to it,
    files = glob.glob("music_data_separate/Instrumental/*/*_[0-9]*.npz")
    labels, keep = [], []
    for f in files:
        lab = np.load(f)["label"]
        if not np.isnan(lab).any():
            labels.append(lab)
            keep.append(f)
    labels = np.array(labels)
    print(f"{len(keep)} labeled clips searched")

    for name, (v, a) in MOODS.items():
        out = os.path.join("mood_reference", name)
        os.makedirs(out, exist_ok=True)
        picked, songs = [], set()
        for i in np.argsort(np.linalg.norm(labels - [v, a], axis=1)):  
            song = os.path.basename(os.path.dirname(keep[i]))
            if song not in songs:
                picked.append(i)
                songs.add(song)
            if len(picked) == args.n_ref:
                break
        for k, i in enumerate(picked):
            real_db = np.load(keep[i])["spec"]
            audio, _ = sf.read(keep[i].replace(".npz", ".wav"), dtype="float32")
            sf.write(f"{out}/real{k}_1_original.wav", audio / (np.abs(audio).max() + 1e-8) * 0.9, 22050)
            db_to_wav(real_db, f"{out}/real{k}_2_mel_vocoder.wav", args.n_iter)
            x = torch.from_numpy(np.pad((real_db + 40) / 40, ((0, 0), (0, 3)), constant_values=-1.0)).float()[None, None]
            q, _, _ = vq(network(x.to(device)))
            db_to_wav(to_db(network.decode(q, (128, 176))), f"{out}/real{k}_3_through_vqvae.wav", args.n_iter)
            print(f"[{name}] real{k}: {os.path.basename(keep[i])}  label v {labels[i][0]:+.2f} a {labels[i][1]:+.2f}")
            torch.manual_seed(k)
            _, db = generate_best(transformer, vq, network, [v, a], args)
            db_to_wav(db, f"{out}/generated{k}.wav", args.n_iter)
        print(f"[{name}] (v {v:+.2f}, a {a:+.2f}) saved to {out}/", flush=True)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--valence", type=float, default=0.2)
    ap.add_argument("--arousal", type=float, default=-0.2)
    ap.add_argument("--temperature", type=float, default=1.1)    
    ap.add_argument("--top_k", type=int, default=300)       

    # 1.0 = ask for the mood as given, 2.0 = exaggerate it     
    ap.add_argument("--mood_scale", type=float, default=1.0)  

    # mood guidance strength, 1.0 = off   
    ap.add_argument("--guidance", type=float, default=2.5)   

     # how many codebook rounds to decode    
    ap.add_argument("--rounds", type=int, default=ROUNDS) 
    # Griffin-Lim iterations(used in previous iteration of the vocoder)      
    ap.add_argument("--n_iter", type=int, default=100)           
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=str, default="generated.wav")
   
    # generate every preset mood
    ap.add_argument("--all", action="store_true")                

    # path to a real .npz clip
    ap.add_argument("--accuracy", action="store_true")
    ap.add_argument("--real_ids", type=str, default=None)      
    ap.add_argument("--mood_tests", action="store_true")
    
     # real clips for each preset mood, to listen to
    ap.add_argument("--reference", action="store_true")    

    # real clips (and generated ones) per mood     
    ap.add_argument("--n_ref", type=int, default=3)   

    # clips generated per request, the clearest is kept        
    ap.add_argument("--candidates", type=int, default=6)         
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    network, vq, transformer = load_models()

    if args.real_ids:
        real_ids_test(args.real_ids, network, vq, args)
    elif args.accuracy:
        accuracy_test(transformer)
    elif args.mood_tests:
        label_check()
        real_correlation()
        real_by_mood()
    elif args.reference:
        reference_clips(transformer, network, vq, args)
    elif args.all:
        for name, (v, a) in MOODS.items():
            ids, db = generate_best(transformer, vq, network, [v, a], args)
            db_to_wav(db, f"mood_{name}.wav", args.n_iter)
            report(db, f"mood_{name}", ids)
    else:
        ids, db = generate_best(transformer, vq, network, [args.valence, args.arousal], args)
        db_to_wav(db, args.out, args.n_iter)
        report(db, os.path.splitext(args.out)[0], ids)
        print("saved", args.out, "valence", args.valence, "arousal", args.arousal)