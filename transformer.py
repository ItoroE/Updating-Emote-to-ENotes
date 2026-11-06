from train import CNNNet, CodebookQuant, ROUNDS, GRID
import math
import torch
import torch.nn as nn
from dataset import list_songs, ClipDataset
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim


class MusicTransformer(nn.Module):
   #latest version of the transformer
    def __init__(self, vocab, d, heads, max_len, guided=False):
        super().__init__()
        self.guided = guided
        if guided:
            self.null_mood = nn.Parameter(torch.zeros(d))
            self.emo2 = nn.Linear(d, d)
        self.token = nn.Embedding(vocab, d)
        self.position = nn.Embedding(max_len, d)
        self.emo = nn.Sequential(
            nn.Linear(2, d), 
            nn.SiLU(), 
            nn.Linear(d, d))

        #Pre-LN normalization of data, forcing token vecotrs to have around the same weight
        self.lin1 = nn.LayerNorm(d)
        self.attn1 = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff1 = nn.Sequential(
            nn.Linear(d, 4 * d), 
            nn.GELU(), 
            nn.Linear(4 * d, d))

        self.lin2 = nn.LayerNorm(d)
        self.attn2 = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff2 = nn.Sequential(
                    nn.Linear(d, 4 * d), 
                    nn.GELU(), 
                    nn.Linear(4 * d, d))
        
        self.output = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, tokens, label, drop=None):
        # drop: (B,) True = hide the mood for that clip (guided model only)
        L = tokens.shape[1]
        x = self.token(tokens) + self.position(torch.arange(L, device=tokens.device))

        m = self.emo(label)
        if drop is not None and self.guided:
            m = torch.where(drop[:, None], self.null_mood.expand_as(m), m)
        x = x + m.unsqueeze(1)
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tokens.device), diagonal=1)

       #layer 1
        h = self.lin1(x)
        a, _ = self.attn1(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.ff1(x)

        # layer 2
        if self.guided:
            x = x + self.emo2(m).unsqueeze(1)            # the mood again, so the second layer sees it directly
        h = self.lin2(x)
        a, _ = self.attn2(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.ff2(x)

        return self.head(self.output(x))


# ====================================================================================================== version 4
# Version 3 used the mood well, but its clips sound like a texture instead of notes: one note's overtones are spread
# over several of the 8 frequency rows, and 2 layers were not enough to keep the rows in tune and lined up in time
# (validation loss 3.2 of a possible 6.2: it is still very unsure). What changed:
#   - any number of layers (default 4), each one gets the mood again
#   - LayerNorm before the feed-forward part too (version 2/3 skipped it; deeper models are unstable without it)
#   - each id is told which time step, frequency row and round it belongs to, so it does not have to work that
#     out from a plain position number
#   - dropout, so the bigger model does not just memorise the training clips
#   - a cache for generation: earlier ids are not recomputed for every new id, so generating is much faster
class Block(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.heads = heads
        self.mood = nn.Linear(d, d)
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, m, cache=None):
        # cache: None for a whole sequence (training), or a dict that keeps this layer's keys and values (generation)
        B, L, d = x.shape
        x = x + self.mood(m).unsqueeze(1)
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        if cache is not None:
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], dim=2), torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
            assert L == 1 or k.shape[2] == L, "with a cache, feed one new id at a time"
        # no dropout inside attention: on CPU it switches to a slow path and doubles the training time
        a = F.scaled_dot_product_attention(q, k, v, is_causal=(k.shape[2] == L and L > 1))
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, L, d)))
        return x + self.ff(self.ln2(x))


class MusicTransformerV4(nn.Module):
    guided = True

    def __init__(self, vocab, d, heads, max_len, layers=4, dropout=0.1):
        super().__init__()
        self.register_buffer("n_heads", torch.tensor(heads))       # saved with the weights so generate.py can rebuild it
        self.token = nn.Embedding(vocab, d)
        self.position = nn.Embedding(max_len, d)
        self.time = nn.Embedding(GRID[1], d)                        # which time step (0..43)
        self.row = nn.Embedding(GRID[0], d)                         # which frequency row (0..7)
        self.round = nn.Embedding(ROUNDS, d)                        # coarse id or fine correction
        self.emo = nn.Sequential(nn.Linear(2, d), nn.SiLU(), nn.Linear(d, d))
        self.null_mood = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList([Block(d, heads, dropout) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.output = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def new_cache(self):
        return [dict() for _ in self.blocks]

    def forward(self, tokens, label, drop=None, cache=None):
        # input i predicts clip id i, so its time / row / round are those of clip id i (layout: time, row, round)
        L = tokens.shape[1]
        start = cache[0]["k"].shape[2] if cache is not None and "k" in cache[0] else 0
        p = torch.arange(start, start + L, device=tokens.device)
        x = (self.token(tokens) + self.position(p) + self.time((p // (GRID[0] * ROUNDS)).clamp(max=GRID[1] - 1))
             + self.row((p // ROUNDS) % GRID[0]) + self.round(p % ROUNDS))

        m = self.emo(label)
        if drop is not None:
            m = torch.where(drop[:, None], self.null_mood.expand_as(m), m)
        x = self.drop(x + m.unsqueeze(1))
        for i, block in enumerate(self.blocks):
            x = block(x, m, None if cache is None else cache[i])
        return self.head(self.output(x))


START_ID = 1023                                     # one past the last real id
offset = torch.arange(2) * 512                      # [0, 512], one range per round


def load_tokens(path, with_unlabeled=False):
    # with_unlabeled=True (version 4 training): also the clips with no mood label (the first 15 s of every song),
    # returned with has_label False so they are always trained as "no mood". They still teach what music looks like.
    d = torch.load(path)
    keep = torch.ones_like(d["has_label"]).bool() if with_unlabeled else d["has_label"].bool()
    idx = d["idx"][keep].long()                     # (N, 8, 44, 2), values 0..511
    label = d["label"][keep].float()                # (N, 2)

    ids = idx + offset.reshape(1, 1, 1, 2)          # each round gets its own range: 0..1023
    N = ids.shape[0]
    seq = ids.permute(0, 2, 1, 3).reshape(N, -1)    # time-major: (N, 704)
    start = torch.full((N, 1), START_ID, dtype=seq.dtype)
    tokens = torch.cat([start, seq], dim=1)         # (N, 705)
    if with_unlabeled:
        return TensorDataset(tokens, label, d["has_label"][keep].bool())
    return TensorDataset(tokens, label)


if __name__ == "__main__":
   import argparse
   import os
   ap = argparse.ArgumentParser()
   ap.add_argument("--epochs", type=int, default=10)
   ap.add_argument("--batch", type=int, default=16)
   ap.add_argument("--lr", type=float, default=3e-4)
   ap.add_argument("--drop", type=float, default=0.15)       # share of labeled clips trained with the mood hidden
   ap.add_argument("--layers", type=int, default=4)
   ap.add_argument("--dropout", type=float, default=0.1)
   ap.add_argument("--warmup", type=int, default=1000)       # steps of slowly rising learning rate at the start
   ap.add_argument("--labeled_only", action="store_true")    # skip the clips that have no mood label
   ap.add_argument("--overwrite", action="store_true")       # allow replacing an existing transformer.pt
   args = ap.parse_args()

   if os.path.exists("transformer.pt") and not args.overwrite:
        raise SystemExit("transformer.pt already exists. Rename it (for example to transformer_v3.pt) or pass --overwrite.")

   device = "cuda" if torch.cuda.is_available() else "cpu"
   train_data = load_tokens("tokens_train.pt", with_unlabeled=not args.labeled_only)
   valid_data = load_tokens("tokens_val.pt")                 # labeled only, so the numbers compare with older models
   if args.labeled_only:
        train_data = TensorDataset(*train_data.tensors, torch.ones(len(train_data), dtype=torch.bool))
   train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, drop_last=True)
   valid_loader = DataLoader(valid_data, batch_size=args.batch, shuffle=False)
   n_lab = int(train_data.tensors[2].sum())
   print(f"{len(train_data)} train clips ({n_lab} with a mood label), {len(valid_data)} validation clips, "
         f"device {device}", flush=True)

   transformer = MusicTransformerV4(vocab=1024, d=256, heads=4, max_len=705, layers=args.layers,
                                    dropout=args.dropout).to(device)
   print(f"{args.layers} layers, {sum(p.numel() for p in transformer.parameters()) / 1e6:.1f}M weights", flush=True)
   optimizer = optim.AdamW(transformer.parameters(), lr=args.lr, weight_decay=0.01)
   total = args.epochs * len(train_loader)
   scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / args.warmup) *
                                           0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

   best = float("inf")
   for epoch in range(args.epochs):
        transformer.train()
        for step, (tokens, label, has_label) in enumerate(train_loader):
            tokens, label, has_label = tokens.to(device), label.to(device), has_label.to(device)
            # these clips learn "no mood": a random share of the labeled ones, and every unlabeled one
            drop = (torch.rand(tokens.shape[0], device=device) < args.drop) | ~has_label

            logits = transformer(tokens[:, :-1], label, drop)
            loss = F.cross_entropy(logits.reshape(-1, 1024), tokens[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)   # one bad batch cannot wreck a deeper model
            optimizer.step()
            scheduler.step()

            if step % 50 == 0:
                print(f"epoch {epoch}  step {step}  loss {loss.item():.4f}", flush=True)

        # validation: loss with the real mood, with a wrong mood, and with no mood.
        # the bigger the gap between real and wrong, the more the model actually uses the mood
        transformer.eval()
        tot = {"real": 0.0, "wrong": 0.0, "none": 0.0}
        count = 0
        with torch.no_grad():
            for tokens, label in valid_loader:
                tokens, label = tokens.to(device), label.to(device)
                inp, tgt = tokens[:, :-1], tokens[:, 1:].reshape(-1)
                b = tokens.shape[0]
                hide = torch.ones(b, dtype=torch.bool, device=device)
                for k, (lab, d) in {"real": (label, None), "wrong": (label.roll(1, 0), None),
                                    "none": (label, hide)}.items():
                    tot[k] += F.cross_entropy(transformer(inp, lab, d).reshape(-1, 1024), tgt).item() * b
                count += b
        real, wrong, none = (tot[k] / count for k in ("real", "wrong", "none"))
        print(f"epoch {epoch}  VALIDATION loss {real:.4f} | wrong mood {wrong:.4f} (gap {wrong - real:+.4f}) | "
              f"no mood {none:.4f} (gap {none - real:+.4f})", flush=True)

        if real < best:
            best = real
            torch.save(transformer.state_dict(), "transformer.pt")
            print("saved transformer.pt", flush=True)
