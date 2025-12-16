import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_DIR = Path("/scratch/am15577/projects/ML Music Gen")
TOK_DIR = BASE_DIR / "tokenized"
VOCAB_PATH = TOK_DIR / "vocab.json"
TRAIN_PATH = TOK_DIR / "train.npy"

CKPT_PATH_DEFAULT = BASE_DIR / "transformer" / "checkpoints" / "trans_xl_best.pt"
OUT_DIR_DEFAULT = BASE_DIR / "transformer" / "samples"


# ---------------------------------------------------------
# Model definition (must match training)
# ---------------------------------------------------------
BLOCK_SIZE = 256
DROPOUT = 0.0  # inference


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)).view(1, 1, BLOCK_SIZE, BLOCK_SIZE),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.out_proj(y))
        return y


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        d_ff = 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device, dtype=torch.long).unsqueeze(0)

        tok = self.token_emb(idx)
        pos = self.pos_emb(pos)
        x = self.drop(tok + pos)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


# ---------------------------------------------------------
# Vocab helpers
# ---------------------------------------------------------
def load_vocab():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    stoi = vocab["stoi"]
    itos = vocab["itos"]
    return stoi, itos


def encode(text: str, stoi: dict, unk_id: int):
    return [stoi.get(ch, unk_id) for ch in text]


def decode(ids, itos):
    if isinstance(itos, list):
        return "".join(itos[i] for i in ids)
    return "".join(itos[str(i)] for i in ids)


# ---------------------------------------------------------
# Sampling
# ---------------------------------------------------------
@torch.no_grad()
def sample(model, idx, max_new_tokens: int, temperature: float, top_k: Optional[int]):
    device = next(model.parameters()).device
    idx = idx.to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]
        logits = model(idx_cond)[:, -1, :]  # (B,V)
        logits = logits / max(1e-8, temperature)

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (B,1)
        idx = torch.cat([idx, next_id], dim=1)

    return idx


# ---------------------------------------------------------
# Prefix selection from training tokens
# ---------------------------------------------------------
def pick_prefix_from_tokens(train_path: Path, itos, length_chars: int = 256, seed: int = 0):
    rng = np.random.default_rng(seed)
    tokens = np.load(train_path, mmap_mode="r")
    n = tokens.shape[0]

    start = int(rng.integers(0, max(1, n - length_chars - 1)))
    window = tokens[start : start + length_chars].astype(np.int64).tolist()

    txt = decode(window, itos)

    cut = txt.find("X:")
    if cut != -1:
        txt = txt[cut:]

    return txt[:length_chars]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(CKPT_PATH_DEFAULT))
    ap.add_argument("--out_dir", type=str, default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--n_uncond", type=int, default=10)
    ap.add_argument("--n_cond", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    stoi, itos = load_vocab()
    vocab_size = len(stoi)
    unk_id = stoi.get("<unk>", 3)
    print("Vocab size:", vocab_size, "unk_id:", unk_id, flush=True)

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["cfg"]

    model = GPT(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        dropout=DROPOUT,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Unconditional seed
    uncond_seed = "X:1\nT:Generated\nM:4/4\nK:C\n"
    uncond_ids = encode(uncond_seed, stoi, unk_id)

    # Conditional seed from data
    prefix_text = pick_prefix_from_tokens(TRAIN_PATH, itos, length_chars=256, seed=args.seed)
    cond_ids = encode(prefix_text, stoi, unk_id)

    report_lines = []
    report_lines.append("=== Sampling settings ===\n")
    report_lines.append(f"checkpoint: {args.ckpt}\n")
    report_lines.append(f"temperature: {args.temperature}\n")
    report_lines.append(f"top_k: {args.top_k}\n")
    report_lines.append(f"max_new_tokens: {args.max_new_tokens}\n\n")
    report_lines.append("=== Conditional prefix used (first 256 chars) ===\n")
    report_lines.append(prefix_text + "\n\n")

    def write_sample(kind: str, i: int, text: str):
        path = out_dir / f"{kind}_{i:02d}.abc"
        path.write_text(text, encoding="utf-8")
        return path

    # Unconditional samples
    print(f"Generating {args.n_uncond} unconditional samples...", flush=True)
    for i in range(args.n_uncond):
        x = torch.tensor(uncond_ids, dtype=torch.long, device=device)[None, :]
        y = sample(
            model,
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )[0].tolist()
        text = decode(y, itos)

        out_path = write_sample("uncond", i, text)
        is_valid = ("X:" in text) and ("K:" in text)

        report_lines.append(f"----- UNCONDITIONAL {i} (valid={is_valid}) file={out_path.name} -----\n")
        report_lines.append(text[:2000] + ("\n...\n\n" if len(text) > 2000 else "\n\n"))

    # Conditional samples
    print(f"Generating {args.n_cond} conditional samples...", flush=True)
    for i in range(args.n_cond):
        x = torch.tensor(cond_ids, dtype=torch.long, device=device)[None, :]
        y = sample(
            model,
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )[0].tolist()
        text = decode(y, itos)

        out_path = write_sample("cond", i, text)
        is_valid = ("X:" in text) and ("K:" in text)

        report_lines.append(f"----- CONDITIONAL {i} (valid={is_valid}) file={out_path.name} -----\n")
        report_lines.append(text[:2000] + ("\n...\n\n" if len(text) > 2000 else "\n\n"))

    report_path = out_dir / "samples_report.txt"
    report_path.write_text("".join(report_lines), encoding="utf-8")

    print("Wrote samples to:", out_dir, flush=True)
    print("Wrote report to:", report_path, flush=True)
    print("Next: play .abc in an online ABC player, or convert to MIDI locally.", flush=True)


if __name__ == "__main__":
    main()