import sys
import time
import json
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Paths and global training setup
# ---------------------------------------------------------

BASE_DIR = Path("/scratch/am15577/projects/ML Music Gen")
TOK_DIR = BASE_DIR / "tokenized"

TRAIN_PATH = TOK_DIR / "train.npy"
VAL_PATH = TOK_DIR / "val.npy"
VOCAB_PATH = TOK_DIR / "vocab.json"

RESULTS_CSV = BASE_DIR / "scaling_results.csv"

BLOCK_SIZE = 256          # context length
BATCH_SIZE = 64           # sequences per batch
TOKENS_PER_STEP = BLOCK_SIZE * BATCH_SIZE

LR = 3e-4
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
DROPOUT = 0.1

WARMUP_FRACTION = 0.05    # first 5 percent of steps are warmup
EVAL_EVERY = 200          # steps between val evals during training


# ---------------------------------------------------------
# Model size configs
# (these do not need to be exactly 1M/5M/20M/etc, just close)
# ---------------------------------------------------------

MODEL_CONFIGS = {
    "tiny":   dict(n_layers=2,  n_heads=4,  d_model=128),  # ~1M
    "small":  dict(n_layers=4,  n_heads=8,  d_model=256),  # ~5M
    "medium": dict(n_layers=6,  n_heads=6,  d_model=384),  # ~20M
    "large":  dict(n_layers=8,  n_heads=8,  d_model=512),  # ~50M
    "xl":     dict(n_layers=12, n_heads=12, d_model=768),  # ~100M
}


# ---------------------------------------------------------
# Data helpers
# ---------------------------------------------------------

class TokenDataset:
    """
    Simple streaming style dataset on a flat array of token ids.
    """

    def __init__(self, path: Path):
        self.tokens = np.load(path, mmap_mode="r")
        self.effective_tokens = self.tokens.shape[0] - 1

    def num_steps_per_epoch(self) -> int:
        return self.effective_tokens // TOKENS_PER_STEP

    def get_batch(self, step_idx: int, device: torch.device):
        assert 0 <= step_idx < self.num_steps_per_epoch()
        base_index = step_idx * TOKENS_PER_STEP
        B = BATCH_SIZE
        T = BLOCK_SIZE

        x = np.empty((B, T), dtype=np.int64)
        y = np.empty((B, T), dtype=np.int64)

        for i in range(B):
            start = base_index + i * T
            end = start + T
            x[i] = self.tokens[start:end]
            y[i] = self.tokens[start + 1 : end + 1]

        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        return x, y


# ---------------------------------------------------------
# Transformer model
# ---------------------------------------------------------

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
        self.d_model = d_model
        self.block_size = BLOCK_SIZE

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device, dtype=torch.long).unsqueeze(0)

        tok = self.token_emb(idx)
        pos = self.pos_emb(pos)
        x = self.drop(tok + pos)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                reduction="mean",
            )
        return logits, loss


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def load_vocab_size():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return len(vocab["stoi"])


def create_model(model_name: str, vocab_size: int, device: torch.device):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model name {model_name}. Choose from {list(MODEL_CONFIGS.keys())}")
    cfg = MODEL_CONFIGS[model_name]
    print(f"Model config for {model_name}: {cfg}")
    model = GPT(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        dropout=DROPOUT,
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model {model_name} has {n_params:,} trainable parameters")
    return model, n_params


def create_optimizer(model: nn.Module):
    return torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )


def create_scheduler(optimizer, total_steps: int):
    warmup_steps = int(WARMUP_FRACTION * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 1e-8), 1.0)
        return 0.5 * (1.0 + float(torch.cos(torch.tensor(progress * 3.1415926535))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model: nn.Module, dataset: TokenDataset, device: torch.device):
    model.eval()
    steps = dataset.num_steps_per_epoch()
    total_loss = 0.0
    for step_idx in range(steps):
        x, y = dataset.get_batch(step_idx, device)
        _, loss = model(x, y)
        total_loss += loss.item()
    avg_loss = total_loss / max(1, steps)
    model.train()
    return avg_loss


def append_results_csv(row: dict):
    header = [
        "model_name",
        "mode",
        "n_params",
        "steps_per_epoch",
        "total_train_tokens",
        "final_val_loss",
        "wall_clock_seconds",
        "gpu_mem_start_mb",
        "gpu_mem_end_mb",
        "gpu_mem_peak_mb",
    ]
    file_exists = RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write(",".join(header) + "\n")
        vals = [
            str(row.get(col, "")) for col in header
        ]
        f.write(",".join(vals) + "\n")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python train_transformers.py <model_name>            # train 1 epoch")
        print("  python train_transformers.py <model_name> profile   # profile GPU only")
        print(f"Available models: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    model_name = sys.argv[1]
    mode = "train"
    if len(sys.argv) >= 3 and sys.argv[2] == "profile":
        mode = "profile"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  mode: {mode}")

    vocab_size = load_vocab_size()
    print(f"Vocab size: {vocab_size}")

    model, n_params = create_model(model_name, vocab_size, device)

    gpu_mem_before = None
    gpu_mem_after = None
    gpu_mem_peak = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        gpu_mem_before = torch.cuda.memory_allocated() / 1024**2
        print(f"GPU memory at start: {gpu_mem_before:.2f} MB")

    if mode == "profile":
        # one dummy batch to estimate GPU footprint
        B = BATCH_SIZE
        T = BLOCK_SIZE
        x = torch.randint(0, vocab_size, (B, T), device=device, dtype=torch.long)
        y = torch.randint(0, vocab_size, (B, T), device=device, dtype=torch.long)
        _, loss = model(x, y)
        loss.backward()  # include activations + grads
        if device.type == "cuda":
            gpu_mem_after = torch.cuda.memory_allocated() / 1024**2
            gpu_mem_peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[profile] loss={loss.item():.4f}")
        print(f"[profile] n_params={n_params:,}")
        if device.type == "cuda":
            print(f"[profile] GPU mem peak: {gpu_mem_peak:.2f} MB")

        row = dict(
            model_name=model_name,
            mode="profile",
            n_params=n_params,
            steps_per_epoch=0,
            total_train_tokens=0,
            final_val_loss="",
            wall_clock_seconds=0.0,
            gpu_mem_start_mb=round(gpu_mem_before or 0.0, 3),
            gpu_mem_end_mb=round(gpu_mem_after or 0.0, 3),
            gpu_mem_peak_mb=round(gpu_mem_peak or 0.0, 3),
        )
        append_results_csv(row)
        return

    # mode == train
    print("Loading datasets (memmap)...")
    train_ds = TokenDataset(TRAIN_PATH)
    val_ds = TokenDataset(VAL_PATH)

    steps_per_epoch = train_ds.num_steps_per_epoch()
    print(f"Train effective tokens: {train_ds.effective_tokens:,}")
    print(f"Tokens per step:        {TOKENS_PER_STEP:,}")
    print(f"Steps per epoch:        {steps_per_epoch:,}")

    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer, total_steps=steps_per_epoch)

    print("Starting training for exactly 1 epoch")
    start_time = time.time()

    global_step = 0
    for step_idx in range(steps_per_epoch):
        x, y = train_ds.get_batch(step_idx, device)
        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        global_step += 1

        if global_step % 50 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"Step {global_step}/{steps_per_epoch}  loss {loss.item():.4f}  lr {lr:.6f}")

        if global_step % EVAL_EVERY == 0 or global_step == steps_per_epoch:
            val_loss = evaluate(model, val_ds, device)
            print(f"[eval] step {global_step}  val_loss {val_loss:.4f}")

    end_time = time.time()
    elapsed = end_time - start_time

    final_val_loss = evaluate(model, val_ds, device)

    if device.type == "cuda":
        gpu_mem_after = torch.cuda.memory_allocated() / 1024**2
        gpu_mem_peak = torch.cuda.max_memory_allocated() / 1024**2

    print("===========================================")
    print(f"Model name:          {model_name}")
    print(f"Parameters:          {n_params:,}")
    print(f"Final val loss:      {final_val_loss:.4f}")
    print(f"Steps in 1 epoch:    {steps_per_epoch}")
    print(f"Total train tokens:  {steps_per_epoch * TOKENS_PER_STEP:,}")
    print(f"Wall clock seconds:  {elapsed:.1f}")
    print(f"Wall clock minutes:  {elapsed / 60.0:.1f}")
    if device.type == "cuda":
        print(f"GPU memory start:     {gpu_mem_before:.2f} MB")
        print(f"GPU memory end:       {gpu_mem_after:.2f} MB")
        print(f"GPU memory peak:      {gpu_mem_peak:.2f} MB")
    else:
        print("GPU memory tracking unavailable (CPU).")
    print("===========================================")

    row = dict(
        model_name=model_name,
        mode="train",
        n_params=n_params,
        steps_per_epoch=steps_per_epoch,
        total_train_tokens=steps_per_epoch * TOKENS_PER_STEP,
        final_val_loss=round(float(final_val_loss), 6),
        wall_clock_seconds=round(float(elapsed), 3),
        gpu_mem_start_mb=round(gpu_mem_before or 0.0, 3),
        gpu_mem_end_mb=round(gpu_mem_after or 0.0, 3),
        gpu_mem_peak_mb=round(gpu_mem_peak or 0.0, 3),
    )
    append_results_csv(row)


if __name__ == "__main__":
    main()