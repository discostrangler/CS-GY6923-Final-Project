import sys
import time
import json
from pathlib import Path

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

CHECKPOINT_DIR = BASE_DIR / "transformer" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BLOCK_SIZE = 256          # context length
BATCH_SIZE = 64           # sequences per batch
TOKENS_PER_STEP = BLOCK_SIZE * BATCH_SIZE

LR = 3e-4
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
DROPOUT = 0.1

WARMUP_FRACTION = 0.05    # first 5 percent of steps are warmup
EVAL_EVERY = 200          # steps between val evals during training

# We only care about xl here, but keep config explicitly
MODEL_CONFIG_XL = dict(n_layers=12, n_heads=12, d_model=768)


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
            y[i] = self.tokens[start + 1: end + 1]

        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        return x, y


# ---------------------------------------------------------
# Transformer model (same as before)
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
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)).view(
                1, 1, BLOCK_SIZE, BLOCK_SIZE
            ),
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

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device, dtype=torch.long).unsqueeze(0)

        tok = self.token_emb(idx)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok + pos_emb)

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

def load_vocab():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab, len(vocab["stoi"])


def create_scheduler(optimizer, total_steps: int):
    warmup_steps = int(WARMUP_FRACTION * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 1e-8), 1.0)
        # cosine decay
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


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    # Optional CLI arg: number of epochs
    if len(sys.argv) >= 2:
        num_epochs = int(sys.argv[1])
    else:
        num_epochs = 5  # default

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  epochs: {num_epochs}", flush=True)

    vocab, vocab_size = load_vocab()
    print(f"Vocab size: {vocab_size}", flush=True)

    cfg = MODEL_CONFIG_XL
    model = GPT(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        dropout=DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model xl has {n_params:,} trainable parameters", flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        gpu_mem_before = torch.cuda.memory_allocated() / 1024**2
        print(f"GPU memory at start: {gpu_mem_before:.2f} MB", flush=True)
    else:
        gpu_mem_before = None

    print("Loading datasets (memmap)...", flush=True)
    train_ds = TokenDataset(TRAIN_PATH)
    val_ds = TokenDataset(VAL_PATH)

    steps_per_epoch = train_ds.num_steps_per_epoch()
    print(f"Train effective tokens: {train_ds.effective_tokens:,}", flush=True)
    print(f"Tokens per step:        {TOKENS_PER_STEP:,}", flush=True)
    print(f"Steps per epoch:        {steps_per_epoch:,}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = steps_per_epoch * num_epochs
    scheduler = create_scheduler(optimizer, total_steps=total_steps)

    best_val = float("inf")
    best_path = CHECKPOINT_DIR / "trans_xl_best.pt"

    print("Starting training", flush=True)
    start_time = time.time()

    global_step = 0
    LOG_EVERY = 50
    BAR_WIDTH = 30

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}", flush=True)
        for step_idx in range(steps_per_epoch):
            x, y = train_ds.get_batch(step_idx, device)
            logits, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % LOG_EVERY == 0:
                lr = scheduler.get_last_lr()[0]
                frac = (step_idx + 1) / steps_per_epoch
                percent = 100.0 * frac
                filled = int(BAR_WIDTH * frac)
                bar = "#" * filled + "." * (BAR_WIDTH - filled)
                print(
                    f"Step {global_step:6d}  [{bar}] {percent:5.1f}%  "
                    f"loss {loss.item():.4f}  lr {lr:.6f}",
                    flush=True,
                )

            if global_step % EVAL_EVERY == 0 or (
                epoch == num_epochs and step_idx == steps_per_epoch - 1
            ):
                val_loss = evaluate(model, val_ds, device)
                print(
                    f"[eval] epoch {epoch} step {global_step}  val_loss {val_loss:.4f}",
                    flush=True,
                )
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "cfg": cfg,
                            "vocab": vocab,
                        },
                        best_path,
                    )
                    print(
                        f"Saved new best checkpoint to {best_path} "
                        f"(val_loss {best_val:.4f})",
                        flush=True,
                    )

    end_time = time.time()
    elapsed = end_time - start_time

    if device.type == "cuda":
        gpu_mem_after = torch.cuda.memory_allocated() / 1024**2
        gpu_mem_peak = torch.cuda.max_memory_allocated() / 1024**2
    else:
        gpu_mem_after = None
        gpu_mem_peak = None

    print("===========================================", flush=True)
    print(f"Finished training in {elapsed/3600:.2f} hours", flush=True)
    print(f"Best val loss:      {best_val:.4f}", flush=True)
    print(f"Best checkpoint:    {best_path}", flush=True)
    if device.type == "cuda":
        print(f"GPU memory start:   {gpu_mem_before:.2f} MB", flush=True)
        print(f"GPU memory end:     {gpu_mem_after:.2f} MB", flush=True)
        print(f"GPU memory peak:    {gpu_mem_peak:.2f} MB", flush=True)
    print("===========================================", flush=True)


if __name__ == "__main__":
    main()