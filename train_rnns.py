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
# RNN model size configs (matched to transformer params)
# ---------------------------------------------------------

MODEL_CONFIGS = {
    # approx params: 4.43e5 transformer vs 4.29e5 RNN
    "tiny":   dict(n_layers=2, d_model=160),
    # approx params: 3.25e6 transformer vs 3.26e6 RNN
    "small":  dict(n_layers=2, d_model=448),
    # approx params: 1.08e7 transformer vs 1.08e7 RNN
    "medium": dict(n_layers=2, d_model=816),
    # approx params: 2.54e7 transformer vs 2.53e7 RNN
    "large":  dict(n_layers=3, d_model=1024),
    # approx params: 8.53e7 transformer vs 8.54e7 RNN
    "xl":     dict(n_layers=4, d_model=1632),
}


# ---------------------------------------------------------
# Data helpers
# ---------------------------------------------------------

class TokenDataset:
    """
    Simple streaming style dataset on a flat array of token ids.
    Same as transformer script.
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
# RNN language model
# ---------------------------------------------------------

class RNNLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, dropout: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        self.token_emb = nn.Embedding(vocab_size, d_model)

        self.rnn = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # tie weights for fairness with transformer setup
        self.head.weight = self.token_emb.weight

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.token_emb.weight, -0.1, 0.1)
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, idx, targets=None):
        # idx: (B, T)
        x = self.token_emb(idx)             # (B, T, C)
        out, _ = self.rnn(x)                # (B, T, C)
        out = self.ln_f(out)
        logits = self.head(out)             # (B, T, V)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, V),
                targets.view(B * T),
                reduction="mean",
            )
        return logits, loss


# ---------------------------------------------------------
# Helpers shared with transformer script
# ---------------------------------------------------------

def load_vocab_size():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return len(vocab["stoi"])


def create_model(model_name: str, vocab_size: int, device: torch.device):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model name {model_name}. Choose from {list(MODEL_CONFIGS.keys())}")
    cfg = MODEL_CONFIGS[model_name]
    print(f"RNN config for {model_name}: {cfg}")
    model = RNNLM(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        dropout=DROPOUT,
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"RNN {model_name} has {n_params:,} trainable parameters")
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
        vals = [str(row.get(col, "")) for col in header]
        f.write(",".join(vals) + "\n")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python train_rnns.py <model_name>            # train 1 epoch")
        print("  python train_rnns.py <model_name> profile   # profile GPU only")
        print(f"Available RNN models: {list(MODEL_CONFIGS.keys())}")
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
            model_name=f"rnn_{model_name}",
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
    print(f"RNN model name:       {model_name}")
    print(f"Parameters:           {n_params:,}")
    print(f"Final val loss:       {final_val_loss:.4f}")
    print(f"Steps in 1 epoch:     {steps_per_epoch}")
    print(f"Total train tokens:   {steps_per_epoch * TOKENS_PER_STEP:,}")
    print(f"Wall clock seconds:   {elapsed:.1f}")
    print(f"Wall clock minutes:   {elapsed / 60.0:.1f}")
    if device.type == "cuda":
        print(f"GPU memory start:     {gpu_mem_before:.2f} MB")
        print(f"GPU memory end:       {gpu_mem_after:.2f} MB")
        print(f"GPU memory peak:      {gpu_mem_peak:.2f} MB")
    else:
        print("GPU memory tracking unavailable (CPU).")
    print("===========================================")

    row = dict(
        model_name=f"rnn_{model_name}",
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