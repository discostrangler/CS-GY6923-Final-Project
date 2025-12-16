import sys
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path("/scratch/am15577/projects/ML Music Gen")
TOK_DIR = PROJECT_ROOT / "tokenized"
TRAIN_PATH = TOK_DIR / "train.npy"
VAL_PATH = TOK_DIR / "val.npy"
VOCAB_PATH = TOK_DIR / "vocab.json"

CHECKPOINT_DIR = PROJECT_ROOT / "transformer" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BLOCK_SIZE = 256
BATCH_SIZE = 64
TOKENS_PER_STEP = BLOCK_SIZE * BATCH_SIZE

LR = 3e-4
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
DROPOUT = 0.1

WARMUP_FRACTION = 0.05
EVAL_EVERY = 200


MODEL_CONFIG_XL = dict(n_layers=12, n_heads=12, d_model=768)


class TokenDataset:
    def __init__(self, path: Path):
        self.tokens = np.load(path, mmap_mode="r")
        self.effective_tokens = self.tokens.shape[0] - 1

    def num_steps_per_epoch(self) -> int:
        return self.effective_tokens // TOKENS_PER_STEP

    def get_batch(self, step_idx: int, device: torch.device):
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
        return 0.5 * (1.0 + float(torch.cos(torch.tensor(progress * 3.1415926535))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, dataset, device):
    model.eval()
    steps = dataset.num_steps_per_epoch()
    total_loss = 0.0
    for step_idx in range(steps):
        x, y = dataset.get_batch(step_idx, device)
        _, loss = model(x, y)
        total_loss += loss.item()
    avg = total_loss / max(1, steps)
    model.train()
    return avg


def main():
    # optional arg: number of epochs
    if len(sys.argv) >= 2:
        num_epochs = int(sys.argv[1])
    else:
        num_epochs = 3  # default

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}, training trans_xl for {num_epochs} epochs")

    vocab, vocab_size = load_vocab()
    print(f"Vocab size: {vocab_size}")

    cfg = MODEL_CONFIG_XL
    model = GPT(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        dropout=DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trans_xl parameters: {n_params:,}")

    train_ds = TokenDataset(TRAIN_PATH)
    val_ds = TokenDataset(VAL_PATH)
    steps_per_epoch = train_ds.num_steps_per_epoch()
    print(f"Steps per epoch: {steps_per_epoch}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    scheduler = create_scheduler(optimizer, total_steps=steps_per_epoch * num_epochs)

    best_val = float("inf")
    best_path = CHECKPOINT_DIR / "trans_xl_best.pt"

    start_time = time.time()
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
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
                print(f"Step {global_step} loss {loss.item():.4f} lr {lr:.6f}")

            if global_step % EVAL_EVERY == 0 or step_idx == steps_per_epoch - 1:
                val_loss = evaluate(model, val_ds, device)
                print(f"[eval] epoch {epoch} step {global_step} val_loss {val_loss:.4f}")
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
                    print(f"Saved new best checkpoint to {best_path} (val_loss {best_val:.4f})")

    elapsed = time.time() - start_time
    print(f"Finished training in {elapsed/3600:.2f} hours")
    print(f"Best val loss seen: {best_val:.4f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()