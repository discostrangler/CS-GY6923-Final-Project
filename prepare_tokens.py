import json
from pathlib import Path

import numpy as np

BASE_DIR = Path("/scratch/am15577/projects/ML Music Gen")
TOK_DIR = BASE_DIR / "tokenized"

VOCAB_PATH = TOK_DIR / "vocab.json"
CORPUS_PATH = TOK_DIR / "all_abc_clean.txt"

# We only need enough tokens so that train >= 100M
TOTAL_TARGET = 120_000_000  # max characters to use

TRAIN_FRAC = 0.98
VAL_FRAC = 0.01
TEST_FRAC = 0.01


def load_vocab(path: Path):
    print(f"Loading vocab from {path}", flush=True)
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    stoi = vocab["stoi"]
    unk_id = stoi["<unk>"]
    print(f"Vocab size: {len(stoi)}; unk_id={unk_id}", flush=True)
    return stoi, unk_id


def stream_text_to_ids(path: Path, stoi: dict, unk_id: int, total_target: int):
    """
    Stream characters from the corpus file and convert to ids.
    Stop once we have total_target characters or EOF.
    This avoids loading the whole file into memory.
    """
    print(f"Streaming corpus from {path}", flush=True)
    ids = np.empty(total_target, dtype=np.uint16)
    get = stoi.get

    filled = 0
    chunk_size = 1_000_000

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            for ch in chunk:
                if filled >= total_target:
                    break
                ids[filled] = get(ch, unk_id)
                filled += 1
            if filled >= total_target:
                break
            if filled and filled % 10_000_000 == 0:
                print(f"  processed {filled:,} characters", flush=True)

    print(f"Total characters converted: {filled:,}", flush=True)

    if filled == 0:
        raise RuntimeError("Corpus appears to be empty or unreadable")

    # If we did not fill the entire array, trim it
    if filled < total_target:
        print(
            f"Corpus ended early; using {filled:,} characters "
            f"instead of target {total_target:,}",
            flush=True,
        )
        ids = ids[:filled]

    return ids


def make_splits(ids: np.ndarray):
    n_total = len(ids)
    print(f"Total ids: {n_total:,}", flush=True)

    n_train = int(n_total * TRAIN_FRAC)
    n_val = int(n_total * VAL_FRAC)
    n_test = n_total - n_train - n_val

    if n_train < 100_000_000:
        print(
            f"Warning: train split has only {n_train:,} tokens, "
            f"which is less than 100M required by the spec.",
            flush=True,
        )
    else:
        print(f"Train tokens: {n_train:,} (>= 100M requirement)", flush=True)

    print(f"Val tokens:   {n_val:,}", flush=True)
    print(f"Test tokens:  {n_test:,}", flush=True)

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]

    return train_ids, val_ids, test_ids


def save_splits(train_ids, val_ids, test_ids, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.npy"
    val_path = out_dir / "val.npy"
    test_path = out_dir / "test.npy"

    print(f"Saving train to {train_path}", flush=True)
    np.save(train_path, train_ids)

    print(f"Saving val to {val_path}", flush=True)
    np.save(val_path, val_ids)

    print(f"Saving test to {test_path}", flush=True)
    np.save(test_path, test_ids)

    meta = {
        "dtype": str(train_ids.dtype),
        "train_tokens": int(len(train_ids)),
        "val_tokens": int(len(val_ids)),
        "test_tokens": int(len(test_ids)),
        "total_tokens": int(len(train_ids) + len(val_ids) + len(test_ids)),
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC,
        "total_target": TOTAL_TARGET,
    }

    meta_path = out_dir / "meta.json"
    print(f"Saving meta to {meta_path}", flush=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Done saving splits.", flush=True)


if __name__ == "__main__":
    stoi, unk_id = load_vocab(VOCAB_PATH)
    ids = stream_text_to_ids(CORPUS_PATH, stoi, unk_id, TOTAL_TARGET)
    train_ids, val_ids, test_ids = make_splits(ids)
    save_splits(train_ids, val_ids, test_ids, TOK_DIR)