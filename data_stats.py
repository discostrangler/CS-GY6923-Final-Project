# save as analyze_dataset.py in /scratch/am15577/projects/ML Music Gen

import json
from pathlib import Path

import numpy as np

BASE_DIR = Path("/scratch/am15577/projects/ML Music Gen")
TOK_DIR = BASE_DIR / "tokenized"

VOCAB_PATH = TOK_DIR / "vocab.json"
CORPUS_PATH = TOK_DIR / "all_abc_clean.txt"
TRAIN_PATH = TOK_DIR / "train.npy"
VAL_PATH = TOK_DIR / "val.npy"
TEST_PATH = TOK_DIR / "test.npy"


def load_vocab():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    stoi = vocab["stoi"]
    print(f"Vocab size: {len(stoi)}")
    return vocab


def load_token_counts():
    train = np.load(TRAIN_PATH, mmap_mode="r")
    val = np.load(VAL_PATH, mmap_mode="r")
    test = np.load(TEST_PATH, mmap_mode="r")

    print(f"Train tokens: {train.shape[0]:,}")
    print(f"Val tokens:   {val.shape[0]:,}")
    print(f"Test tokens:  {test.shape[0]:,}")
    print(f"Total tokens: {train.shape[0] + val.shape[0] + test.shape[0]:,}")


def seq_length_stats():
    """
    Estimate sequence length distribution using <bos>/<eos> markers
    in all_abc.txt, counting characters between them.
    """

    lengths = []
    cur_len = 0
    inside = False

    with open(CORPUS_PATH, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s == "<bos>":
                # start new tune
                if inside and cur_len > 0:
                    lengths.append(cur_len)
                inside = True
                cur_len = 0
            elif s == "<eos>":
                # end tune
                if inside:
                    lengths.append(cur_len)
                inside = False
                cur_len = 0
            else:
                if inside:
                    # count all characters in this line including newline
                    cur_len += len(line)

    if inside and cur_len > 0:
        lengths.append(cur_len)

    if not lengths:
        print("No tunes found for length stats.")
        return

    arr = np.array(lengths)
    print(f"Number of tunes: {arr.shape[0]:,}")
    print(f"Min length:      {arr.min():,}")
    print(f"Median length:   {np.median(arr):,.0f}")
    print(f"95th percentile: {np.percentile(arr, 95):,.0f}")
    print(f"99th percentile: {np.percentile(arr, 99):,.0f}")
    print(f"Max length:      {arr.max():,}")


if __name__ == "__main__":
    print("=== Vocab stats ===")
    load_vocab()
    print("\n=== Token counts ===")
    load_token_counts()
    print("\n=== Sequence length stats (chars per tune) ===")
    seq_length_stats()