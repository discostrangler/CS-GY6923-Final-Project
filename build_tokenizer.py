import os
import json
from pathlib import Path

# Paths
ABC_DIR = Path("/scratch/am15577/projects/ML Music Gen/abc_data")
OUT_DIR = Path("/scratch/am15577/projects/ML Music Gen/tokenized")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Special tokens
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

def collect_abc_files(root: Path):
    abc_files = [p for p in root.rglob("*.abc") if p.is_file()]
    print(f"Found {len(abc_files)} .abc files")
    return abc_files

def build_char_vocab(abc_files):
    charset = set()
    for path in abc_files:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                charset.update(text)
        except Exception as e:
            print(f"Error reading {path}: {e}")
    charset = sorted(list(charset))
    print(f"Found {len(charset)} unique characters")
    return charset

def make_token_maps(charset):
    stoi = {}
    itos = {}

    next_id = 0
    for tok in SPECIAL_TOKENS:
        stoi[tok] = next_id
        itos[next_id] = tok
        next_id += 1

    for ch in charset:
        if ch in stoi:
            continue
        stoi[ch] = next_id
        itos[next_id] = ch
        next_id += 1

    print(f"Total vocab size (including specials): {len(stoi)}")
    return stoi, itos

def save_vocab(stoi, itos):
    vocab_path = OUT_DIR / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "stoi": stoi,
                "itos": {int(k): v for k, v in itos.items()},
                "special_tokens": SPECIAL_TOKENS,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved vocab to {vocab_path}")

def build_concatenated_corpus(abc_files):
    out_txt_path = OUT_DIR / "all_abc.txt"
    bos = "<bos>"
    eos = "<eos>"

    with open(out_txt_path, "w", encoding="utf-8") as out_f:
        for path in abc_files:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                out_f.write(bos + "\n")
                out_f.write(text)
                if not text.endswith("\n"):
                    out_f.write("\n")
                out_f.write(eos + "\n")
            except Exception as e:
                print(f"Error reading {path}: {e}")

    print(f"Saved concatenated corpus to {out_txt_path}")

if __name__ == "__main__":
    print(f"ABC_DIR: {ABC_DIR}")
    print(f"OUT_DIR: {OUT_DIR}")
    abc_files = collect_abc_files(ABC_DIR)
    if not abc_files:
        print("No .abc files found. Check ABC_DIR path.")
        raise SystemExit(1)
    charset = build_char_vocab(abc_files)
    stoi, itos = make_token_maps(charset)
    save_vocab(stoi, itos)
    build_concatenated_corpus(abc_files)