from pathlib import Path

BASE_DIR = Path("/scratch/am15577/projects/ML Music Gen")
TOK_DIR = BASE_DIR / "tokenized"

INPUT_PATH = TOK_DIR / "all_abc.txt"
OUTPUT_PATH = TOK_DIR / "all_abc_clean.txt"

# Tune length limits in characters between <bos> and <eos>
MIN_CHARS = 200
MAX_CHARS = 8000


def main():
    print(f"Reading from: {INPUT_PATH}")
    print(f"Writing cleaned corpus to: {OUTPUT_PATH}")
    print(f"Keeping tunes with {MIN_CHARS} <= length <= {MAX_CHARS} chars")

    kept = 0
    dropped_short = 0
    dropped_long = 0
    total = 0

    out = OUTPUT_PATH.open("w", encoding="utf-8")

    with INPUT_PATH.open("r", encoding="utf-8", errors="ignore") as f:
        current_lines = []
        cur_len = 0
        inside = False
        too_long = False

        for line in f:
            stripped = line.strip()

            if stripped == "<bos>":
                # finalize previous tune if any
                if inside:
                    if MIN_CHARS <= cur_len <= MAX_CHARS and not too_long:
                        kept += 1
                        for l in current_lines:
                            out.write(l)
                    else:
                        if cur_len < MIN_CHARS:
                            dropped_short += 1
                        else:
                            dropped_long += 1

                total += 1
                inside = True
                current_lines = [line]
                cur_len = 0
                too_long = False

            elif stripped == "<eos>":
                if inside:
                    # end of tune marker
                    if not too_long:
                        current_lines.append(line)

                    if MIN_CHARS <= cur_len <= MAX_CHARS and not too_long:
                        kept += 1
                        for l in current_lines:
                            out.write(l)
                    else:
                        if cur_len < MIN_CHARS:
                            dropped_short += 1
                        else:
                            dropped_long += 1

                inside = False
                current_lines = []
                cur_len = 0
                too_long = False

            else:
                if inside:
                    cur_len += len(line)
                    if not too_long:
                        if cur_len > MAX_CHARS:
                            # mark as too long and drop buffered text to save memory
                            too_long = True
                            current_lines = []
                        else:
                            current_lines.append(line)

        # handle last tune if file does not end cleanly
        if inside:
            if MIN_CHARS <= cur_len <= MAX_CHARS and not too_long:
                kept += 1
                for l in current_lines:
                    out.write(l)
            else:
                if cur_len < MIN_CHARS:
                    dropped_short += 1
                else:
                    dropped_long += 1

    out.close()

    print(f"Total tunes scanned:    {total}")
    print(f"Kept tunes:             {kept}")
    print(f"Dropped too short:      {dropped_short}")
    print(f"Dropped too long:       {dropped_long}")
    print(f"Cleaned corpus written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()