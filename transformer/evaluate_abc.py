import argparse
import csv
import re
import shutil
from pathlib import Path

from music21 import converter, meter


HEADER_LINE_RE = re.compile(r"^\s*([A-Za-z]):\s*(.*)\s*$")


def strip_leading_garbage(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    # Find first plausible ABC header line like "X:" or "T:" or "M:" etc.
    start = None
    for i, ln in enumerate(lines):
        if HEADER_LINE_RE.match(ln):
            start = i
            break
    if start is None:
        return text.strip() + "\n"
    return "\n".join(lines[start:]).strip() + "\n"


def split_header_body(text: str):
    """
    ABC header is a block of lines matching "<letter>: ...".
    Body starts at first non-header line after seeing at least one header line.
    """
    lines = text.splitlines()
    header = []
    body = []

    seen_header = False
    in_body = False

    for ln in lines:
        if not in_body and HEADER_LINE_RE.match(ln):
            seen_header = True
            header.append(ln.strip())
        else:
            if seen_header:
                in_body = True
            if in_body:
                body.append(ln)

    # If we never found a header, treat everything as body
    if not seen_header:
        return [], lines

    # Drop leading empty lines in body
    while body and body[0].strip() == "":
        body.pop(0)

    return header, body


def build_clean_header(existing_header_lines, default_L="1/8"):
    """
    Keep any existing valid tags, but enforce required tags.
    Prefer existing values when present.
    """
    tags = {}
    for ln in existing_header_lines:
        m = HEADER_LINE_RE.match(ln)
        if not m:
            continue
        k, v = m.group(1).upper(), m.group(2).strip()
        # Keep first occurrence of each tag
        if k not in tags:
            tags[k] = v

    if "X" not in tags:
        tags["X"] = "1"
    if "T" not in tags:
        tags["T"] = "generated"
    if "M" not in tags:
        tags["M"] = "4/4"
    if "L" not in tags:
        tags["L"] = default_L
    if "K" not in tags:
        tags["K"] = "C"

    # Write a minimal header in a stable order
    ordered = ["X", "T", "M", "L", "K"]
    out = [f"{k}:{tags[k]}" for k in ordered]

    # Optionally keep a few harmless extras if present (tempo, meter info, etc.)
    for extra in ["Q", "R", "C"]:
        if extra in tags:
            out.insert(ordered.index("K"), f"{extra}:{tags[extra]}")

    return out


def repair_abc(text: str, default_L="1/8") -> str:
    text = strip_leading_garbage(text)
    header_lines, body_lines = split_header_body(text)

    clean_header = build_clean_header(header_lines, default_L=default_L)

    # If body is empty, still return header with blank line
    body = "\n".join(body_lines).strip()
    if body:
        return "\n".join(clean_header) + "\n\n" + body.strip() + "\n"
    return "\n".join(clean_header) + "\n\n"


def clean_stream_for_midi(s):
    """
    Remove duplicate TimeSignature objects (keep first per Part).
    """
    try:
        parts = list(s.parts) if hasattr(s, "parts") else [s]
        for p in parts:
            tss = list(p.recurse().getElementsByClass(meter.TimeSignature))
            if len(tss) <= 1:
                continue
            # Keep the first, remove the rest
            for ts in tss[1:]:
                try:
                    p.remove(ts, recurse=True)
                except Exception:
                    try:
                        ts.activeSite.remove(ts)
                    except Exception:
                        pass
    except Exception:
        pass
    return s


def write_examples(examples_dir: Path, abc_path: Path, midi_path: Path, idx: int):
    examples_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(abc_path, examples_dir / f"example_{idx:02d}.abc")
    if midi_path.exists():
        shutil.copy2(midi_path, examples_dir / f"example_{idx:02d}.mid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--repair_header", action="store_true")
    ap.add_argument("--default_L", default="1/8")
    ap.add_argument("--examples", type=int, default=5)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    midi_dir = out_dir / "midi"
    examples_dir = out_dir / "examples"
    midi_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    abc_files = sorted(input_dir.glob("*.abc"))

    per_file_csv = out_dir / "per_file_results.csv"
    links_file = out_dir / "abcjs_links.txt"

    parsed_ok = 0
    midi_ok = 0
    examples_written = 0

    rows = []
    links = []

    for f in abc_files:
        parse_ok = 0
        midi_ok_flag = 0
        parse_error = ""
        midi_error = ""

        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            if args.repair_header:
                text = repair_abc(text, default_L=args.default_L)

            s = converter.parseData(text, format="abc")
            parse_ok = 1
            parsed_ok += 1

            s = clean_stream_for_midi(s)

            midi_out = midi_dir / (f.stem + ".mid")
            s.write("midi", fp=str(midi_out))
            midi_ok_flag = 1
            midi_ok += 1

            if examples_written < args.examples:
                repaired_abc = examples_dir / f"{f.stem}_repaired.abc"
                repaired_abc.write_text(text, encoding="utf-8")
                write_examples(examples_dir, repaired_abc, midi_out, examples_written)
                examples_written += 1

            links.append(f"{f.name}\t{(examples_dir / (f.stem + '_repaired.abc')).as_posix()}")

        except Exception as e:
            if parse_ok == 0:
                parse_error = repr(e)
            else:
                midi_error = repr(e)

        rows.append([f.name, parse_ok, midi_ok_flag, parse_error, midi_error])

    with per_file_csv.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["file", "parse_ok", "midi_ok", "parse_error", "midi_error"])
        w.writerows(rows)

    with links_file.open("w", encoding="utf-8") as fp:
        fp.write("file\tabc_path\n")
        for ln in links:
            fp.write(ln + "\n")

    total = len(abc_files)
    print(f"input_dir: {input_dir}")
    print(f"total_abc_files: {total}")
    print(f"parsed_ok: {parsed_ok} ({(parsed_ok/total*100 if total else 0):.2f}%)")
    print(f"midi_ok: {midi_ok} ({(midi_ok/total*100 if total else 0):.2f}%)")
    print(f"examples_written: {examples_written}")
    print(f"per_file_csv: {per_file_csv}")
    print(f"midi_dir: {midi_dir}")
    print(f"examples_dir: {examples_dir}")
    print(f"links_file: {links_file}")


if __name__ == "__main__":
    main()