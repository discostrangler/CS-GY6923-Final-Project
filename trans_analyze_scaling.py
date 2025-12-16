import re
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(".")
MODELS = ["tiny", "small", "medium", "large", "xl"]


def parse_log(model_name: str):
    """
    Parse a trans_<model>.out file and extract metrics.
    """
    log_path = BASE_DIR / f"trans_{model_name}.out"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing log file {log_path}")

    text = log_path.read_text()

    def grab(pattern, cast=float, allow_commas=False):
        m = re.search(pattern, text)
        if not m:
            return None
        s = m.group(1)
        if allow_commas:
            s = s.replace(",", "")
        return cast(s)

    n_params = grab(r"Parameters:\s+([\d,]+)", int, allow_commas=True)
    final_val = grab(r"Final val loss:\s+([0-9.]+)", float)
    steps = grab(r"Steps in 1 epoch:\s+(\d+)", int)
    total_tokens = grab(r"Total train tokens:\s+([\d,]+)", int, allow_commas=True)
    wall_sec = grab(r"Wall clock seconds:\s+([0-9.]+)", float)
    gpu_peak = grab(r"GPU memory peak:\s+([0-9.]+)\s*MB", float)

    return dict(
        model_name=model_name,
        n_params=n_params,
        final_val_loss=final_val,
        steps_per_epoch=steps,
        total_train_tokens=total_tokens,
        wall_clock_seconds=wall_sec,
        gpu_mem_peak_mb=gpu_peak,
    )


def fit_power_law_with_offset(N, L):
    """
    Fit L = a * N^(-alpha) + c using a simple grid search
    over alpha and c. Only uses numpy.
    """
    N = np.asarray(N, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)

    alpha_grid = np.linspace(0.01, 1.0, 200)
    c_max = float(L.min()) * 0.9
    c_grid = np.linspace(0.0, c_max, 200)

    best_err = float("inf")
    best_a, best_alpha, best_c = None, None, None

    for alpha in alpha_grid:
        N_term = N ** (-alpha)
        denom = np.dot(N_term, N_term)
        if denom == 0.0:
            continue
        for c in c_grid:
            y = L - c
            a = np.dot(N_term, y) / denom
            pred = a * N_term + c
            err = np.mean((L - pred) ** 2)
            if err < best_err:
                best_err = err
                best_a, best_alpha, best_c = a, alpha, c

    return best_a, best_alpha, best_c


def build_summary_and_write_csv():
    """
    Parse all logs, fit power law, and write scaling_summary.csv
    which includes both per model metrics and fit parameters.
    Returns (rows, fit_a, fit_alpha, fit_c).
    """
    rows = []
    for m in MODELS:
        row = parse_log(m)
        rows.append(row)

    # Fit power law on (N, L)
    N = np.array([r["n_params"] for r in rows], dtype=np.float64)
    L = np.array([r["final_val_loss"] for r in rows], dtype=np.float64)
    fit_a, fit_alpha, fit_c = fit_power_law_with_offset(N, L)

    print(f"Fitted power law: L ≈ {fit_a:.4f} * N^(-{fit_alpha:.4f}) + {fit_c:.4f}")
    print(f"Scaling exponent alpha ≈ {fit_alpha:.4f}")

    # Add fit parameters into each row before writing CSV
    for r in rows:
        r["fit_a"] = fit_a
        r["fit_alpha"] = fit_alpha
        r["fit_c"] = fit_c

    summary_path = BASE_DIR / "scaling_summary.csv"
    fieldnames = [
        "model_name",
        "n_params",
        "final_val_loss",
        "steps_per_epoch",
        "total_train_tokens",
        "wall_clock_seconds",
        "gpu_mem_peak_mb",
        "fit_a",
        "fit_alpha",
        "fit_c",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("Wrote summary with fit parameters to", summary_path)
    for r in rows:
        print(
            f"{r['model_name']:6s}  params={r['n_params']:,}  "
            f"val_loss={r['final_val_loss']:.4f}  "
            f"time={r['wall_clock_seconds']:.1f}s  "
            f"gpu_peak={r['gpu_mem_peak_mb']:.1f}MB"
        )

    return rows, fit_a, fit_alpha, fit_c


def make_scaling_plot(rows, fit_a, fit_alpha, fit_c):
    """
    Create scaling_plot.png with log X axis and linear Y axis.
    Uses already fitted L = a * N^(-alpha) + c.
    """
    rows = sorted(rows, key=lambda r: r["n_params"])
    N = np.array([r["n_params"] for r in rows], dtype=np.float64)
    L = np.array([r["final_val_loss"] for r in rows], dtype=np.float64)

    N_fit = np.logspace(np.log10(N.min()), np.log10(N.max()), 200)
    L_fit = fit_a * N_fit ** (-fit_alpha) + fit_c

    plt.figure()
    plt.scatter(N, L)
    for r in rows:
        plt.text(
            r["n_params"],
            r["final_val_loss"],
            r["model_name"],
            fontsize=8,
            ha="center",
            va="bottom",
        )

    plt.plot(N_fit, L_fit)

    plt.xscale("log")
    plt.xlabel("Model size N (number of parameters, log scale)")
    plt.ylabel("Validation loss after 1 epoch")
    plt.title(f"Transformer scaling (alpha ≈ {fit_alpha:.3f})")
    plt.tight_layout()
    out_path = BASE_DIR / "scaling_plot.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved scaling plot to", out_path)


def parse_training_curve(model_name: str):
    """
    Parse 'Step X/Y  loss Z  lr ...' lines from trans_<model>.out
    and return lists of (step, loss).
    """
    log_path = BASE_DIR / f"trans_{model_name}.out"
    if not log_path.exists():
        print(f"No log file for training curve: {log_path}")
        return [], []

    lines = log_path.read_text().splitlines()
    steps = []
    losses = []

    pattern = re.compile(r"Step\s+(\d+)/(\d+)\s+loss\s+([0-9.]+)")

    for line in lines:
        m = pattern.search(line)
        if m:
            step = int(m.group(1))
            loss = float(m.group(3))
            steps.append(step)
            losses.append(loss)

    return steps, losses


def make_training_curves_all():
    """
    Plot all training loss curves in a single figure.
    No extra CSVs, only one PNG.
    """
    any_data = False

    plt.figure()
    for m in MODELS:
        steps, losses = parse_training_curve(m)
        if not steps:
            print(f"No training curve data found for {m}, skipping.")
            continue

        any_data = True
        plt.plot(steps, losses, label=m)

    if not any_data:
        print("No training curve data for any model. Not writing combined plot.")
        plt.close()
        return

    plt.xlabel("Step")
    plt.ylabel("Training loss")
    plt.title("Training loss curves for all transformer sizes")
    plt.legend()
    plt.tight_layout()
    png_path = BASE_DIR / "train_curves_all.png"
    plt.savefig(png_path, dpi=200)
    plt.close()
    print(f"Wrote combined training curves plot to {png_path}")


if __name__ == "__main__":
    rows, fit_a, fit_alpha, fit_c = build_summary_and_write_csv()
    make_scaling_plot(rows, fit_a, fit_alpha, fit_c)
    make_training_curves_all()