import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re

BASE = Path("/scratch/am15577/projects/ML Music Gen")
TRANS_CSV = BASE / "transformer" / "trans_scaling_summary.csv"
RNN_CSV = BASE / "rnn" / "rnn_scaling_summary.csv"
OUT_DIR = BASE / "compare"
OUT_DIR.mkdir(exist_ok=True)


SIZES = ["tiny", "small", "medium", "large", "xl"]


def load_csv(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = r["model_name"]
            rows[name] = r
    return rows


def load_training_curve(path):
    """
    Path = *.out file, extract Step X/Y loss
    """
    if not path.exists():
        return [], []
    txt = path.read_text().splitlines()
    steps, losses = [], []
    pat = re.compile(r"Step\s+(\d+)/(\d+)\s+loss\s+([0-9.]+)")
    for line in txt:
        m = pat.search(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(3)))
    return steps, losses


def make_combined_scaling_plot(trans_rows, rnn_rows):
    # Extract
    N_t = np.array([int(trans_rows[s]["n_params"]) for s in SIZES])
    L_t = np.array([float(trans_rows[s]["final_val_loss"]) for s in SIZES])

    N_r = np.array([int(rnn_rows[s]["n_params"]) for s in SIZES])
    L_r = np.array([float(rnn_rows[s]["final_val_loss"]) for s in SIZES])

    # Fit curves using same formula for both
    def fit_powerlaw(N, L):
        alpha_grid = np.linspace(0.01, 1.0, 200)
        c_max = float(min(L)) * 0.9
        c_grid = np.linspace(0.0, c_max, 200)

        best = (None, None, None, float("inf"))
        N = N.astype(float)
        L = L.astype(float)

        for alpha in alpha_grid:
            term = N ** (-alpha)
            denom = np.dot(term, term)
            if denom == 0:
                continue
            for c in c_grid:
                y = L - c
                a = np.dot(term, y) / denom
                pred = a * term + c
                err = np.mean((pred - L)**2)
                if err < best[3]:
                    best = (a, alpha, c, err)
        return best[:3]

    a_t, alpha_t, c_t = fit_powerlaw(N_t, L_t)
    a_r, alpha_r, c_r = fit_powerlaw(N_r, L_r)

    # Plot
    plt.figure()
    plt.scatter(N_t, L_t, color="blue", label="Transformer")
    plt.scatter(N_r, L_r, color="red", label="RNN")

    N_fit = np.logspace(np.log10(min(N_t.min(), N_r.min())),
                        np.log10(max(N_t.max(), N_r.max())), 200)

    L_fit_T = a_t * N_fit**(-alpha_t) + c_t
    L_fit_R = a_r * N_fit**(-alpha_r) + c_r

    plt.plot(N_fit, L_fit_T, color="blue", linestyle="-")
    plt.plot(N_fit, L_fit_R, color="red", linestyle="-")

    for s in SIZES:
        plt.text(int(trans_rows[s]["n_params"]), float(trans_rows[s]["final_val_loss"]),
                 f"T-{s}", color="blue", fontsize=8)
        plt.text(int(rnn_rows[s]["n_params"]), float(rnn_rows[s]["final_val_loss"]),
                 f"R-{s}", color="red", fontsize=8)

    plt.xscale("log")
    plt.xlabel("Model size N (parameters, log scale)")
    plt.ylabel("Validation loss after 1 epoch")
    plt.title("Transformer vs RNN Scaling Comparison")
    plt.legend()
    plt.tight_layout()

    out = OUT_DIR / "compare_scaling_plot.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)

    # Print α for your report
    print(f"Transformer alpha ≈ {alpha_t:.4f}")
    print(f"RNN alpha ≈ {alpha_r:.4f}")


def make_pairwise_curves(size, trans_rows, rnn_rows):
    trans_log = BASE / "transformer" / f"trans_{size}.out"
    rnn_log = BASE / "rnn" / f"rnn_{size}.out"

    t_steps, t_loss = load_training_curve(trans_log)
    r_steps, r_loss = load_training_curve(rnn_log)

    if not t_steps or not r_steps:
        print(f"Skipping {size}: missing curve data")
        return

    plt.figure()
    plt.plot(t_steps, t_loss, label=f"Transformer-{size}", linewidth=2)
    plt.plot(r_steps, r_loss, label=f"RNN-{size}", linewidth=2)

    plt.xlabel("Step")
    plt.ylabel("Training Loss")
    plt.title(f"Training Curve Comparison: {size}")
    plt.legend()
    plt.tight_layout()

    out = OUT_DIR / f"compare_curve_{size}.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)


def write_compare_table(trans_rows, rnn_rows):
    out = OUT_DIR / "compare_table.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["size", "trans_params", "rnn_params",
                    "trans_val_loss", "rnn_val_loss",
                    "trans_time_sec", "rnn_time_sec",
                    "trans_gpu_MB", "rnn_gpu_MB"])
        for s in SIZES:
            w.writerow([
                s,
                trans_rows[s]["n_params"],
                rnn_rows[s]["n_params"],
                trans_rows[s]["final_val_loss"],
                rnn_rows[s]["final_val_loss"],
                trans_rows[s]["wall_clock_seconds"],
                rnn_rows[s]["wall_clock_seconds"],
                trans_rows[s]["gpu_mem_peak_mb"],
                rnn_rows[s]["gpu_mem_peak_mb"],
            ])
    print("Saved:", out)


if __name__ == "__main__":
    trans = load_csv(TRANS_CSV)
    rnn = load_csv(RNN_CSV)

    write_compare_table(trans, rnn)
    make_combined_scaling_plot(trans, rnn)

    for size in SIZES:
        make_pairwise_curves(size, trans, rnn)