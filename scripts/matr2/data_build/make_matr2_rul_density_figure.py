"""Generate paper/figures/matr2_rul_density.png -- two-panel, shared RUL x-axis,
designed to be self-explanatory without surrounding paper text:
  top:    per-cell test error vs. true RUL (worst 3 cells flagged)
  bottom: train_base's RUL density on the same x-axis
A shaded band + connector lines tie the worst top-panel points directly to the
thin region of the bottom-panel histogram, so the causal claim ("errors are
worst where training data is thinnest") is visible without reading the caption.

Sources:
  - ipynb/exp_v3/matr1_feature_cache/matr2_v1v2inter_ensemble_seed0.pkl
    (per-cell test predictions/errors for the designated result, RMSE 210.7)
  - ipynb/exp_v3/matr1_feature_cache/feature_cache_matr2.pkl (train_base_label)
"""
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
PRED_PKL = os.path.join(REPO, "ipynb", "exp_v3", "matr1_feature_cache", "matr2_v1v2inter_ensemble_seed0.pkl")
CACHE_PKL = os.path.join(REPO, "ipynb", "exp_v3", "matr1_feature_cache", "feature_cache_matr2.pkl")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "paper", "figures", "matr2_rul_density.png")

BLUE = "#4C72B0"
RED = "#C44E52"
BAND_LO, BAND_HI = 1600, 2050  # RUL region containing the 3 worst-error cells


def as_np(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


with open(PRED_PKL, "rb") as f:
    pred = pickle.load(f)
test_true = as_np(pred["test_true"]).flatten()
test_pred = as_np(pred["ens_test"]).flatten()
err = test_pred - test_true

with open(CACHE_PKL, "rb") as f:
    d = pickle.load(f)
train_base = as_np(d["train_base_label"]).flatten()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.8), sharex=True,
                                gridspec_kw={"height_ratios": [1.3, 1], "hspace": 0.1})

fig.suptitle("MATR2: the worst-predicted cells sit exactly where training data is thinnest",
             fontsize=12.5, fontweight="bold", y=0.99)

# --- top: per-cell error vs true RUL ---
order = np.argsort(-np.abs(err))
worst3 = order[:3]
colors = [RED if i in worst3 else "#9DB4D1" for i in range(len(err))]
sizes = [65 if i in worst3 else 28 for i in range(len(err))]
ax1.scatter(test_true, err, c=colors, s=sizes, edgecolor="white", linewidth=0.4, zorder=3)
ax1.axhline(0, color="black", linewidth=0.8, zorder=1)
ax1.axvspan(BAND_LO, BAND_HI, color="grey", alpha=0.13, zorder=0)
ax1.set_ylim(-1000, 220)
for i in worst3:
    ax1.annotate(f"RUL {test_true[i]:.0f}, error {err[i]:+.0f}", xy=(test_true[i], err[i]),
                 xytext=(12, 0), textcoords="offset points", ha="left", va="center", fontsize=8,
                 color=RED, fontweight="bold")

ax1.scatter([], [], c="#9DB4D1", s=28, label="other test cells (n=37)")
ax1.scatter([], [], c=RED, s=65, label="3 worst-error cells")
ax1.legend(loc="upper left", fontsize=8.5, frameon=False)

ax1.set_ylabel("Error = predicted − actual RUL\n(negative = under-predicted)")
rmse_val = float(np.sqrt(np.mean(err ** 2)))
ax1.text(0.015, 0.03, f"MATR2 designated result: test RMSE = {rmse_val:.1f}  (README benchmark = 149)",
         transform=ax1.transAxes, ha="left", va="bottom", fontsize=8.5, color="#333")
ax1.spines[["top", "right"]].set_visible(False)

# --- bottom: train_base RUL density ---
bins = np.arange(0, 2800, 150)
counts, _, _ = ax2.hist(train_base, bins=bins, color=BLUE, alpha=0.88)
ax2.axvspan(BAND_LO, BAND_HI, color="grey", alpha=0.13, zorder=0)

peak_bin = int(counts.max())
band_mask = (bins[:-1] >= BAND_LO) & (bins[:-1] < BAND_HI)
band_avg = counts[band_mask].mean()
ax2.annotate(f"peak: {peak_bin} train cells\nper 150-cycle bin", xy=(500, peak_bin),
             xytext=(650, peak_bin - 3), fontsize=8.5, color="#333",
             ha="left", va="top")
ax2.annotate(f"only ~{band_avg:.0f} cells\nper bin here", xy=((BAND_LO + BAND_HI) / 2, band_avg),
             xytext=((BAND_LO + BAND_HI) / 2, 25), fontsize=8.5, color="#333",
             ha="center", va="bottom",
             arrowprops=dict(arrowstyle="-", color="#333", lw=0.8))

ax2.set_ylabel("# train_base cells\n(training examples)")
ax2.set_xlabel("RUL (cycles)")
ax2.set_xlim(0, 2800)
ax2.spines[["top", "right"]].set_visible(False)

# --- connector lines: worst-3 points (top) -> their x-position on the histogram (bottom) ---
for i in worst3:
    con = ConnectionPatch(xyA=(test_true[i], err[i]), coordsA=ax1.transData,
                           xyB=(test_true[i], ax2.get_ylim()[1]), coordsB=ax2.transData,
                           color=RED, linewidth=0.7, linestyle=":", zorder=1)
    fig.add_artist(con)

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
print(f"Saved {os.path.abspath(OUT_PATH)}")
