"""Generate paper/figures/matr2_rul_smallmultiples.png -- three stacked RUL
histograms (this work's train_base, BatteryML README train M1tr, test), one
color/panel each on a shared x-axis, with median lines and a direct annotation
of the key insight (test's median sits much closer to train_base's than to
M1tr's). Designed to be self-explanatory without surrounding paper text.

Sources:
  - ipynb/exp_v3/matr1_feature_cache/matr2_v1v2inter_ensemble_seed0.pkl (test_true)
  - ipynb/exp_v3/matr1_feature_cache/feature_cache_matr2.pkl (train_base_label)
  - ipynb/exp_v3/data_split/feature_raw_cache.pkl (meta.official_role == "MATR1_train")
"""
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
PRED_PKL = os.path.join(REPO, "ipynb", "exp_v3", "matr1_feature_cache", "matr2_v1v2inter_ensemble_seed0.pkl")
CACHE_PKL = os.path.join(REPO, "ipynb", "exp_v3", "matr1_feature_cache", "feature_cache_matr2.pkl")
META_PKL = os.path.join(REPO, "ipynb", "exp_v3", "data_split", "feature_raw_cache.pkl")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "paper", "figures", "matr2_rul_smallmultiples.png")


def as_np(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


with open(PRED_PKL, "rb") as f:
    pred = pickle.load(f)
test_true = as_np(pred["test_true"]).flatten()

with open(CACHE_PKL, "rb") as f:
    d = pickle.load(f)
train_base = as_np(d["train_base_label"]).flatten()

with open(META_PKL, "rb") as f:
    _, meta = pickle.load(f)
m1tr_rul = meta.loc[meta.official_role == "MATR1_train", "rul"].to_numpy(dtype=float)

SERIES = [
    ("This work's train_base", train_base, "#2b5c8f"),
    ("BatteryML README train (M1tr)", m1tr_rul, "#d9822b"),
    ("Test (true RUL)", test_true, "#2b9348"),
]
med_train, med_m1tr, med_test = (np.median(train_base), np.median(m1tr_rul), np.median(test_true))

bins = np.arange(0, 2800, 150)
fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.2), sharex=True)

fig.suptitle("MATR2: BatteryML's own train pool is a worse RUL match to test\nthan this work's train_base is",
             fontsize=12.5, fontweight="bold", y=1.0)

for ax, (name, arr, color) in zip(axes, SERIES):
    ax.hist(arr, bins=bins, color=color, alpha=0.92, edgecolor="white", linewidth=0.4)
    ax.axvline(np.median(arr), color="black", linestyle="--", linewidth=1.2, label=f"median {np.median(arr):.0f}")
    ax.text(0.99, 0.90, f"{name}\nn={len(arr)}, median={np.median(arr):.0f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color="#222")
    ax.legend(loc="upper left", fontsize=8, frameon=False, handlelength=1.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("count")

# vertical guide showing test's median cutting across all 3 panels, to make the mismatch visible
for ax in axes:
    ax.axvline(med_test, color="#2b9348", linestyle=":", linewidth=1.3, alpha=0.7, zorder=0)

axes[0].text(0.99, 0.58, f"test's median ({med_test:.0f}) is only {abs(med_train - med_test):.0f} away  \u2192  close match",
             transform=axes[0].transAxes, ha="right", va="top", fontsize=8.5, color="#2b5c8f", fontweight="bold")
axes[1].text(0.99, 0.58, f"test's median ({med_test:.0f}) is {abs(med_m1tr - med_test):.0f} away  \u2192  poor match",
             transform=axes[1].transAxes, ha="right", va="top", fontsize=8.5, color="#d9822b", fontweight="bold")

axes[-1].set_xlabel("RUL (cycles)")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved {os.path.abspath(OUT_PATH)}")
