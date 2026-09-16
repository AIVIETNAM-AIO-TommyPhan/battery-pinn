"""Computes MATR2's train-vs-test FEATURE shift (isolating true b1/b2-vs-b3 protocol
difference from the HUST-pooling confound -- HUST uses different units/scale on
several raw features and would otherwise dominate the comparison), and generates
paper/figures/matr2_rul_vs_shift.png: per test cell, true RUL vs. a feature-shift
score (on the 3 scalar-branch features the model actually uses, §3.3), colored by
prediction error. Prints the standardized-mean-difference (SMD) table and the
RUL-vs-error / shift-vs-error correlations cited in §5.7.

Sources:
  - ipynb/exp_v3/matr1_feature_cache/feature_cache_matr2.pkl
    (train_base_scalar_raw, test_scalar_raw, feature_names, train_ids)
  - ipynb/exp_v3/matr1_feature_cache/matr2_v1v2inter_ensemble_seed0.pkl
    (per-cell test predictions for the designated result, RMSE 210.7)
"""
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
CACHE_PKL = os.path.join(REPO, "ipynb", "exp_v3", "matr1_feature_cache", "feature_cache_matr2.pkl")
PRED_PKL = os.path.join(REPO, "ipynb", "exp_v3", "matr1_feature_cache", "matr2_v1v2inter_ensemble_seed0.pkl")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "paper", "figures", "matr2_rul_vs_shift.png")

MODEL_FEATS = ["qdlin_diff_std", "voltage_slope_50_90", "voltage_soc_90"]  # §3.3 scalar branch


def as_np(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


with open(CACHE_PKL, "rb") as f:
    d = pickle.load(f)
names = list(d["feature_names"])
tr = as_np(d["train_base_scalar_raw"])
te = as_np(d["test_scalar_raw"])
tr_ids = list(d["train_ids"])

is_hust = np.array(["HUST" in c for c in tr_ids])
tr_matr = tr[~is_hust]  # MATR-family only: MATR1_train + MATR1_test + MATR34-b4

# --- SMD per feature (all 34), MATR-only train vs. test -- printed for §5.7's text ---
smd_by_name = {}
for i, name in enumerate(names):
    a = tr_matr[:, i]
    b = te[:, i]
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        continue
    am, astd = a.mean(), a.std()
    bm, bstd = b.mean(), b.std()
    pooled = np.sqrt((astd ** 2 + bstd ** 2) / 2) + 1e-9
    smd = (bm - am) / pooled
    smd_by_name[name] = smd
for name, smd in sorted(smd_by_name.items(), key=lambda kv: -abs(kv[1])):
    print(f"{name:30s} SMD={smd:+.2f}")
print("\nModel's 3 scalar-branch features (§3.3):", {f: round(smd_by_name[f], 2) for f in MODEL_FEATS})

# --- per-cell feature-shift score (RMS z-score on the 3 model features) vs. RUL/error ---
idx = [names.index(f) for f in MODEL_FEATS]
tr_mean = np.nanmean(tr_matr[:, idx], axis=0)
tr_std = np.nanstd(tr_matr[:, idx], axis=0) + 1e-9
z = (te[:, idx] - tr_mean) / tr_std
shift_score = np.sqrt((z ** 2).mean(axis=1))

with open(PRED_PKL, "rb") as f:
    pred = pickle.load(f)
te_ids = list(d["test_ids"])
test_true = as_np(pred["test_true"]).flatten()
test_pred = as_np(pred["ens_test"]).flatten()
err = test_pred - test_true
abs_err = np.abs(err)

r_shift = np.corrcoef(shift_score, abs_err)[0, 1]
r_rul = np.corrcoef(test_true, abs_err)[0, 1]
print(f"\ncorr(shift_score, |err|) = {r_shift:.2f}")
print(f"corr(true_RUL, |err|)    = {r_rul:.2f}")

fig, ax = plt.subplots(figsize=(7.5, 6))
sc = ax.scatter(shift_score, test_true, c=abs_err, cmap="Reds", s=90, edgecolor="#444", linewidth=0.5, vmin=0)
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("|prediction error| (RUL)")

worst3 = np.argsort(-abs_err)[:3]
for i in worst3:
    ax.annotate(te_ids[i].replace("MATR34_MATR_", ""), xy=(shift_score[i], test_true[i]),
                xytext=(8, 6), textcoords="offset points", fontsize=8, fontweight="bold")

ax.set_xlabel("Feature-shift score (RMS z-score vs. MATR train, on the 3 model features)")
ax.set_ylabel("True RUL (cycles)")
ax.set_title(f"MATR2 test cells: RUL vs. feature shift, colored by error\n"
             f"corr(RUL, |err|)={r_rul:.2f}   corr(shift, |err|)={r_shift:.2f}", fontsize=10.5)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=170, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {os.path.abspath(OUT_PATH)}")
