"""Build a proper train_base/val/test split for SNL as a modeling target
(it has so far only existed as a donor pool for CRUH/CRUSH augmentation,
per report_expirement/day_0906/report.md's Open Item #3).

Official split: SNLTrainTestSplitter (batteryml/train_test_split/SNL_split.py),
25 declared test cells, eol_soh=0.9 (README target: 200 RMSE).

Censoring disclosure (CLAUDE.md requirement -- flag, don't silently drop):
SNL has 61 total processed cells; only 31 have a valid (non-NaN) RUL label
under eol_soh=0.9 -- the other 30 are RIGHT-CENSORED (never crossed 90% SOH
within the observed cycling window) and are excluded from this screen. Of
the 25 officially-declared test cells, only 15 have valid labels; of the
remaining 36 officially-declared train cells, only 16 have valid labels.
This screen therefore evaluates on a 15-cell (of 25) subset of the official
test set -- not the full official test set -- because the other 10 cannot
be scored (no ground-truth RUL). This must be stated alongside any result.

Val carve: RUL-tercile-stratified from the 16 labeled official-train cells
(mirrors build_hust_feature_cache.py's approach), VAL_FRAC~0.3 (pre-declared
before looking at test), SPLIT_SEED=0.
"""
import os
import pickle
from collections import Counter

import numpy as np
import torch

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
CRUH_DIR = os.path.join(REPO, "ipynb", "exp_v3", "cruh")
OUT_DIR = os.path.join(REPO, "ipynb", "exp_v3", "snl")
os.makedirs(OUT_DIR, exist_ok=True)
SRC_PKL = os.path.join(CRUH_DIR, "feature_cache_snl.pkl")
OUT_PKL = os.path.join(OUT_DIR, "feature_cache_snl_split.pkl")

VAL_FRAC = 0.3
SPLIT_SEED = 0

OFFICIAL_TEST_IDS = {
    'SNL_18650_LFP_25C_0-100_0.5-1C_a', 'SNL_18650_LFP_25C_0-100_0.5-2C_a', 'SNL_18650_LFP_25C_0-100_0.5-3C_a',
    'SNL_18650_LFP_25C_0-100_0.5-3C_b', 'SNL_18650_LFP_35C_0-100_0.5-1C_b', 'SNL_18650_LFP_35C_0-100_0.5-2C_a',
    'SNL_18650_NCA_15C_0-100_0.5-1C_a', 'SNL_18650_NCA_15C_0-100_0.5-1C_b', 'SNL_18650_NCA_25C_0-100_0.5-0.5C_a',
    'SNL_18650_NCA_25C_0-100_0.5-1C_a', 'SNL_18650_NCA_25C_0-100_0.5-1C_b', 'SNL_18650_NCA_25C_0-100_0.5-1C_c',
    'SNL_18650_NCA_25C_20-80_0.5-0.5C_a', 'SNL_18650_NCA_25C_20-80_0.5-0.5C_b', 'SNL_18650_NCA_25C_20-80_0.5-0.5C_c',
    'SNL_18650_NCA_35C_0-100_0.5-1C_a', 'SNL_18650_NCA_35C_0-100_0.5-1C_d', 'SNL_18650_NCA_35C_0-100_0.5-2C_b',
    'SNL_18650_NMC_15C_0-100_0.5-2C_b', 'SNL_18650_NMC_25C_0-100_0.5-1C_d', 'SNL_18650_NMC_25C_0-100_0.5-2C_b',
    'SNL_18650_NMC_25C_0-100_0.5-3C_c', 'SNL_18650_NMC_25C_0-100_0.5-3C_d', 'SNL_18650_NMC_35C_0-100_0.5-1C_b',
    'SNL_18650_NMC_35C_0-100_0.5-2C_b',
}

with open(SRC_PKL, "rb") as f:
    D = pickle.load(f)
ids = D["cell_ids"]
feature = D["feature"]
label = D["label"]
scalar_raw = D["scalar_raw"]
feature_names = D["feature_names"]

# feature_cache_snl.pkl (donor-pool cache) leaves several feature families as
# genuine NaN for SNL -- temperature_*, internal_resistance_*, qdlin_diff_*
# (17/34 columns) -- unlike feature_cache_snl_base.pkl's older pipeline,
# which nan_to_num's these to 0.0 (confirmed by inspecting
# feature_cache_snl_cluster_n1.pkl's qdlin_diff_std column). TOP3_FEATURES
# in the training script includes qdlin_diff_std, so leaving this as NaN
# crashes training (NaN loss from epoch 1) -- match the established
# convention: fill with 0.0.
n_nan_cols = int(torch.isnan(scalar_raw).any(dim=0).sum())
if n_nan_cols:
    print(f"NaN-filling {n_nan_cols}/{scalar_raw.shape[1]} scalar columns with 0.0 "
          f"(unavailable feature families for SNL, matching feature_cache_snl_base.pkl's convention)")
    scalar_raw = torch.nan_to_num(scalar_raw, nan=0.0)
y = label.numpy().ravel()
assert not np.isnan(y).any(), "donor-pool cache should already be NaN-filtered to 31 labeled cells"

n_official_test_declared = len(OFFICIAL_TEST_IDS)
test_mask = np.array([cid in OFFICIAL_TEST_IDS for cid in ids])
print(f"labeled pool n={len(ids)} (of 61 total SNL cells; 30 excluded as right-censored, NaN label)")
print(f"official test declared: {n_official_test_declared}; labeled subset actually usable as test: {test_mask.sum()}")
print(f"labeled official-train subset: {(~test_mask).sum()}")

test_idx = np.where(test_mask)[0]
train_idx = np.where(~test_mask)[0]

train_y = y[train_idx]
tercile_edges = np.quantile(train_y, [1/3, 2/3])
tercile = np.digitize(train_y, tercile_edges)

rng = np.random.RandomState(SPLIT_SEED)
n_val_target = max(1, round(VAL_FRAC * len(train_idx)))
val_local = []
for t in range(3):
    members = np.where(tercile == t)[0]
    if len(members) == 0:
        continue
    take = max(1, round(VAL_FRAC * len(members))) if len(val_local) < n_val_target else 0
    take = min(take, len(members))
    if take > 0:
        val_local.extend(rng.choice(members, size=take, replace=False).tolist())
val_local = sorted(set(val_local))[:n_val_target]
train_base_local = [i for i in range(len(train_idx)) if i not in val_local]

val_idx = train_idx[val_local]
train_base_idx = train_idx[train_base_local]

print(f"\nFINAL: train_base={len(train_base_idx)}  val={len(val_idx)}  test={len(test_idx)}")


def source_bucket(cid):
    return cid.split("_")[2] if cid.startswith("SNL_18650") else "OTHER"  # chemistry, e.g. NCA/NMC/LFP


print(f"train_base RUL range={y[train_base_idx].min():.0f}-{y[train_base_idx].max():.0f}")
print(f"val RUL range={y[val_idx].min():.0f}-{y[val_idx].max():.0f}")
print(f"test RUL range={y[test_idx].min():.0f}-{y[test_idx].max():.0f}")
print(f"train_base chemistry: {dict(Counter(source_bucket(ids[i]) for i in train_base_idx))}")
print(f"val chemistry:        {dict(Counter(source_bucket(ids[i]) for i in val_idx))}")
print(f"test chemistry:       {dict(Counter(source_bucket(ids[i]) for i in test_idx))}")

train_base_scalar_raw = scalar_raw[train_base_idx]
val_scalar_raw = scalar_raw[val_idx]
test_scalar_raw = scalar_raw[test_idx]

mu = train_base_scalar_raw.mean(dim=0, keepdim=True)
sigma = train_base_scalar_raw.std(dim=0, keepdim=True).clamp_min(1e-6)

out = dict(
    train_base_feature=feature[train_base_idx], train_base_label=label[train_base_idx],
    train_base_scalar_raw=train_base_scalar_raw, train_base_scalar=(train_base_scalar_raw - mu) / sigma,
    train_ids=[ids[i] for i in train_base_idx],
    val_feature=feature[val_idx], val_label=label[val_idx],
    val_scalar_raw=val_scalar_raw, val_scalar=(val_scalar_raw - mu) / sigma,
    val_ids=[ids[i] for i in val_idx],
    test_feature=feature[test_idx], test_label=label[test_idx],
    test_scalar_raw=test_scalar_raw, test_scalar=(test_scalar_raw - mu) / sigma,
    test_ids=[ids[i] for i in test_idx],
    feature_names=feature_names,
    censoring_note=(
        "SNL has 61 total processed cells; only 31 have a valid (non-NaN) "
        "RUL label at eol_soh=0.9 (RULLabelAnnotator). The other 30 are "
        "right-censored (never crossed 90% SOH in the observed window) and "
        "are excluded from this cache/screen -- disclosed, not silently "
        "dropped. Of the 25 officially-declared SNLTrainTestSplitter test "
        "cells, only 15 have valid labels and are usable as test here; the "
        "other 10 declared-test cells are among the 30 censored and are "
        "absent from this test set entirely -- this is a 15-of-25 subset "
        "of the official test partition, not the full official test set."
    ),
)

with open(OUT_PKL, "wb") as f:
    pickle.dump(out, f)
print(f"\nsaved -> {OUT_PKL}")
