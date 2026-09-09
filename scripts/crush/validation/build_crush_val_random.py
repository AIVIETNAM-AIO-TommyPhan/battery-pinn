"""Third diagnostic val-rebuild for CRUSH: plain random val, no stratification
by RUL or feature-similarity (per explicit request to try the simplest
baseline against the quantile/similarity attempts). Built on top of the
ALREADY-DEDUPED cache (feature_cache_crush_hust10_calce15_fullval_withsoh_dedup.pkl)
so this run isolates "does a bigger, unstratified random val fix the
val/test correlation" without re-introducing the leakage bug.

Pool = deduped train_base (n=55) + current val (n=15) = 70 unique cells
(test n=44 untouched). VAL_SIZE=20 (per request, "20/66" -- actual unique
pool here is 70, not 66; using 70 and VAL_SIZE=20 as specified).
SPLIT_SEED=0, pre-declared, plain np.random.RandomState.choice, no
stratification of any kind.
"""
import os
import pickle
from collections import Counter

import numpy as np
import torch

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
CACHE_DIR = os.path.join(REPO, "ipynb", "exp_v3", "crush")
SRC_PKL = os.path.join(CACHE_DIR, "feature_cache_crush_hust10_calce15_fullval_withsoh_dedup.pkl")
OUT_PKL = os.path.join(CACHE_DIR, "feature_cache_crush_hust10_calce15_fullval_withsoh_randval.pkl")

VAL_SIZE = 20
SPLIT_SEED = 0

with open(SRC_PKL, "rb") as f:
    C = pickle.load(f)


def source(cid):
    if cid.startswith("CALCE"):
        return "CALCE"
    if cid.startswith("RWTH"):
        return "RWTH"
    if cid.startswith("UL-PUR") or cid.startswith("UL_PUR"):
        return "UL_PUR"
    if cid.startswith("SNL"):
        return "SNL"
    if cid.startswith("HNEI"):
        return "HNEI"
    return "OTHER"


pool_ids = list(C["train_ids"]) + list(C["val_ids"])
pool_feature = torch.cat([C["train_base_feature"], C["val_feature"]], dim=0)
pool_label = torch.cat([C["train_base_label"], C["val_label"]], dim=0)
pool_scalar_raw = torch.cat([C["train_base_scalar_raw"], C["val_scalar_raw"]], dim=0)
pool_soh = torch.cat([C["train_soh"], C["val_soh"]], dim=0)
n_pool = len(pool_ids)
assert n_pool == len(set(pool_ids)), "pool should already be leak-free (built from dedup cache)"
print(f"pool n={n_pool} (deduped, test excluded)", flush=True)

rng = np.random.RandomState(SPLIT_SEED)
val_pos = sorted(rng.choice(n_pool, size=VAL_SIZE, replace=False).tolist())
train_pos = [i for i in range(n_pool) if i not in val_pos]

new_val_ids = [pool_ids[i] for i in val_pos]
new_train_ids = [pool_ids[i] for i in train_pos]
y = pool_label.numpy().ravel()

print(f"NEW val (random): n={len(new_val_ids)} composition={dict(Counter(source(c) for c in new_val_ids))}")
print(f"NEW val RUL range={y[val_pos].min():.0f}-{y[val_pos].max():.0f} mean={y[val_pos].mean():.1f}")
print(f"NEW train_base: n={len(new_train_ids)} composition={dict(Counter(source(c) for c in new_train_ids))}")
print(f"NEW train_base RUL range={y[train_pos].min():.0f}-{y[train_pos].max():.0f} mean={y[train_pos].mean():.1f}")
test_y = C["test_label"].numpy().ravel()
print(f"(reference) test RUL range={test_y.min():.0f}-{test_y.max():.0f} mean={test_y.mean():.1f}")

new_train_scalar_raw = pool_scalar_raw[train_pos]
new_val_scalar_raw = pool_scalar_raw[val_pos]
test_scalar_raw = C["test_scalar_raw"]

mu = new_train_scalar_raw.mean(dim=0, keepdim=True)
sigma = new_train_scalar_raw.std(dim=0, keepdim=True).clamp_min(1e-6)

out = dict(C)
out["train_base_feature"] = pool_feature[train_pos]
out["train_base_label"] = pool_label[train_pos]
out["train_base_scalar_raw"] = new_train_scalar_raw
out["train_base_scalar"] = (new_train_scalar_raw - mu) / sigma
out["train_ids"] = new_train_ids
out["train_soh"] = pool_soh[train_pos]

out["val_feature"] = pool_feature[val_pos]
out["val_label"] = pool_label[val_pos]
out["val_scalar_raw"] = new_val_scalar_raw
out["val_scalar"] = (new_val_scalar_raw - mu) / sigma
out["val_ids"] = new_val_ids
out["val_soh"] = pool_soh[val_pos]

out["test_scalar"] = (test_scalar_raw - mu) / sigma
out["dedup_note"] = C.get("dedup_note", "")
out["retune_note"] = (
    f"val rebuilt as a plain random sample (VAL_SIZE={VAL_SIZE}, SPLIT_SEED={SPLIT_SEED}, "
    "no stratification) of the deduped train_base+val pool (n=70); test untouched. "
    "Simplest baseline against the RUL-quantile-stratified and test-centroid-similarity "
    "diagnostic rebuilds."
)

with open(OUT_PKL, "wb") as f:
    pickle.dump(out, f)
print(f"\nsaved -> {OUT_PKL}")
