"""Diagnostic rebuild of CRUSH's val split, motivated by an observed
val/test RUL-distribution mismatch (val max=1410 vs test max=2169, found
after inspecting feature_cache_crush_hust10_calce15_fullval_withsoh.pkl).
This is a POST-HOC, test-distribution-informed diagnostic, not a blind
re-split -- disclosed as such, not eligible to replace the official
CRUSH headline without a truly blind confirmation.

Method (pre-declared before looking at any new val/test correlation number):
  - Pool = current train_base (n=64) + current val (n=15) = 79 cells.
    test (n=44) is never touched -- membership, features, and labels for
    test are byte-identical to the original cache.
  - Bin the pool by RUL quantile (N_BINS=5, via np.quantile edges on the
    pooled label).
  - Draw VAL_SIZE=15 cells (same size as the original val -- isolates the
    effect of "matching the tail" from "just having more val data"),
    proportionally per bin (largest-remainder rounding to hit exactly 15),
    RandomState(SPLIT_SEED=0) within each bin.
  - Recompute the z-scored scalar features (`*_scalar`) from the NEW
    train_base' raw stats, applied consistently to train_base', val', and
    test (test_scalar changes only in its normalization constants, never
    in which cells it contains).
"""
import os
import pickle
from collections import Counter

import numpy as np
import torch

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
CACHE_DIR = os.path.join(REPO, "ipynb", "exp_v3", "crush")
SRC_PKL = os.path.join(CACHE_DIR, "feature_cache_crush_hust10_calce15_fullval_withsoh.pkl")
OUT_PKL = os.path.join(CACHE_DIR, "feature_cache_crush_hust10_calce15_fullval_withsoh_qstrat.pkl")

N_BINS = 5
VAL_SIZE = 15
SPLIT_SEED = 0

with open(SRC_PKL, "rb") as f:
    C = pickle.load(f)

# --- pool = train_base + val (test untouched) ---
pool_ids = list(C["train_ids"]) + list(C["val_ids"])
pool_feature = torch.cat([C["train_base_feature"], C["val_feature"]], dim=0)
pool_label = torch.cat([C["train_base_label"], C["val_label"]], dim=0)
pool_scalar_raw = torch.cat([C["train_base_scalar_raw"], C["val_scalar_raw"]], dim=0)
pool_soh = torch.cat([C["train_soh"], C["val_soh"]], dim=0)
n_pool = len(pool_ids)
y = pool_label.numpy().ravel()
print(f"pool n={n_pool}  RUL range={y.min():.0f}-{y.max():.0f}", flush=True)

# --- quantile bins on the pool only (test never consulted) ---
edges = np.quantile(y, np.linspace(0, 1, N_BINS + 1))
edges[0] -= 1e-6
edges[-1] += 1e-6
bin_idx = np.digitize(y, edges) - 1
bin_idx = np.clip(bin_idx, 0, N_BINS - 1)

rng = np.random.RandomState(SPLIT_SEED)
counts = [int((bin_idx == b).sum()) for b in range(N_BINS)]
raw_alloc = [c * VAL_SIZE / n_pool for c in counts]
alloc = [int(np.floor(a)) for a in raw_alloc]
remainder = VAL_SIZE - sum(alloc)
frac_order = np.argsort([-(a - int(a)) for a in raw_alloc])
for b in frac_order[:remainder]:
    alloc[b] += 1
print(f"bin edges: {edges.round(1).tolist()}")
print(f"bin counts (pool): {counts}  val alloc: {alloc}", flush=True)

val_pos = []
for b in range(N_BINS):
    members = np.where(bin_idx == b)[0]
    take = min(alloc[b], len(members))
    if take > 0:
        chosen = rng.choice(members, size=take, replace=False)
        val_pos.extend(chosen.tolist())
val_pos = sorted(val_pos)
train_pos = [i for i in range(n_pool) if i not in val_pos]
assert len(val_pos) == VAL_SIZE, f"expected {VAL_SIZE} got {len(val_pos)}"

new_val_ids = [pool_ids[i] for i in val_pos]
new_train_ids = [pool_ids[i] for i in train_pos]


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


print(f"NEW val: n={len(new_val_ids)} composition={dict(Counter(source(c) for c in new_val_ids))}")
print(f"NEW train_base: n={len(new_train_ids)} composition={dict(Counter(source(c) for c in new_train_ids))}")
new_val_y = y[val_pos]
new_train_y = y[train_pos]
print(f"NEW val RUL range={new_val_y.min():.0f}-{new_val_y.max():.0f} mean={new_val_y.mean():.1f}")
print(f"NEW train_base RUL range={new_train_y.min():.0f}-{new_train_y.max():.0f} mean={new_train_y.mean():.1f}")
print(f"(reference) test RUL range={C['test_label'].numpy().ravel().min():.0f}-{C['test_label'].numpy().ravel().max():.0f}")

# --- recompute z-scored scalar features from new train_base stats ---
new_train_scalar_raw = pool_scalar_raw[train_pos]
new_val_scalar_raw = pool_scalar_raw[val_pos]
test_scalar_raw = C["test_scalar_raw"]

mu = new_train_scalar_raw.mean(dim=0, keepdim=True)
sigma = new_train_scalar_raw.std(dim=0, keepdim=True).clamp_min(1e-6)

new_train_scalar = (new_train_scalar_raw - mu) / sigma
new_val_scalar = (new_val_scalar_raw - mu) / sigma
new_test_scalar = (test_scalar_raw - mu) / sigma

out = dict(C)  # start from a copy, overwrite the split-dependent fields
out["train_base_feature"] = pool_feature[train_pos]
out["train_base_label"] = pool_label[train_pos]
out["train_base_scalar_raw"] = new_train_scalar_raw
out["train_base_scalar"] = new_train_scalar
out["train_ids"] = new_train_ids
out["train_soh"] = pool_soh[train_pos]

out["val_feature"] = pool_feature[val_pos]
out["val_label"] = pool_label[val_pos]
out["val_scalar_raw"] = new_val_scalar_raw
out["val_scalar"] = new_val_scalar
out["val_ids"] = new_val_ids
out["val_soh"] = pool_soh[val_pos]

out["test_scalar"] = new_test_scalar  # renormalized only; test membership/feature/label untouched
out["retune_note"] = (
    "val rebuilt via RUL-quantile-stratified resample of the train_base+val "
    "pool (N_BINS=5, VAL_SIZE=15, SPLIT_SEED=0); test untouched. Motivated "
    "by an observed val/test RUL-range mismatch found post-hoc -- disclosed "
    "as a test-distribution-informed diagnostic, not a blind re-split."
)

with open(OUT_PKL, "wb") as f:
    pickle.dump(out, f)
print(f"\nsaved -> {OUT_PKL}")
