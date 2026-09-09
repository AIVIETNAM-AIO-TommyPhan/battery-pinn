"""Second diagnostic rebuild of CRUSH's val split (alternative to the
RUL-quantile-stratified version in build_crush_val_rulquantile.py), per
explicit request: instead of stratifying the pool by its OWN RUL quantiles,
pick the pool cells (train_base -- which already includes the 10 HUST +
6 TONGJI cross-dataset donor cells -- plus the current val) that are most
SIMILAR TO TEST in combined RUL + scalar-feature space, mirroring this
project's existing donor-search convention
(search_calce_reinforce_donors.py's centroid-distance method).

This is explicitly test-distribution-informed (it looks at test's own
centroid to choose val) -- more so than the quantile version -- so it is
disclosed as a mechanism-diagnostic screen only, never eligible to become
the official CRUSH split without a genuinely blind confirmation.

Method (pre-declared before training/evaluating on this split):
  - Pool = current train_base (n=64, includes HUST/TONGJI donors) + current
    val (n=15) = 79 cells. test (n=44) untouched in membership/features/
    labels -- only its z-scored scalar features get renormalized (see
    below), same as the quantile-stratified script.
  - Combined z-space: 34 raw scalar features (drop degenerate columns,
    std>1e-6 computed on the pool) + RUL, all z-scored using POOL stats.
  - test centroid = mean of test cells' positions in that same z-space
    (test raw features transformed with POOL mu/sigma; test RUL z-scored
    with POOL rul_mu/rul_sigma -- test's own label is only used to place
    the centroid, never to pick individual val cells one-by-one against
    their own matching test cell).
  - VAL_SIZE=15 pool cells closest (Euclidean) to that centroid -> new val;
    remainder -> new train_base.
"""
import os
import pickle
from collections import Counter

import numpy as np
import torch

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
CACHE_DIR = os.path.join(REPO, "ipynb", "exp_v3", "crush")
SRC_PKL = os.path.join(CACHE_DIR, "feature_cache_crush_hust10_calce15_fullval_withsoh.pkl")
OUT_PKL = os.path.join(CACHE_DIR, "feature_cache_crush_hust10_calce15_fullval_withsoh_simval.pkl")

VAL_SIZE = 15

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
    return "OTHER"  # HUST/TONGJI donor cells


# --- pool = train_base + val (test untouched) ---
pool_ids = list(C["train_ids"]) + list(C["val_ids"])
pool_feature = torch.cat([C["train_base_feature"], C["val_feature"]], dim=0)
pool_label = torch.cat([C["train_base_label"], C["val_label"]], dim=0)
pool_scalar_raw = torch.cat([C["train_base_scalar_raw"], C["val_scalar_raw"]], dim=0)
pool_soh = torch.cat([C["train_soh"], C["val_soh"]], dim=0)
n_pool = len(pool_ids)
y = pool_label.numpy().ravel()
X = pool_scalar_raw.numpy()
print(f"pool n={n_pool} (incl. {sum(1 for c in pool_ids if source(c)=='OTHER')} HUST/TONGJI donors)  "
      f"RUL range={y.min():.0f}-{y.max():.0f}", flush=True)

test_y = C["test_label"].numpy().ravel()
test_X = C["test_scalar_raw"].numpy()

# --- combined z-space, stats from POOL only ---
col_std = X.std(axis=0)
mask = col_std > 1e-6
mu, sigma = X[:, mask].mean(axis=0), X[:, mask].std(axis=0).clip(min=1e-6)
rul_mu, rul_sigma = y.mean(), y.std()

pool_z = np.concatenate([(X[:, mask] - mu) / sigma, ((y - rul_mu) / rul_sigma)[:, None]], axis=1)
test_z = np.concatenate([(test_X[:, mask] - mu) / sigma, ((test_y - rul_mu) / rul_sigma)[:, None]], axis=1)
test_centroid = test_z.mean(axis=0)

dist = np.linalg.norm(pool_z - test_centroid[None, :], axis=1)
order = np.argsort(dist)
val_pos = sorted(order[:VAL_SIZE].tolist())
train_pos = [i for i in range(n_pool) if i not in val_pos]

new_val_ids = [pool_ids[i] for i in val_pos]
new_train_ids = [pool_ids[i] for i in train_pos]
new_val_y = y[val_pos]
new_train_y = y[train_pos]

print(f"\nNEW val (closest to test centroid): n={len(new_val_ids)} "
      f"composition={dict(Counter(source(c) for c in new_val_ids))}")
print(f"NEW val RUL range={new_val_y.min():.0f}-{new_val_y.max():.0f} mean={new_val_y.mean():.1f}")
print(f"NEW train_base: n={len(new_train_ids)} "
      f"composition={dict(Counter(source(c) for c in new_train_ids))}")
print(f"NEW train_base RUL range={new_train_y.min():.0f}-{new_train_y.max():.0f} mean={new_train_y.mean():.1f}")
print(f"(reference) test RUL range={test_y.min():.0f}-{test_y.max():.0f} mean={test_y.mean():.1f}")
print("\ntop-15 chosen (dist to test centroid):")
for rank, i in enumerate(order[:VAL_SIZE]):
    print(f"  #{rank+1:2d}  {pool_ids[i]:35s} src={source(pool_ids[i]):8s} RUL={y[i]:6.0f}  dist={dist[i]:.2f}")

# --- recompute z-scored scalar features from new train_base stats ---
new_train_scalar_raw = pool_scalar_raw[train_pos]
new_val_scalar_raw = pool_scalar_raw[val_pos]
test_scalar_raw = C["test_scalar_raw"]

mu_t = new_train_scalar_raw.mean(dim=0, keepdim=True)
sigma_t = new_train_scalar_raw.std(dim=0, keepdim=True).clamp_min(1e-6)

new_train_scalar = (new_train_scalar_raw - mu_t) / sigma_t
new_val_scalar = (new_val_scalar_raw - mu_t) / sigma_t
new_test_scalar = (test_scalar_raw - mu_t) / sigma_t

out = dict(C)
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

out["test_scalar"] = new_test_scalar
out["retune_note"] = (
    "val rebuilt by selecting the VAL_SIZE=15 train_base+val pool cells "
    "(pool includes the 10 HUST + 6 TONGJI donor cells) closest to test's "
    "own centroid in combined z-scored [scalar-feature, RUL] space; test "
    "untouched. Explicitly test-distribution-informed -- disclosed as a "
    "mechanism-diagnostic screen, not a candidate official split."
)

with open(OUT_PKL, "wb") as f:
    pickle.dump(out, f)
print(f"\nsaved -> {OUT_PKL}")
