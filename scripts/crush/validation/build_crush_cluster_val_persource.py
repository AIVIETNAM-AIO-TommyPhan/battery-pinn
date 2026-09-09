"""CRUSH val construction v3: cluster EACH source separately (own z-score
scale, own K via silhouette), instead of one K-means over the whole mixed
54-cell pool. Fixes the diagnosed problem: a single mixed clustering put
almost all of RWTH into one big homogeneous cluster (cluster0, size 29 --
exactly RWTH's count) while SNL's much wider RUL range got fragmented into
several tiny mixed clusters shared with CALCE/UL_PUR, so a fixed
n_per_cluster or 50%-cap rule either starved SNL's train_base (n=4 ceiling)
or barely changed anything (capped version). Clustering SNL on its own scale
lets its own K-means find structure across its actual RUL range (34-2169)
instead of that range being swamped by cross-source feature differences.

Each source's cluster draw is capped at 50% per cluster (train_base always
keeps >=half of every within-source cluster), same principle as
build_crush_cluster_val_capped.py, just applied per-source instead of
globally.
"""
import os
import sys
import pickle
from collections import Counter

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
sys.path.insert(0, REPO)

CACHE_DIR = os.path.join(REPO, "ipynb", "exp_v3", "crush")
BASE_PKL = os.path.join(CACHE_DIR, "feature_cache_crush_base.pkl")
SPLIT_SEED = 0
N_PER_CLUSTER_CEILING = 3


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


with open(BASE_PKL, "rb") as f:
    B = pickle.load(f)

all_ids = B["train_pool_ids"]
all_scalar_raw = B["train_pool_scalar_raw"].numpy()
all_label = B["train_pool_label"].numpy().ravel()
feature_names = B["feature_names"]

valid_mask = ~np.isnan(all_label)
train_ids = [i for i, m in zip(all_ids, valid_mask) if m]
train_scalar_raw = all_scalar_raw[valid_mask]
train_label = all_label[valid_mask]
n_train = len(train_ids)

col_std = train_scalar_raw.std(axis=0)
nondegenerate_mask = col_std > 1e-6

src_of = [source(c) for c in train_ids]
print("train_pool (valid) composition:", dict(Counter(src_of)), flush=True)

val_idx_global = []
for src in sorted(set(src_of)):
    idxs = [i for i, s in enumerate(src_of) if s == src]
    n_src = len(idxs)
    X = train_scalar_raw[idxs][:, nondegenerate_mask]
    y = train_label[idxs]

    if n_src < 4:
        # too few to cluster meaningfully -- keep all in train_base
        print(f"  {src}: n={n_src} (too few to cluster, all -> train_base)", flush=True)
        continue

    X_z = (X - X.mean(axis=0)) / X.std(axis=0).clip(min=1e-6)
    y_z = (y - y.mean()) / y.std().clip(min=1e-6) if y.std() > 1e-6 else np.zeros_like(y)
    combined = np.concatenate([X_z, y_z[:, None]], axis=1)

    best_k, best_sil, best_labels = None, -1.0, None
    for k in range(2, min(7, n_src)):
        km = KMeans(n_clusters=k, random_state=SPLIT_SEED, n_init=10)
        cl = km.fit_predict(combined)
        if len(set(cl)) < 2:
            continue
        sil = silhouette_score(combined, cl)
        if sil > best_sil:
            best_k, best_sil, best_labels = k, sil, cl

    if best_k is None:
        print(f"  {src}: n={n_src} (no valid clustering found, all -> train_base)", flush=True)
        continue

    rng = np.random.RandomState(SPLIT_SEED)
    src_val_local = []
    cluster_sizes = {}
    for c in range(best_k):
        members = np.where(best_labels == c)[0]
        cluster_sizes[c] = len(members)
        cap = int(np.floor(len(members) * 0.5))
        take = min(N_PER_CLUSTER_CEILING, cap)
        if take > 0:
            chosen = rng.choice(members, size=take, replace=False)
            src_val_local.extend(chosen.tolist())
    src_val_global = [idxs[i] for i in src_val_local]
    val_idx_global.extend(src_val_global)
    print(f"  {src}: n={n_src}  K={best_k} (silhouette={best_sil:.3f})  "
          f"cluster_sizes={cluster_sizes}  val={len(src_val_global)}  train_base={n_src-len(src_val_global)}", flush=True)

val_idx = sorted(val_idx_global)
train_base_idx = [i for i in range(n_train) if i not in val_idx]
val_ids = [train_ids[i] for i in val_idx]
train_base_ids = [train_ids[i] for i in train_base_idx]

print(f"\nTOTAL: val={len(val_ids)}  train_base={len(train_base_ids)}")
print("train_base composition:", dict(Counter(source(c) for c in train_base_ids)))
print("val composition:       ", dict(Counter(source(c) for c in val_ids)))

valid_test_mask = ~np.isnan(B["test_label"].numpy().ravel())
test_ids_valid = [i for i, m in zip(B["test_ids"], valid_test_mask) if m]
test_feature_valid = B["test_feature"][valid_test_mask]
test_label_valid = B["test_label"][valid_test_mask]
test_scalar_raw_valid = B["test_scalar_raw"][valid_test_mask]

train_pool_feature_valid = B["train_pool_feature"][valid_mask]
train_pool_label_valid = torch.tensor(train_label, dtype=torch.float32)
train_pool_scalar_raw_valid = torch.tensor(train_scalar_raw, dtype=torch.float32)

train_base_feature = train_pool_feature_valid[train_base_idx]
train_base_label = train_pool_label_valid[train_base_idx]
train_base_scalar_raw = train_pool_scalar_raw_valid[train_base_idx]
val_feature = train_pool_feature_valid[val_idx]
val_label = train_pool_label_valid[val_idx]
val_scalar_raw = train_pool_scalar_raw_valid[val_idx]

_mean = torch.nan_to_num(train_base_scalar_raw.nanmean(dim=0), nan=0.0)
train_base_f = torch.where(torch.isnan(train_base_scalar_raw), _mean.expand_as(train_base_scalar_raw), train_base_scalar_raw)
val_f = torch.where(torch.isnan(val_scalar_raw), _mean.expand_as(val_scalar_raw), val_scalar_raw)
test_f = torch.where(torch.isnan(test_scalar_raw_valid), _mean.expand_as(test_scalar_raw_valid), test_scalar_raw_valid)
_std = train_base_f.std(dim=0).clamp_min(1e-6)

C = {
    "train_base_feature": train_base_feature, "train_base_label": train_base_label,
    "val_feature": val_feature, "val_label": val_label,
    "test_feature": test_feature_valid, "test_label": test_label_valid,
    "train_base_scalar": (train_base_f - _mean) / _std, "val_scalar": (val_f - _mean) / _std, "test_scalar": (test_f - _mean) / _std,
    "train_base_scalar_raw": train_base_f, "val_scalar_raw": val_f, "test_scalar_raw": test_f,
    "feature_names": feature_names,
    "train_ids": train_base_ids, "val_ids": val_ids, "test_ids": test_ids_valid,
}
out_pkl = os.path.join(CACHE_DIR, "feature_cache_crush_cluster_persource.pkl")
with open(out_pkl, "wb") as f:
    pickle.dump(C, f)
print(f"\nsaved -> {out_pkl}")
