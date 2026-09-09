"""SNL standalone val construction: same K-means-on-[non-degenerate
feature]+[RUL] z-score method validated on HUST/CRUH/CRUSH, applied to
SNL's own official 36-cell train pool (only 16 valid-labeled after the
RUL<=100 filter -- very tight for clustering, K range kept small).

Step 1: sweep K=2..5 by silhouette on the 16 valid train cells.
Step 2: sweep n_per_cluster in {1,2} (16 cells is far smaller than
HUST/CRUH/CRUSH's pools -- can't afford a 3+/cluster draw).
"""
import os
import sys
import pickle

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
sys.path.insert(0, REPO)

CACHE_DIR = os.path.join(REPO, "ipynb", "exp_v3", "snl")
BASE_PKL = os.path.join(CACHE_DIR, "feature_cache_snl_base.pkl")

K_RANGE = range(2, 6)
N_PER_CLUSTER_SWEEP = [1, 2]
SPLIT_SEED = 0

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
print(f"valid train cells: {n_train}/{len(all_ids)}", flush=True)

col_std = train_scalar_raw.std(axis=0)
nondegenerate_mask = col_std > 1e-6
nondeg_names = [n for n, m in zip(feature_names, nondegenerate_mask) if m]
print(f"non-degenerate features: {len(nondeg_names)}/{len(feature_names)}", flush=True)

X_feat = train_scalar_raw[:, nondegenerate_mask]
X_feat_z = (X_feat - X_feat.mean(axis=0)) / X_feat.std(axis=0).clip(min=1e-6)
y_z = (train_label - train_label.mean()) / train_label.std()
combined = np.concatenate([X_feat_z, y_z[:, None]], axis=1)

best_k, best_sil, best_labels = None, -1.0, None
for k in K_RANGE:
    if k >= n_train:
        continue
    km = KMeans(n_clusters=k, random_state=SPLIT_SEED, n_init=10)
    cluster_labels = km.fit_predict(combined)
    if len(set(cluster_labels)) < 2:
        continue
    sil = silhouette_score(combined, cluster_labels)
    print(f"K={k}  silhouette={sil:.3f}", flush=True)
    if sil > best_sil:
        best_k, best_sil, best_labels = k, sil, cluster_labels

print(f"\nSelected K={best_k}  silhouette={best_sil:.3f}", flush=True)
cluster_sizes = {c: int((best_labels == c).sum()) for c in range(best_k)}
print(f"cluster sizes: {cluster_sizes}", flush=True)

valid_test_mask = ~np.isnan(B["test_label"].numpy().ravel())
test_ids_valid = [i for i, m in zip(B["test_ids"], valid_test_mask) if m]
test_feature_valid = B["test_feature"][valid_test_mask]
test_label_valid = B["test_label"][valid_test_mask]
test_scalar_raw_valid = B["test_scalar_raw"][valid_test_mask]

train_pool_feature_valid = B["train_pool_feature"][valid_mask]
train_pool_label_valid = torch.tensor(train_label, dtype=torch.float32)
train_pool_scalar_raw_valid = torch.tensor(train_scalar_raw, dtype=torch.float32)

for n_per_cluster in N_PER_CLUSTER_SWEEP:
    rng = np.random.RandomState(SPLIT_SEED)
    val_idx = []
    for c in range(best_k):
        members = np.where(best_labels == c)[0]
        take = min(n_per_cluster, len(members))
        chosen = rng.choice(members, size=take, replace=False)
        val_idx.extend(chosen.tolist())
    val_idx = sorted(val_idx)
    train_base_idx = [i for i in range(n_train) if i not in val_idx]

    val_ids = [train_ids[i] for i in val_idx]
    train_base_ids = [train_ids[i] for i in train_base_idx]

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
        "cluster_K": best_k, "cluster_silhouette": best_sil, "n_per_cluster": n_per_cluster,
    }
    out_pkl = os.path.join(CACHE_DIR, f"feature_cache_snl_cluster_n{n_per_cluster}.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(C, f)
    print(f"n_per_cluster={n_per_cluster}  val={len(val_ids)}  train_base={len(train_base_ids)}  "
          f"val_RUL(mean={val_label.mean():.1f})  train_base_RUL(mean={train_base_label.mean():.1f})  "
          f"-> {out_pkl}", flush=True)
