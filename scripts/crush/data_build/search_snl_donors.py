"""Search MATR2 + HUST for cells feature+RUL-close to SNL's own centroid
(combined z-distance, the original HUST-style method -- not the CALCE-style
features-only + RUL-range-filter variant, since here we're testing general
"more density helps" rather than filling a specific extrapolation gap).
"""
import os
import sys
import pickle

import numpy as np
import torch

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
CRUSH_DIR = os.path.join(REPO, "ipynb", "exp_v3", "crush")
MATR_DIR = os.path.join(REPO, "ipynb", "exp_v3", "matr1_feature_cache")
HUST_DIR = os.path.join(REPO, "ipynb", "exp_v3", "hust")
N_DONORS = 25

with open(os.path.join(CRUSH_DIR, "feature_cache_crush_fromraw_final.pkl"), "rb") as f:
    C = pickle.load(f)
feature_names = C["feature_names"]

def source(cid):
    if cid.startswith("SNL"):
        return "SNL"
    return "OTHER"

train_ids = C["train_ids"]
snl_idx = [i for i, c in enumerate(train_ids) if source(c) == "SNL"]
snl_scalar = C["train_base_scalar_raw"][snl_idx].numpy()
snl_rul = C["train_base_label"][snl_idx].numpy().ravel()
print(f"SNL anchor: n={len(snl_idx)}  RUL range=[{snl_rul.min():.0f},{snl_rul.max():.0f}]", flush=True)

pools = []
SPLIT_KEYS = [("train_base", "train"), ("val", "val"), ("test", "test")]
with open(os.path.join(MATR_DIR, "feature_cache_matr2.pkl"), "rb") as f:
    d = pickle.load(f)
    for split, idk in SPLIT_KEYS:
        pools.append(dict(source="MATR2", feature=d[f"{split}_feature"], label=d[f"{split}_label"].numpy().ravel(),
                           scalar_raw=d[f"{split}_scalar_raw"].numpy(), ids=d[f"{idk}_ids"], names=d["feature_names"]))
with open(os.path.join(HUST_DIR, "feature_cache_hust.pkl"), "rb") as f:
    d = pickle.load(f)
    for split, idk in SPLIT_KEYS:
        pools.append(dict(source="HUST", feature=d[f"{split}_feature"], label=d[f"{split}_label"].numpy().ravel(),
                           scalar_raw=d[f"{split}_scalar_raw"].numpy(), ids=d[f"{idk}_ids"], names=d["feature_names"]))

for p in pools:
    assert p["names"] == feature_names

all_feature = torch.cat([p["feature"] for p in pools], dim=0)
all_scalar = np.concatenate([p["scalar_raw"] for p in pools], axis=0)
all_label = np.concatenate([p["label"] for p in pools], axis=0)
all_ids = sum([list(p["ids"]) for p in pools], [])
all_source = sum([[p["source"]] * len(p["ids"]) for p in pools], [])
print(f"combined candidate pool: {len(all_ids)} cells", flush=True)

combined_for_var = np.concatenate([snl_scalar, all_scalar], axis=0)
mask = combined_for_var.std(axis=0) > 1e-6
snl_feat = snl_scalar[:, mask]
mu, sigma = snl_feat.mean(axis=0), snl_feat.std(axis=0).clip(min=1e-6)
rul_mu, rul_sigma = snl_rul.mean(), snl_rul.std()

snl_combined_z = np.concatenate([(snl_feat - mu) / sigma, ((snl_rul - rul_mu) / rul_sigma)[:, None]], axis=1)
centroid = snl_combined_z.mean(axis=0)

cand_feat_z = (all_scalar[:, mask] - mu) / sigma
cand_rul_z = (all_label - rul_mu) / rul_sigma
cand_combined_z = np.concatenate([cand_feat_z, cand_rul_z[:, None]], axis=1)
dist = np.linalg.norm(cand_combined_z - centroid[None, :], axis=1)

order = np.argsort(dist)
top_idx = order[:N_DONORS]
print(f"\n=== top {N_DONORS} SNL-like donors (combined feature+RUL distance) ===")
for rank, i in enumerate(top_idx):
    print(f"  #{rank+1}  {all_ids[i]:30s} src={all_source[i]:8s} RUL={all_label[i]:6.0f}  dist={dist[i]:.2f}")

donor_ids = [all_ids[i] for i in top_idx]
donor_source = [all_source[i] for i in top_idx]
donor_feature = all_feature[top_idx]
donor_label = torch.tensor(all_label[top_idx], dtype=torch.float32)
donor_scalar_raw = torch.tensor(all_scalar[top_idx], dtype=torch.float32)

with open(os.path.join(CRUSH_DIR, "snl_donors_top10.pkl"), "wb") as f:
    pickle.dump(dict(donor_ids=donor_ids, donor_source=donor_source, donor_feature=donor_feature,
                      donor_label=donor_label, donor_scalar_raw=donor_scalar_raw, feature_names=feature_names), f)
print(f"\nsaved -> {os.path.join(CRUSH_DIR, 'snl_donors_top10.pkl')}")
