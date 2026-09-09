"""Search MATR2 + HUST + SNL + Tongji for cells feature+RUL-close to CALCE's
OWN centroid (combined z-distance, original HUST-style method) -- unlike the
earlier CRUH search (which deliberately targeted a HIGH-RUL range to fill an
extrapolation gap), here CALCE's train_base already covers test's RUL range
reasonably (train 221-337 vs test 144-382) -- the real problem is just n=3
train examples, too few to resist the asymmetric loss's upward pull. So we
want cells similar to CALCE's OWN distribution (not a specific target range)
to reinforce its low-RUL representation.
"""
import os
import sys
import pickle

import numpy as np
import torch

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
CRUSH_DIR = os.path.join(REPO, "ipynb", "exp_v3", "crush")
CRUH_DIR = os.path.join(REPO, "ipynb", "exp_v3", "cruh")
TONGJI_DIR = os.path.join(REPO, "ipynb", "exp_v3", "tongji")
MATR_DIR = os.path.join(REPO, "ipynb", "exp_v3", "matr1_feature_cache")
HUST_DIR = os.path.join(REPO, "ipynb", "exp_v3", "hust")
N_DONORS = 15

with open(os.path.join(CRUSH_DIR, "feature_cache_crush_hust10_fullval.pkl"), "rb") as f:
    C = pickle.load(f)
feature_names = C["feature_names"]


def source(cid):
    if cid.startswith("CALCE"):
        return "CALCE"
    return "OTHER"


train_ids = C["train_ids"]
calce_idx = [i for i, c in enumerate(train_ids) if source(c) == "CALCE"]
calce_scalar = C["train_base_scalar_raw"][calce_idx].numpy()
calce_rul = C["train_base_label"][calce_idx].numpy().ravel()
print(f"CALCE anchor: n={len(calce_idx)}  RUL={sorted(calce_rul.round(0))}", flush=True)

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
with open(os.path.join(CRUH_DIR, "feature_cache_snl.pkl"), "rb") as f:
    d = pickle.load(f)
    pools.append(dict(source="SNL", feature=d["feature"], label=d["label"].numpy().ravel(),
                       scalar_raw=d["scalar_raw"].numpy(), ids=d["cell_ids"], names=d["feature_names"]))
with open(os.path.join(TONGJI_DIR, "feature_cache_tongji.pkl"), "rb") as f:
    d = pickle.load(f)
    pools.append(dict(source="Tongji", feature=d["feature"], label=d["label"].numpy().ravel(),
                       scalar_raw=d["scalar_raw"].numpy(), ids=d["cell_ids"], names=d["feature_names"]))

for p in pools:
    assert p["names"] == feature_names, f"schema mismatch: {p['source']}"

all_feature = torch.cat([p["feature"] for p in pools], dim=0)
all_scalar = np.concatenate([p["scalar_raw"] for p in pools], axis=0)
all_label = np.concatenate([p["label"] for p in pools], axis=0)
all_ids = sum([list(p["ids"]) for p in pools], [])
all_source = sum([[p["source"]] * len(p["ids"]) for p in pools], [])
print(f"combined candidate pool: {len(all_ids)} cells", flush=True)

# exclude cells already used as HUST donors for SNL reinforcement (avoid double-dipping)
with open(os.path.join(CRUSH_DIR, "snl_donors_top10.pkl"), "rb") as f:
    already_used = set(pickle.load(f)["donor_ids"])
keep_mask = np.array([cid not in already_used for cid in all_ids])
all_feature = all_feature[keep_mask]
all_scalar = all_scalar[keep_mask]
all_label = all_label[keep_mask]
all_ids = [c for c, k in zip(all_ids, keep_mask) if k]
all_source = [s for s, k in zip(all_source, keep_mask) if k]
print(f"after excluding already-used SNL-donor HUST cells: {len(all_ids)} cells", flush=True)

combined_for_var = np.concatenate([calce_scalar, all_scalar], axis=0)
mask = combined_for_var.std(axis=0) > 1e-6
calce_feat = calce_scalar[:, mask]
mu, sigma = calce_feat.mean(axis=0), calce_feat.std(axis=0).clip(min=1e-6)
rul_mu, rul_sigma = calce_rul.mean(), calce_rul.std()

calce_combined_z = np.concatenate([(calce_feat - mu) / sigma, ((calce_rul - rul_mu) / rul_sigma)[:, None]], axis=1)
centroid = calce_combined_z.mean(axis=0)

cand_feat_z = (all_scalar[:, mask] - mu) / sigma
cand_rul_z = (all_label - rul_mu) / rul_sigma
cand_combined_z = np.concatenate([cand_feat_z, cand_rul_z[:, None]], axis=1)
dist = np.linalg.norm(cand_combined_z - centroid[None, :], axis=1)

order = np.argsort(dist)
top_idx = order[:N_DONORS]
print(f"\n=== top {N_DONORS} CALCE-reinforcing donors (combined feature+RUL distance) ===")
for rank, i in enumerate(top_idx):
    print(f"  #{rank+1}  {all_ids[i]:30s} src={all_source[i]:8s} RUL={all_label[i]:6.0f}  dist={dist[i]:.2f}")

donor_ids = [all_ids[i] for i in top_idx]
donor_source = [all_source[i] for i in top_idx]
donor_feature = all_feature[top_idx]
donor_label = torch.tensor(all_label[top_idx], dtype=torch.float32)
donor_scalar_raw = torch.tensor(all_scalar[top_idx], dtype=torch.float32)

with open(os.path.join(CRUSH_DIR, "calce_reinforce_donors_top15.pkl"), "wb") as f:
    pickle.dump(dict(donor_ids=donor_ids, donor_source=donor_source, donor_feature=donor_feature,
                      donor_label=donor_label, donor_scalar_raw=donor_scalar_raw, feature_names=feature_names), f)
print(f"\nsaved -> {os.path.join(CRUSH_DIR, 'calce_reinforce_donors_top15.pkl')}")
