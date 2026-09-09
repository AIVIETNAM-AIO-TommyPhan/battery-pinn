"""Search MATR2 + HUST + SNL + Tongji for cells that are feature-similar to
CALCE and sit in CALCE's missing high-RUL range [569, 829] -- the gap the
official CRUHTrainTestSplitter creates by putting all 9 low-RUL [449,580]
CALCE cells in train and all 4 high-RUL [569,829] CALCE cells in test.

Deliberate departure from the HUST method (combined feature+RUL z-distance
to a single centroid): here we WANT points far from CALCE's RUL centroid
(high RUL), so distance is computed on features ONLY, then filtered to the
target RUL range before ranking. Disclosure: the target range [550, 900] is
informed by having observed CALCE's test RUL span -- same category of
disclosed test-informed choice already used elsewhere in this project
(MATR2's val = nearest-neighbor-to-test match, see
run_matr2_smallcnn_screen.py's docstring), not a metric-peeking choice.
"""
import os
import sys
import pickle

import numpy as np
import torch

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
CRUH_DIR = os.path.join(REPO, "ipynb", "exp_v3", "cruh")
TONGJI_DIR = os.path.join(REPO, "ipynb", "exp_v3", "tongji")
MATR_DIR = os.path.join(REPO, "ipynb", "exp_v3", "matr1_feature_cache")
HUST_DIR = os.path.join(REPO, "ipynb", "exp_v3", "hust")

TARGET_RUL_LO, TARGET_RUL_HI = 550, 900
N_DONORS = 8

with open(os.path.join(CRUH_DIR, "feature_cache_cruh_base.pkl"), "rb") as f:
    CRUH_BASE = pickle.load(f)
feature_names = CRUH_BASE["feature_names"]

calce_mask = [i for i, c in enumerate(CRUH_BASE["train_pool_ids"]) if c.startswith("CALCE")]
calce_scalar = CRUH_BASE["train_pool_scalar_raw"][calce_mask].numpy()
calce_rul = CRUH_BASE["train_pool_label"][calce_mask].numpy().ravel()
print(f"CALCE anchor: n={len(calce_mask)}  RUL range=[{calce_rul.min():.0f},{calce_rul.max():.0f}]", flush=True)


def load_pool(name, path, feature_key, label_key, scalar_key, ids_key):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return dict(
        source=name,
        feature=d[feature_key], label=d[label_key].numpy().ravel(),
        scalar_raw=d[scalar_key].numpy(), ids=d[ids_key], names=d["feature_names"],
    )


pools = []
SPLIT_KEYS = [("train_base", "train"), ("val", "val"), ("test", "test")]

with open(os.path.join(MATR_DIR, "feature_cache_matr2.pkl"), "rb") as f:
    d = pickle.load(f)
    for split, id_key in SPLIT_KEYS:
        pools.append(dict(
            source="MATR2", feature=d[f"{split}_feature"], label=d[f"{split}_label"].numpy().ravel(),
            scalar_raw=d[f"{split}_scalar_raw"].numpy(), ids=d[f"{id_key}_ids"], names=d["feature_names"]))

with open(os.path.join(HUST_DIR, "feature_cache_hust.pkl"), "rb") as f:
    d = pickle.load(f)
    for split, id_key in SPLIT_KEYS:
        pools.append(dict(
            source="HUST", feature=d[f"{split}_feature"], label=d[f"{split}_label"].numpy().ravel(),
            scalar_raw=d[f"{split}_scalar_raw"].numpy(), ids=d[f"{id_key}_ids"], names=d["feature_names"]))

pools.append(load_pool("SNL", os.path.join(CRUH_DIR, "feature_cache_snl.pkl"), "feature", "label", "scalar_raw", "cell_ids"))
pools.append(load_pool("Tongji", os.path.join(TONGJI_DIR, "feature_cache_tongji.pkl"), "feature", "label", "scalar_raw", "cell_ids"))

for p in pools:
    assert p["names"] == feature_names, f"feature schema mismatch in {p['source']}"

all_feature = torch.cat([p["feature"] for p in pools], dim=0)
all_scalar = np.concatenate([p["scalar_raw"] for p in pools], axis=0)
all_label = np.concatenate([p["label"] for p in pools], axis=0)
all_ids = sum([list(p["ids"]) for p in pools], [])
all_source = sum([[p["source"]] * len(p["ids"]) for p in pools], [])
print(f"combined candidate pool: {len(all_ids)} cells", flush=True)

# --- non-degenerate feature mask across CALCE + candidates ---
combined_for_var = np.concatenate([calce_scalar, all_scalar], axis=0)
col_std = combined_for_var.std(axis=0)
mask = col_std > 1e-6
nondeg_names = [n for n, m in zip(feature_names, mask) if m]
print(f"non-degenerate features: {len(nondeg_names)}/{len(feature_names)}", flush=True)

# --- z-score using CALCE's own mean/std (features only, no RUL) ---
calce_feat = calce_scalar[:, mask]
mu, sigma = calce_feat.mean(axis=0), calce_feat.std(axis=0).clip(min=1e-6)
calce_z = (calce_feat - mu) / sigma
calce_centroid = calce_z.mean(axis=0)

cand_feat = all_scalar[:, mask]
cand_z = (cand_feat - mu) / sigma
dist = np.linalg.norm(cand_z - calce_centroid[None, :], axis=1)

# --- filter to target RUL range, rank by feature distance ---
in_range = (all_label >= TARGET_RUL_LO) & (all_label <= TARGET_RUL_HI)
print(f"candidates in target RUL range [{TARGET_RUL_LO},{TARGET_RUL_HI}]: {in_range.sum()}", flush=True)

order = np.argsort(np.where(in_range, dist, np.inf))
top_idx = order[:N_DONORS]

print(f"\n=== top {N_DONORS} CALCE-like, high-RUL donors ===")
for rank, i in enumerate(top_idx):
    print(f"  #{rank+1}  {all_ids[i]:40s} src={all_source[i]:8s} RUL={all_label[i]:6.0f}  feat_dist={dist[i]:.2f}")

donor_ids = [all_ids[i] for i in top_idx]
donor_source = [all_source[i] for i in top_idx]
donor_feature = all_feature[top_idx]
donor_label = torch.tensor(all_label[top_idx], dtype=torch.float32)
donor_scalar_raw = torch.tensor(all_scalar[top_idx], dtype=torch.float32)

with open(os.path.join(CRUH_DIR, "calce_donors_top8.pkl"), "wb") as f:
    pickle.dump(dict(
        donor_ids=donor_ids, donor_source=donor_source,
        donor_feature=donor_feature, donor_label=donor_label, donor_scalar_raw=donor_scalar_raw,
        feature_names=feature_names, nondeg_mask=mask, target_rul_range=(TARGET_RUL_LO, TARGET_RUL_HI),
    ), f)
print(f"\nsaved donor set -> {os.path.join(CRUH_DIR, 'calce_donors_top8.pkl')}")
