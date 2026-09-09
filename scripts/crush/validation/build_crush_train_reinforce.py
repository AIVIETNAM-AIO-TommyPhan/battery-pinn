"""Reinforce train_base after the random-val (20/70) rebuild dropped it to
n=50 -- the no-early-stop diagnostic on that split showed train/test RMSE
"exploding" together past epoch ~300 (val/test both climbing from ~250/370
to 500+/900+), plausibly worsened by the small train_base. Per explicit
request ("thêm train thôi" -- just add more train data instead of shrinking
train to grow val), this searches MATR2 + HUST + SNL(61-pool) + Tongji for
NEW donor cells (excluding anything already used anywhere in the current
CRUSH id sets, including the 10 HUST + 6 TONGJI donors already merged) that
are close to CRUSH's OWN train_base centroid in combined
[scalar-feature, RUL] z-space -- general reinforcement, not targeted at any
one weak source this time. val (the good random 20/70 split, r=0.868) and
test are left completely untouched.
"""
import os
import pickle
from collections import Counter

import numpy as np
import torch

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
CRUSH_DIR = os.path.join(REPO, "ipynb", "exp_v3", "crush")
CRUH_DIR = os.path.join(REPO, "ipynb", "exp_v3", "cruh")
TONGJI_DIR = os.path.join(REPO, "ipynb", "exp_v3", "tongji")
MATR_DIR = os.path.join(REPO, "ipynb", "exp_v3", "matr1_feature_cache")
HUST_DIR = os.path.join(REPO, "ipynb", "exp_v3", "hust")

N_DONORS = 20
SRC_PKL = os.path.join(CRUSH_DIR, "feature_cache_crush_hust10_calce15_fullval_withsoh_randval.pkl")
OUT_PKL = os.path.join(CRUSH_DIR, "feature_cache_crush_hust10_calce15_fullval_withsoh_randval_reinforced.pkl")

with open(SRC_PKL, "rb") as f:
    C = pickle.load(f)
feature_names = C["feature_names"]

already_used = set(C["train_ids"]) | set(C["val_ids"]) | set(C["test_ids"])
print(f"already-used CRUSH ids (train+val+test): {len(already_used)}", flush=True)


def normalize(cid):
    # cross-cache naming variants for the same physical cell, e.g. CRUSH's
    # bare HUST id "4-3" vs MATR2-pool's embedded "HUST_HUST_4-3" -- these
    # are NOT byte-identical in scalar_raw (MATR2's extractor populated
    # extra feature columns CRUSH's left as 0-padding for the same cell),
    # so signature-matching alone missed this; strip known prefixes instead.
    for pre in ("HUST_HUST_", "HUST_"):
        if cid.startswith(pre):
            return cid[len(pre):]
    return cid


already_used_norm = {normalize(c) for c in already_used}
print(f"already-used normalized ids: {len(already_used_norm)}", flush=True)

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
print(f"combined candidate pool (raw): {len(all_ids)} cells", flush=True)

valid_mask = ~np.isnan(all_label)
not_used_id = np.array([cid not in already_used for cid in all_ids])
not_used_norm = np.array([normalize(cid) not in already_used_norm for cid in all_ids])
keep_mask = valid_mask & not_used_id & not_used_norm
print(f"excluded by id-match: {(~not_used_id & valid_mask).sum()}, "
      f"excluded by normalized-id-match (id looked new): {(not_used_id & ~not_used_norm & valid_mask).sum()}", flush=True)
all_feature = all_feature[keep_mask]
all_scalar = all_scalar[keep_mask]
all_label = all_label[keep_mask]
all_ids = [c for c, k in zip(all_ids, keep_mask) if k]
all_source = [s for s, k in zip(all_source, keep_mask) if k]
print(f"candidate pool after excluding used-ids/sigs + NaN labels: {len(all_ids)} cells "
      f"({dict(Counter(all_source))})", flush=True)

# also dedup WITHIN the remaining candidate pool itself (e.g. MATR2's cache
# embeds HUST cells under a "HUST_HUST_*" alias, which can collide with the
# separately-loaded HUST pool's own copy of the same physical cell).
seen_norm = set()
internal_keep = []
for i, cid in enumerate(all_ids):
    n = normalize(cid)
    if n in seen_norm:
        continue
    seen_norm.add(n)
    internal_keep.append(i)
n_before = len(all_ids)
all_feature = all_feature[internal_keep]
all_scalar = all_scalar[internal_keep]
all_label = all_label[internal_keep]
all_ids = [all_ids[i] for i in internal_keep]
all_source = [all_source[i] for i in internal_keep]
print(f"after internal cross-pool dedup: {n_before} -> {len(all_ids)}", flush=True)

# --- CRUSH's own train_base centroid (combined z-space, CRUSH train_base stats) ---
train_scalar = C["train_base_scalar_raw"].numpy()
train_label = C["train_base_label"].numpy().ravel()
col_std = train_scalar.std(axis=0)
mask = col_std > 1e-6
mu, sigma = train_scalar[:, mask].mean(axis=0), train_scalar[:, mask].std(axis=0).clip(min=1e-6)
rul_mu, rul_sigma = train_label.mean(), train_label.std()

train_z = np.concatenate([(train_scalar[:, mask] - mu) / sigma, ((train_label - rul_mu) / rul_sigma)[:, None]], axis=1)
centroid = train_z.mean(axis=0)

cand_z = np.concatenate([(all_scalar[:, mask] - mu) / sigma, ((all_label - rul_mu) / rul_sigma)[:, None]], axis=1)
dist = np.linalg.norm(cand_z - centroid[None, :], axis=1)
order = np.argsort(dist)
top_idx = order[:N_DONORS]

print(f"\n=== top {N_DONORS} general-reinforcement donors (dist to CRUSH train_base centroid) ===")
for rank, i in enumerate(top_idx):
    print(f"  #{rank+1:2d}  {all_ids[i]:30s} src={all_source[i]:8s} RUL={all_label[i]:6.0f}  dist={dist[i]:.2f}")

donor_ids = [all_ids[i] for i in top_idx]
donor_source = [all_source[i] for i in top_idx]
donor_feature = all_feature[top_idx]
donor_label = torch.tensor(all_label[top_idx], dtype=torch.float32)
donor_scalar_raw = torch.tensor(all_scalar[top_idx], dtype=torch.float32)
donor_soh = torch.ones(len(top_idx), C["train_soh"].shape[1], dtype=torch.float32)  # no cycle-level SOH available cheaply for cross-dataset donors; matches build_crush_soh_cache's own missing-cell fallback (constant 1.0)

new_train_ids = list(C["train_ids"]) + donor_ids
new_train_feature = torch.cat([C["train_base_feature"], donor_feature], dim=0)
new_train_label = torch.cat([C["train_base_label"], donor_label], dim=0)
new_train_scalar_raw = torch.cat([C["train_base_scalar_raw"], donor_scalar_raw], dim=0)
new_train_soh = torch.cat([C["train_soh"], donor_soh], dim=0)

mu_t = new_train_scalar_raw.mean(dim=0, keepdim=True)
sigma_t = new_train_scalar_raw.std(dim=0, keepdim=True).clamp_min(1e-6)

out = dict(C)
out["train_ids"] = new_train_ids
out["train_base_feature"] = new_train_feature
out["train_base_label"] = new_train_label
out["train_base_scalar_raw"] = new_train_scalar_raw
out["train_base_scalar"] = (new_train_scalar_raw - mu_t) / sigma_t
out["train_soh"] = new_train_soh
out["val_scalar"] = (C["val_scalar_raw"] - mu_t) / sigma_t
out["test_scalar"] = (C["test_scalar_raw"] - mu_t) / sigma_t
out["reinforce_note"] = (
    f"train_base reinforced with {N_DONORS} new cross-dataset donors "
    "(MATR2/HUST/SNL/Tongji, closest to CRUSH train_base's own centroid in "
    "combined feature+RUL z-space, excluding all ids already used anywhere "
    "in CRUSH incl. prior donors). val (random 20/70, r=0.868) and test "
    "untouched. Motivated by train/test RMSE instability seen with "
    "train_base=50 in the no-early-stop diagnostic."
)

print(f"\nNEW train_base: n={len(new_train_ids)} (was {len(C['train_ids'])})")
with open(OUT_PKL, "wb") as f:
    pickle.dump(out, f)
print(f"saved -> {OUT_PKL}")
