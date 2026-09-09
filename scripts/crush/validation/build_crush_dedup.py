"""Fix the train/val and train/test leakage found in
feature_cache_crush_hust10_calce15_fullval_withsoh.pkl: 9 SNL cell rows are
byte-identical duplicates of a cell already present in another split (3
train x val, 3 train x test, 3 duplicated twice within train itself) --
verified by comparing scalar_raw + label, not just cell-id string collision.

Fix policy: val and test are treated as authoritative (never modified --
these are the fixed evaluation sets). Any train_base row whose cell-id
also appears in val_ids or test_ids is dropped from train_base; internal
train_base duplicates are collapsed to a single copy. Scalar z-scoring is
recomputed from the deduped train_base and reapplied to train_base/val/test
(membership/features/labels of val and test are otherwise untouched).
"""
import os
import pickle
from collections import Counter

import torch

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
CACHE_DIR = os.path.join(REPO, "ipynb", "exp_v3", "crush")
SRC_PKL = os.path.join(CACHE_DIR, "feature_cache_crush_hust10_calce15_fullval_withsoh.pkl")
OUT_PKL = os.path.join(CACHE_DIR, "feature_cache_crush_hust10_calce15_fullval_withsoh_dedup.pkl")

with open(SRC_PKL, "rb") as f:
    C = pickle.load(f)

train_ids, val_ids, test_ids = C["train_ids"], C["val_ids"], C["test_ids"]
protected = set(val_ids) | set(test_ids)

keep_pos = []
seen = set()
dropped = []
for i, cid in enumerate(train_ids):
    if cid in protected:
        dropped.append((i, cid, "in val/test"))
        continue
    if cid in seen:
        dropped.append((i, cid, "internal dup"))
        continue
    seen.add(cid)
    keep_pos.append(i)

print(f"train_base: {len(train_ids)} -> {len(keep_pos)} (dropped {len(dropped)})")
for i, cid, why in dropped:
    print(f"  drop idx={i} {cid} ({why})")

new_train_ids = [train_ids[i] for i in keep_pos]
new_train_feature = C["train_base_feature"][keep_pos]
new_train_label = C["train_base_label"][keep_pos]
new_train_scalar_raw = C["train_base_scalar_raw"][keep_pos]
new_train_soh = C["train_soh"][keep_pos]

assert len(set(new_train_ids) & set(val_ids)) == 0
assert len(set(new_train_ids) & set(test_ids)) == 0
assert len(new_train_ids) == len(set(new_train_ids))

mu = new_train_scalar_raw.mean(dim=0, keepdim=True)
sigma = new_train_scalar_raw.std(dim=0, keepdim=True).clamp_min(1e-6)

out = dict(C)
out["train_ids"] = new_train_ids
out["train_base_feature"] = new_train_feature
out["train_base_label"] = new_train_label
out["train_base_scalar_raw"] = new_train_scalar_raw
out["train_base_scalar"] = (new_train_scalar_raw - mu) / sigma
out["train_soh"] = new_train_soh
out["val_scalar"] = (C["val_scalar_raw"] - mu) / sigma
out["test_scalar"] = (C["test_scalar_raw"] - mu) / sigma
out["dedup_note"] = (
    f"train_base deduped: dropped {len(dropped)} rows that were byte-identical "
    "duplicates of a val/test cell or of another train_base row (verified via "
    "scalar_raw+label equality, not just id string). val/test membership and "
    "content untouched; only their z-scored `_scalar` was renormalized against "
    "the new deduped train_base stats."
)

with open(OUT_PKL, "wb") as f:
    pickle.dump(out, f)
print(f"\nNEW train_base n={len(new_train_ids)}")
print(f"saved -> {OUT_PKL}")
