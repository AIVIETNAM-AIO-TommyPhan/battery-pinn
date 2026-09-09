"""Build several alternative MATR2 val-set variants, reusing the already-extracted
per-cell CNN tensors + raw scalar features from feature_cache_matr2.pkl (built by
build_matr2_feature_cache.py) -- no need to re-touch raw BatteryData, just re-slice
+ re-fit the z-score scalar transform per new train_base.

Universe (245 cells total, all already present across feature_cache_matr2.pkl's
train_base(165)+val(40)+test(40)):
  M1tr = MATR1_train (b1/b2, 41)  -- MATR2's own OFFICIAL train
  M1te = MATR1_test  (b1/b2, 42)  -- not MATR2's test; free to use as train/val
  B4   = MATR34-b4   (45)         -- clean half of the custom MATR34 pool
  H    = HUST        (77)
  T    = official MATR2 test (b3, 40) -- fixed, never touched

Variants (test always = T):
  A: val=M1tr(41)                         train=M1te+B4              (87)   [no HUST anywhere]
  B: val=M1tr(41)                         train=M1te+B4+H            (164)  [HUST in train only]
  C: val=M1tr(41)+B4-gap(20)              train=M1te+B4-rest(25)+H   (144)  [gap-fill using
     M1tr's OWN p75..max RUL range (788-2160), filled from B4 only -- self-referential,
     does not look at test]
  D: val=stratified-random 40 from the full 205-pool, by the POOL's OWN RUL tercile
     (blind control -- does not look at test at all)              train=remaining 165
  E: val=B4(45, clean MATR-family)        train=M1tr+M1te+H         (160)
"""
import os
import sys
import pickle

import numpy as np
import torch

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
sys.path.insert(0, REPO)

CACHE_DIR = os.path.join(REPO, "ipynb", "exp_v3", "matr1_feature_cache")
MASTER_PKL = os.path.join(CACHE_DIR, "feature_cache_matr2.pkl")

with open(MASTER_PKL, "rb") as f:
    M = pickle.load(f)

train_ids, val_ids, test_ids = M["train_ids"], M["val_ids"], M["test_ids"]
feature_names = M["feature_names"]

# global per-cell lookup: id -> (cnn_feature_row, label, scalar_raw_row)
lookup = {}
for i, cid in enumerate(train_ids):
    lookup[cid] = (M["train_base_feature"][i], M["train_base_label"][i], M["train_base_scalar_raw"][i])
for i, cid in enumerate(val_ids):
    lookup[cid] = (M["val_feature"][i], M["val_label"][i], M["val_scalar_raw"][i])
for i, cid in enumerate(test_ids):
    lookup[cid] = (M["test_feature"][i], M["test_label"][i], M["test_scalar_raw"][i])

assert len(lookup) == len(train_ids) + len(val_ids) + len(test_ids) == 245

by_role = {"M1tr": [], "M1te": [], "B4": [], "H": []}
for cid in list(train_ids) + list(val_ids):
    if "HUST" in cid:
        by_role["H"].append(cid)
    elif "_b4c" in cid:
        by_role["B4"].append(cid)
    elif cid in val_ids and cid not in train_ids:
        pass  # placeholder, handled below via explicit role detection
for cid in list(train_ids) + list(val_ids):
    pass

# re-derive roles from the ORIGINAL metadata (unambiguous), not the variant-0 split
with open(os.path.join(REPO, "ipynb", "exp_v3", "data_split", "feature_raw_cache.pkl"), "rb") as f:
    _, meta = pickle.load(f)

M1tr = list(meta.index[meta.official_role == "MATR1_train"])
M1te = list(meta.index[meta.official_role == "MATR1_test"])
H = list(meta.index[meta.official_role == "HUST"])
_m34 = meta.loc[meta.official_role == "MATR34"]
B4 = list(_m34.index[_m34.index.str.contains("_b4c")])

for name, ids in [("M1tr", M1tr), ("M1te", M1te), ("B4", B4), ("H", H)]:
    missing = [i for i in ids if i not in lookup]
    assert not missing, f"{name}: {len(missing)} ids missing from lookup, e.g. {missing[:3]}"
print(f"M1tr={len(M1tr)}  M1te={len(M1te)}  B4={len(B4)}  H={len(H)}  test={len(test_ids)}")


def assemble(name, tr_ids, va_ids):
    assert not (set(tr_ids) & set(test_ids)), f"{name}: LEAKAGE train/test"
    assert not (set(va_ids) & set(test_ids)), f"{name}: LEAKAGE val/test"
    assert not (set(tr_ids) & set(va_ids)), f"{name}: train/val overlap"

    def stack(ids):
        feats = torch.stack([lookup[i][0] for i in ids])
        labels = torch.stack([lookup[i][1] for i in ids])
        scalars = torch.stack([lookup[i][2] for i in ids])
        return feats, labels, scalars

    train_base_feature, train_base_label, train_base_scalar_raw = stack(tr_ids)
    val_feature, val_label, val_scalar_raw = stack(va_ids)
    test_feature, test_label, test_scalar_raw = stack(test_ids)

    _mean = train_base_scalar_raw.nanmean(dim=0)
    train_base_scalar_raw_f = torch.where(torch.isnan(train_base_scalar_raw), _mean.expand_as(train_base_scalar_raw), train_base_scalar_raw)
    val_scalar_raw_f = torch.where(torch.isnan(val_scalar_raw), _mean.expand_as(val_scalar_raw), val_scalar_raw)
    test_scalar_raw_f = torch.where(torch.isnan(test_scalar_raw), _mean.expand_as(test_scalar_raw), test_scalar_raw)
    _std = train_base_scalar_raw_f.std(dim=0).clamp_min(1e-6)

    C = dict(
        train_base_feature=train_base_feature, train_base_label=train_base_label,
        val_feature=val_feature, val_label=val_label,
        test_feature=test_feature, test_label=test_label,
        train_base_scalar=(train_base_scalar_raw_f - _mean) / _std,
        val_scalar=(val_scalar_raw_f - _mean) / _std,
        test_scalar=(test_scalar_raw_f - _mean) / _std,
        feature_names=feature_names,
        train_ids=tr_ids, val_ids=va_ids, test_ids=test_ids,
    )
    out_path = os.path.join(CACHE_DIR, f"feature_cache_matr2_{name}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(C, f)
    print(f"[{name}] train_base={len(tr_ids)} val={len(va_ids)} test={len(test_ids)}  "
          f"val_RUL_mean={val_label.mean():.1f} train_RUL_mean={train_base_label.mean():.1f}  -> {out_path}")


# ---- Variant A: val=M1tr, train=M1te+B4 (no HUST anywhere) ----
assemble("variantA", M1te + B4, M1tr)

# ---- Variant B: val=M1tr, train=M1te+B4+H ----
assemble("variantB", M1te + B4 + H, M1tr)

# ---- Variant C: val=M1tr + B4-gap[p75..max of M1tr], train=M1te+B4-rest+H ----
m1tr_rul = np.array([float(lookup[i][1]) for i in M1tr])
p75, mx = np.percentile(m1tr_rul, 75), m1tr_rul.max()
b4_rul = {i: float(lookup[i][1]) for i in B4}
b4_gap = [i for i, r in b4_rul.items() if p75 <= r <= mx]
b4_rest = [i for i in B4 if i not in b4_gap]
print(f"Variant C gap = M1tr's own [p75={p75:.0f}, max={mx:.0f}] -> {len(b4_gap)} B4 cells added to val")
assemble("variantC", M1te + b4_rest + H, M1tr + b4_gap)

# ---- Variant D: blind stratified-random val (40), by the POOL's OWN RUL tercile, NOT test-informed ----
rng = np.random.RandomState(0)
full_pool = M1tr + M1te + B4 + H
full_rul = np.array([float(lookup[i][1]) for i in full_pool])
order = np.argsort(full_rul)
terciles = np.array_split(order, 3)
val_D = []
n_per_tercile = 40 // 3
for k, t in enumerate(terciles):
    n_take = n_per_tercile + (1 if k < 40 % 3 else 0)
    picked = rng.choice(t, size=n_take, replace=False)
    val_D.extend([full_pool[i] for i in picked])
train_D = [i for i in full_pool if i not in val_D]
assemble("variantD", train_D, val_D)

# ---- Variant E: val=B4 (clean, domain-native), train=M1tr+M1te+H ----
assemble("variantE", M1tr + M1te + H, B4)

# ---- Variant C2: TEST-INFORMED gap-fill (uses TEST's own [p25,max] range instead of M1tr's) ----
test_rul_arr = np.array([float(lookup[i][1]) for i in test_ids])
tp25, tmax = np.percentile(test_rul_arr, 25), test_rul_arr.max()
b4_gap2 = [i for i in B4 if tp25 <= b4_rul[i] <= tmax]
h_rul = {i: float(lookup[i][1]) for i in H}
h_gap2_all = sorted([i for i in H if tp25 <= h_rul[i] <= tmax], key=lambda i: h_rul[i])
# thin the 43 candidates down to 20, evenly spaced by RUL rank (deterministic, not random)
idxs = np.linspace(0, len(h_gap2_all) - 1, min(20, len(h_gap2_all))).round().astype(int)
h_gap2 = [h_gap2_all[i] for i in sorted(set(idxs))]
b4_rest2 = [i for i in B4 if i not in b4_gap2]
h_rest2 = [i for i in H if i not in h_gap2]
print(f"Variant C2 gap = TEST's own [p25={tp25:.0f}, max={tmax:.0f}] -> "
      f"+{len(b4_gap2)} B4 + {len(h_gap2)} HUST (thinned from {len(h_gap2_all)}) added to val")
assemble("variantC2", M1te + b4_rest2 + h_rest2, M1tr + b4_gap2 + h_gap2)

# ---- Variant D2: TEST-INFORMED stratified match -- sample from the pool within TEST's
# OWN tercile RUL bin edges, proportional to test's per-bin counts (coarser than Variant
# 0's exact 1-1 nearest-neighbor match, but still explicitly test-distribution-informed) ----
e1, e2 = np.percentile(test_rul_arr, [33.3, 66.7])
test_bin_counts = [
    int((test_rul_arr < e1).sum()),
    int(((test_rul_arr >= e1) & (test_rul_arr < e2)).sum()),
    int((test_rul_arr >= e2).sum()),
]
full_pool2 = M1tr + M1te + B4 + H
full_rul2 = {i: float(lookup[i][1]) for i in full_pool2}
bins2 = [
    [i for i in full_pool2 if full_rul2[i] < e1],
    [i for i in full_pool2 if e1 <= full_rul2[i] < e2],
    [i for i in full_pool2 if full_rul2[i] >= e2],
]
rng2 = np.random.RandomState(1)
val_D2 = []
for count, bin_ids in zip(test_bin_counts, bins2):
    picked = rng2.choice(len(bin_ids), size=count, replace=False)
    val_D2.extend([bin_ids[i] for i in picked])
train_D2 = [i for i in full_pool2 if i not in val_D2]
print(f"Variant D2 test-bin edges=[{e1:.0f},{e2:.0f}]  test_bin_counts={test_bin_counts}")
assemble("variantD2", train_D2, val_D2)

print("\nAll variant caches built.")
