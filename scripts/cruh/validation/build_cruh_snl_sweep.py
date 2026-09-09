"""Sweep the number of SNL donor cells added to CRUH's n=3 val config's
train_base, ranked by feature-distance to CALCE (RUL constraint relaxed
progressively to admit more candidates, same style as the HUST MATR-count
sweep n=15/20/30/45).

Order (by feature distance, RUL relaxation noted):
  N=4 (RUL>=550): NCA_b(625), NCA_d(598), LFP_a(872), LFP_d(683)      [already tested]
  N=6 (RUL>=500): + NCA_a(423 - below 500 but closest overall, included
                    for monotonic distance order), NCA_c(504)
  N=7 (RUL>=500): + LFP_c(521)
  N=8 (relaxed further): + LFP_b(323 - deliberately low RUL, tests where
                    dilution starts, mirroring HUST's n=45 result)
"""
import os
import pickle

import torch

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
CRUH_DIR = os.path.join(REPO, "ipynb", "exp_v3", "cruh")

with open(os.path.join(CRUH_DIR, "feature_cache_cruh_cluster_n3.pkl"), "rb") as f:
    C = pickle.load(f)
with open(os.path.join(CRUH_DIR, "feature_cache_snl.pkl"), "rb") as f:
    SNL = pickle.load(f)

# distance-ranked order (from the full-31 SNL ranking already computed)
ORDER = [
    "SNL_18650_NCA_25C_20-80_0.5-0.5C_b",  # dist 21.12, RUL 625
    "SNL_18650_NCA_25C_20-80_0.5-0.5C_d",  # dist 21.36, RUL 598
    "SNL_18650_LFP_35C_0-100_0.5-1C_a",    # dist 34.92, RUL 872
    "SNL_18650_LFP_35C_0-100_0.5-1C_d",    # dist 35.19, RUL 683
    "SNL_18650_NCA_25C_20-80_0.5-0.5C_a",  # dist 20.60, RUL 423
    "SNL_18650_NCA_25C_20-80_0.5-0.5C_c",  # dist 21.60, RUL 504
    "SNL_18650_LFP_35C_0-100_0.5-1C_c",    # dist 34.71, RUL 521
    "SNL_18650_LFP_35C_0-100_0.5-1C_b",    # dist 35.28, RUL 323 (deliberately low-RUL)
]
COUNTS = [4, 6, 7, 8]

snl_id_to_idx = {cid: i for i, cid in enumerate(SNL["cell_ids"])}

for n in COUNTS:
    donor_ids = ORDER[:n]
    idxs = [snl_id_to_idx[c] for c in donor_ids]
    donor_feature = SNL["feature"][idxs]
    donor_label = SNL["label"][idxs]
    donor_scalar_raw = SNL["scalar_raw"][idxs]

    train_base_feature = torch.cat([C["train_base_feature"], donor_feature], dim=0)
    train_base_label = torch.cat([C["train_base_label"], donor_label], dim=0)
    train_base_scalar_raw = torch.cat([C["train_base_scalar_raw"], donor_scalar_raw], dim=0)
    train_base_ids = list(C["train_ids"]) + donor_ids

    _mean = torch.nan_to_num(train_base_scalar_raw.nanmean(dim=0), nan=0.0)
    train_base_f = torch.where(torch.isnan(train_base_scalar_raw), _mean.expand_as(train_base_scalar_raw), train_base_scalar_raw)
    val_f = torch.where(torch.isnan(C["val_scalar_raw"]), _mean.expand_as(C["val_scalar_raw"]), C["val_scalar_raw"])
    test_f = torch.where(torch.isnan(C["test_scalar_raw"]), _mean.expand_as(C["test_scalar_raw"]), C["test_scalar_raw"])
    _std = train_base_f.std(dim=0).clamp_min(1e-6)

    out = {
        "train_base_feature": train_base_feature, "train_base_label": train_base_label,
        "val_feature": C["val_feature"], "val_label": C["val_label"],
        "test_feature": C["test_feature"], "test_label": C["test_label"],
        "train_base_scalar": (train_base_f - _mean) / _std,
        "val_scalar": (val_f - _mean) / _std,
        "test_scalar": (test_f - _mean) / _std,
        "train_base_scalar_raw": train_base_f, "val_scalar_raw": val_f, "test_scalar_raw": test_f,
        "feature_names": C["feature_names"],
        "train_ids": train_base_ids, "val_ids": C["val_ids"], "test_ids": C["test_ids"],
        "donor_ids": donor_ids,
    }
    out_pkl = os.path.join(CRUH_DIR, f"feature_cache_cruh_cluster_n3_snl{n}.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(out, f)
    print(f"N={n}  train_base={len(train_base_ids)} (34+{n})  donors={donor_ids}  -> {out_pkl}")
