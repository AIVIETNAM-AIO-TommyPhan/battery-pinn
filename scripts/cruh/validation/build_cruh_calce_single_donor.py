"""Build one feature_cache_cruh_cluster_n3_donor{i}.pkl per single donor
(from calce_donors_top8.pkl), adding exactly one donor cell to the winning
n=3 val config's train_base (34 -> 35), instead of all 8 at once -- isolates
which donor actually helps CALCE vs which one hurts.
"""
import os
import pickle
import sys

import torch

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
CRUH_DIR = os.path.join(REPO, "ipynb", "exp_v3", "cruh")

with open(os.path.join(CRUH_DIR, "feature_cache_cruh_cluster_n3.pkl"), "rb") as f:
    C = pickle.load(f)
with open(os.path.join(CRUH_DIR, "calce_donors_top8.pkl"), "rb") as f:
    D = pickle.load(f)

donor_idx = int(sys.argv[1]) if len(sys.argv) > 1 else None
indices = [donor_idx] if donor_idx is not None else list(range(len(D["donor_ids"])))

for i in indices:
    donor_feature = D["donor_feature"][i:i+1]
    donor_label = D["donor_label"][i:i+1]
    donor_scalar_raw = D["donor_scalar_raw"][i:i+1]
    donor_id = D["donor_ids"][i]

    train_base_feature = torch.cat([C["train_base_feature"], donor_feature], dim=0)
    train_base_label = torch.cat([C["train_base_label"], donor_label], dim=0)
    train_base_scalar_raw = torch.cat([C["train_base_scalar_raw"], donor_scalar_raw], dim=0)
    train_base_ids = list(C["train_ids"]) + [donor_id]

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
        "donor_id": donor_id,
    }
    out_pkl = os.path.join(CRUH_DIR, f"feature_cache_cruh_cluster_n3_donor{i}.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(out, f)
    print(f"donor#{i} = {donor_id}  train_base=35 (34+1)  -> {out_pkl}")
