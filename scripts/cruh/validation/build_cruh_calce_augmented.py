"""Add the 8 CALCE-like, high-RUL donor cells (search_calce_donors.py) to
the winning CRUH val config's train_base (feature_cache_cruh_cluster_n3.pkl,
val=17/train_base=34), producing feature_cache_cruh_cluster_n3_calcedonors.pkl
(train_base=42). val/test untouched; normalization stats recomputed from the
augmented train_base only.
"""
import os
import pickle

import torch

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
CRUH_DIR = os.path.join(REPO, "ipynb", "exp_v3", "cruh")

with open(os.path.join(CRUH_DIR, "feature_cache_cruh_cluster_n3.pkl"), "rb") as f:
    C = pickle.load(f)
with open(os.path.join(CRUH_DIR, "calce_donors_top8.pkl"), "rb") as f:
    D = pickle.load(f)

assert D["feature_names"] == C["feature_names"]

train_base_feature = torch.cat([C["train_base_feature"], D["donor_feature"]], dim=0)
train_base_label = torch.cat([C["train_base_label"], D["donor_label"]], dim=0)
train_base_scalar_raw = torch.cat([C["train_base_scalar_raw"], D["donor_scalar_raw"]], dim=0)
train_base_ids = list(C["train_ids"]) + list(D["donor_ids"])

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
    "donor_ids": D["donor_ids"], "donor_source": D["donor_source"],
}
out_pkl = os.path.join(CRUH_DIR, "feature_cache_cruh_cluster_n3_calcedonors.pkl")
with open(out_pkl, "wb") as f:
    pickle.dump(out, f)
print(f"train_base={len(train_base_ids)} (34 CRUH + {len(D['donor_ids'])} donors)  "
      f"val={len(C['val_ids'])}  test={len(C['test_ids'])}")
print(f"saved -> {out_pkl}")
