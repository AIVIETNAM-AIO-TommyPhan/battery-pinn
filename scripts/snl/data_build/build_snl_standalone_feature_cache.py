"""Build feature_cache_snl_base.pkl using SNL's OWN official split
(SNLTrainTestSplitter: train=36, test=25) -- README benchmark: 200 RMSE.
EOL=0.9 (project convention). Isolates SNL from the CRUSH mixed pool to
check whether SNL's catastrophic CRUSH performance (RMSE=831.92, seed 0,
n=4 val config) is intrinsic to SNL itself or an artifact of dilution/
starvation within the 5-source CRUSH pool.

Censored cells (RUL<=100 after RULLabelAnnotator's min_rul_limit filter --
see build_crush_feature_cache.py's docstring for why this is NOT true
right-censoring) are flagged, matching the CRUSH build's treatment.
"""
import os
import sys
import time
import pickle
import importlib.util

import numpy as np
import torch

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
sys.path.insert(0, REPO)

from batteryml.builders import FEATURE_EXTRACTORS, LABEL_ANNOTATORS
from batteryml.data.battery_data import BatteryData
from batteryml.train_test_split.SNL_split import SNLTrainTestSplitter
import BatLiNet  # noqa: F401

spec = importlib.util.spec_from_file_location(
    "build_cruh_feature_cache",
    os.path.join(REPO, "ipynb", "exp_v3", "cruh", "build_cruh_feature_cache.py"))
cruh_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cruh_mod)

DATA_ROOT = os.path.join(REPO, "batteryml", "data", "processed", "SNL")
CACHE_DIR = os.path.join(REPO, "ipynb", "exp_v3", "snl")
OUT_PKL = os.path.join(CACHE_DIR, "feature_cache_snl_base.pkl")
EOL_SOH = 0.9
DIFF_BASE = 9


def build_pool_eol90(cells, tag):
    t0 = time.time()
    annot = LABEL_ANNOTATORS.build({"name": "RULLabelAnnotator", "eol_soh": EOL_SOH})
    labels = annot(cells).float()
    n_censored = int(torch.isnan(labels).sum())
    extractor = FEATURE_EXTRACTORS.build({
        "name": "BatLiNetFeatureExtractor", "smooth_features": False,
        "min_cycle_index": 0, "max_cycle_index": 99, "diff_base": DIFF_BASE})
    cnn_feature = extractor(cells).float()
    scalar_raw = torch.tensor(np.stack([cruh_mod.full_scalar_vec(c) for c in cells]), dtype=torch.float32)
    print(f"  [{tag}] n={len(cells)}  censored(RUL<=100)={n_censored}  "
          f"cnn_feature={tuple(cnn_feature.shape)}  built in {time.time()-t0:.0f}s", flush=True)
    return cnn_feature, labels, scalar_raw


if __name__ == "__main__":
    splitter = SNLTrainTestSplitter(DATA_ROOT)
    train_files, test_files = splitter.split()
    print(f"train_pool={len(train_files)}  test={len(test_files)}", flush=True)

    train_cell_ids = [f.stem for f in train_files]
    test_cell_ids = [f.stem for f in test_files]
    assert not (set(train_cell_ids) & set(test_cell_ids)), "leakage guard FAIL"

    train_cells = [BatteryData.load(str(f)) for f in train_files]
    test_cells = [BatteryData.load(str(f)) for f in test_files]

    train_pool_feature, train_pool_label, train_pool_scalar_raw = build_pool_eol90(train_cells, "train_pool")
    test_feature, test_label, test_scalar_raw = build_pool_eol90(test_cells, "test")

    censored_train = [cid for cid, lab in zip(train_cell_ids, train_pool_label) if torch.isnan(lab)]
    censored_test = [cid for cid, lab in zip(test_cell_ids, test_label) if torch.isnan(lab)]
    print(f"censored train: {len(censored_train)}  censored test: {len(censored_test)}")

    _scalar_mean = torch.nan_to_num(train_pool_scalar_raw.nanmean(dim=0), nan=0.0)
    train_pool_scalar_raw_f = torch.where(torch.isnan(train_pool_scalar_raw), _scalar_mean.expand_as(train_pool_scalar_raw), train_pool_scalar_raw)
    test_scalar_raw_f = torch.where(torch.isnan(test_scalar_raw), _scalar_mean.expand_as(test_scalar_raw), test_scalar_raw)
    _scalar_std = train_pool_scalar_raw_f.std(dim=0).clamp_min(1e-6)

    C = {
        "train_pool_feature": train_pool_feature, "train_pool_label": train_pool_label,
        "test_feature": test_feature, "test_label": test_label,
        "train_pool_scalar": (train_pool_scalar_raw_f - _scalar_mean) / _scalar_std,
        "test_scalar": (test_scalar_raw_f - _scalar_mean) / _scalar_std,
        "train_pool_scalar_raw": train_pool_scalar_raw_f, "test_scalar_raw": test_scalar_raw_f,
        "feature_names": cruh_mod.ALL_FEATURE_NAMES,
        "train_pool_ids": train_cell_ids, "test_ids": test_cell_ids,
        "censored_train_ids": censored_train, "censored_test_ids": censored_test,
        "eol_soh": EOL_SOH,
    }
    with open(OUT_PKL, "wb") as f:
        pickle.dump(C, f)
    print(f"\nsaved -> {OUT_PKL}")
    valid_train = train_pool_label[~torch.isnan(train_pool_label)]
    valid_test = test_label[~torch.isnan(test_label)]
    print(f"train_pool RUL (valid): n={len(valid_train)} mean={valid_train.mean():.1f} std={valid_train.std():.1f} range=[{valid_train.min():.0f},{valid_train.max():.0f}]")
    print(f"test RUL (valid):       n={len(valid_test)} mean={valid_test.mean():.1f} std={valid_test.std():.1f} range=[{valid_test.min():.0f},{valid_test.max():.0f}]")
