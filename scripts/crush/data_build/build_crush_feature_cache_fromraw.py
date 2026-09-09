"""Build feature_cache_crush_base_fromraw.pkl -- same as
build_crush_feature_cache.py but sourcing SNL/UL_PUR/HNEI from freshly
re-preprocessed-from-raw folders (verified byte-identical to the existing
batteryml/data/processed/* for these 3 sources) instead of the existing
processed folders directly. CALCE/RWTH still use the existing processed
folders (no raw zip available locally for either). This is a concrete
end-to-end confirmation that our own CNN pipeline's CRUSH result doesn't
depend on any drift/bug in the pre-existing processed cache.

Build feature_cache_crush_base.pkl for the CRUSH benchmark.

CRUSH pools 5 raw sources (CALCE, RWTH, UL_PUR, SNL, HNEI) via the official
CRUSHTrainTestSplitter -- README benchmark: 330 RMSE (Discharge model).
EOL threshold = 90% (SNL/CRUSH project exception, NOT the 80% default).

Reuses the exact scalar/CNN feature-extraction code already validated on
CRUH/HUST (imported from build_cruh_feature_cache.py, not duplicated), but
with its own build_pool() that uses eol_soh=0.9 (the CRUH importer's
build_pool hardcodes the 0.8 default, which is wrong for CRUSH).

Censored cells (RUL never crosses 90% within observed cycles) are flagged
explicitly here -- not silently dropped -- consistent with the project's
censored-data convention. Any censored cell in the OFFICIAL test set is a
real problem (can't compute RMSE against a NaN label) and is reported, not
hidden.
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
from batteryml.train_test_split.CRUSH_split import CRUSHTrainTestSplitter
import BatLiNet  # noqa: F401

spec = importlib.util.spec_from_file_location(
    "build_cruh_feature_cache",
    os.path.join(REPO, "ipynb", "exp_v3", "cruh", "build_cruh_feature_cache.py"))
cruh_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cruh_mod)

DATA_ROOT = os.path.join(REPO, "batteryml", "data", "processed")
CACHE_DIR = os.path.join(REPO, "ipynb", "exp_v3", "crush")
OUT_PKL = os.path.join(CACHE_DIR, "feature_cache_crush_base_fromraw.pkl")
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
    print(f"  [{tag}] n={len(cells)}  censored(NaN RUL @ eol={EOL_SOH})={n_censored}  "
          f"cnn_feature={tuple(cnn_feature.shape)}  built in {time.time()-t0:.0f}s", flush=True)
    return cnn_feature, labels, scalar_raw


if __name__ == "__main__":
    roots = [
        os.path.join(DATA_ROOT, "CALCE"),  # no raw available locally
        os.path.join(DATA_ROOT, "RWTH"),   # no raw available locally
        os.path.join(REPO, "ipynb", "exp_v3", "crush", "raw_extract", "UL_PUR", "UL_PUR"),  # md5-verified identical to processed
        os.path.join(REPO, "ipynb", "exp_v3", "snl", "processed_fresh"),  # rebuilt from raw SNL.zip
        os.path.join(REPO, "ipynb", "exp_v3", "crush", "raw_extract", "HNEI", "HNEI"),  # md5-verified identical to processed
    ]
    splitter = CRUSHTrainTestSplitter(roots)
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
    print(f"\nCENSORED cells (RUL never crosses {EOL_SOH*100:.0f}% within observed window):")
    print(f"  train: {len(censored_train)}  {censored_train}")
    print(f"  test:  {len(censored_test)}  {censored_test}")
    if censored_test:
        print("  WARNING: censored cells present in the OFFICIAL test set -- "
              "these cannot be scored by standard RMSE and must be handled explicitly downstream.")

    _scalar_mean = train_pool_scalar_raw.nanmean(dim=0)
    _scalar_mean = torch.nan_to_num(_scalar_mean, nan=0.0)
    train_pool_scalar_raw_f = torch.where(torch.isnan(train_pool_scalar_raw), _scalar_mean.expand_as(train_pool_scalar_raw), train_pool_scalar_raw)
    test_scalar_raw_f = torch.where(torch.isnan(test_scalar_raw), _scalar_mean.expand_as(test_scalar_raw), test_scalar_raw)
    _scalar_std = train_pool_scalar_raw_f.std(dim=0).clamp_min(1e-6)

    train_pool_scalar = (train_pool_scalar_raw_f - _scalar_mean) / _scalar_std
    test_scalar = (test_scalar_raw_f - _scalar_mean) / _scalar_std

    C = {
        "train_pool_feature": train_pool_feature, "train_pool_label": train_pool_label,
        "test_feature": test_feature, "test_label": test_label,
        "train_pool_scalar": train_pool_scalar, "test_scalar": test_scalar,
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
    print(f"train_pool RUL (valid only): mean={valid_train.mean():.1f} std={valid_train.std():.1f}")
    print(f"test RUL (valid only):       mean={valid_test.mean():.1f} std={valid_test.std():.1f}")
