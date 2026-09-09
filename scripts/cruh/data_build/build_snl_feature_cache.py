"""Build feature_cache_snl.pkl: scalar + CNN features for all 61 SNL cells
(no train/test split needed -- this is a donor pool for cross-dataset
augmentation search, not an SNL benchmark run).

eol_soh=0.9 for SNL, per project convention (CLAUDE.md: SNL/CRUSH use 0.9,
not the 0.8 default) -- NaN (never crosses 90% within observed cycles) cells
are excluded as right-censored, same treatment as Tongji.

Reuses the CRUH build script's extraction code directly (same 34-feature
schema as MATR2/HUST/CRUH/Tongji).
"""
import os
import sys
import glob
import pickle
import importlib.util

import numpy as np
import torch

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
sys.path.insert(0, REPO)

from batteryml.data.battery_data import BatteryData
from batteryml.builders import LABEL_ANNOTATORS
import BatLiNet  # noqa: F401

spec = importlib.util.spec_from_file_location(
    "build_cruh_feature_cache",
    os.path.join(REPO, "ipynb", "exp_v3", "cruh", "build_cruh_feature_cache.py"))
cruh_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cruh_mod)

DATA_DIR = os.path.join(REPO, "batteryml", "data", "processed", "SNL")
OUT_PKL = os.path.join(REPO, "ipynb", "exp_v3", "cruh", "feature_cache_snl.pkl")

if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pkl")))
    annot = LABEL_ANNOTATORS.build({"name": "RULLabelAnnotator", "eol_soh": 0.9})

    cells, cell_ids, ruls = [], [], []
    for fp in files:
        c = BatteryData.load(fp)
        r = annot([c]).item()
        if np.isnan(r):
            continue
        cells.append(c)
        cell_ids.append(c.cell_id)
        ruls.append(float(r))

    print(f"valid-RUL SNL cells: {len(cells)}/{len(files)}", flush=True)

    # build_pool() internally recomputes RUL at the default eol_soh=0.8 --
    # discard its label output and use the eol_soh=0.9 labels computed above.
    feature, _, scalar_raw = cruh_mod.build_pool(cells, "snl")
    label = torch.tensor(ruls, dtype=torch.float32)

    C = {
        "feature": feature, "label": label, "scalar_raw": scalar_raw,
        "feature_names": cruh_mod.ALL_FEATURE_NAMES, "cell_ids": cell_ids,
    }
    with open(OUT_PKL, "wb") as f:
        pickle.dump(C, f)
    print(f"saved -> {OUT_PKL}")
    print(f"RUL: mean={label.mean():.1f} std={label.std():.1f} range=[{label.min():.0f},{label.max():.0f}]")
