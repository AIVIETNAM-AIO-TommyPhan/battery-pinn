"""Build a SOH trajectory cache (cycles 0-99, SOH=Qd_max/nominal_capacity)
for HUST's winning config (feature_cache_hust_n2_matr20.pkl: train_base=63
[43 HUST + 20 MATR], val=12, test=22) -- for the PINN Tier1 auxiliary SOH
head, same as build_crush_soh_cache.py.
"""
import os
import sys
import pickle

import numpy as np
import torch

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
sys.path.insert(0, REPO)

from batteryml.data.battery_data import BatteryData

CACHE_DIR = os.path.join(REPO, "ipynb", "exp_v3", "hust")
DATA_ROOT = os.path.join(REPO, "batteryml", "data", "processed")
N_CYCLES = 100
SOURCE_DIRS = ["HUST", "MATR", "CALCE", "RWTH", "UL_PUR", "SNL", "HNEI", "TONGJI"]


def find_cell_path(cid):
    for src in SOURCE_DIRS:
        for candidate in (cid, f"{src}_{cid}"):
            p = os.path.join(DATA_ROOT, src, candidate + ".pkl")
            if os.path.exists(p):
                return p
        # strip a leading batch tag like "MATR1_" / "MATR34_" before the real filename
        if "_" in cid:
            stripped = cid.split("_", 1)[1]
            p = os.path.join(DATA_ROOT, src, stripped + ".pkl")
            if os.path.exists(p):
                return p
    return None


def cell_soh_trajectory(cell):
    nominal_cap = cell.nominal_capacity_in_Ah
    soh = []
    for i, cyc in enumerate(cell.cycle_data):
        if i >= N_CYCLES:
            break
        qd = cyc.discharge_capacity_in_Ah
        qd_max = float(np.max(qd)) if len(qd) else np.nan
        soh.append(qd_max / nominal_cap)
    if not soh:
        return np.full(N_CYCLES, 1.0, dtype=np.float32)
    soh = np.array(soh, dtype=np.float32)
    soh = np.nan_to_num(soh, nan=(soh[np.isfinite(soh)].mean() if np.isfinite(soh).any() else 1.0))
    if len(soh) < N_CYCLES:
        pad = np.full(N_CYCLES - len(soh), soh[-1], dtype=np.float32)
        soh = np.concatenate([soh, pad])
    return soh


if __name__ == "__main__":
    with open(os.path.join(CACHE_DIR, "feature_cache_hust_n2_matr20.pkl"), "rb") as f:
        C = pickle.load(f)

    for split, key in [("train_base", "train"), ("val", "val"), ("test", "test")]:
        ids = C[f"{key}_ids"]
        trajs = []
        missing = []
        for cid in ids:
            path = find_cell_path(cid)
            if path is None:
                missing.append(cid)
                trajs.append(np.full(N_CYCLES, 1.0, dtype=np.float32))
                continue
            cell = BatteryData.load(path)
            trajs.append(cell_soh_trajectory(cell))
        if missing:
            print(f"WARNING [{split}]: {len(missing)} cells not found: {missing}", flush=True)
        C[f"{key}_soh"] = torch.tensor(np.stack(trajs), dtype=torch.float32)
        print(f"{key}_soh: {C[f'{key}_soh'].shape}", flush=True)

    out_pkl = os.path.join(CACHE_DIR, "feature_cache_hust_n2_matr20_withsoh.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(C, f)
    print(f"saved -> {out_pkl}")
