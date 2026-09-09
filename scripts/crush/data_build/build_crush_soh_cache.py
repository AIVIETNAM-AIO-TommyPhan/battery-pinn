"""Build a SOH trajectory cache (cycles 0-99, SOH=Qd_max/nominal_capacity)
for the current best CRUSH config's train_base/val/test cells, matching the
format of ipynb/exp_v3/matr1_feature_cache/soh_trajectory_cache.pkl (shape
n_cells x 100) -- needed for the PINN Tier 1 auxiliary SOH head (Wiener_PINN
checklist, ported here per explicit request to prove PINN can help on CRUSH,
where there's much more headroom than the already-saturated MATR1 result).

Cells with fewer than 100 cycles are padded by repeating the last available
SOH value (cells that fail before cycle 100 stay at their final measured SOH
for the remaining slots -- a reasonable proxy, not a physical claim).
"""
import os
import sys
import pickle

import numpy as np
import torch

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
sys.path.insert(0, REPO)

from batteryml.data.battery_data import BatteryData

CACHE_DIR = os.path.join(REPO, "ipynb", "exp_v3", "crush")
DATA_ROOT = os.path.join(REPO, "batteryml", "data", "processed")
N_CYCLES = 100

SOURCE_DIRS = ["CALCE", "RWTH", "UL_PUR", "SNL", "HNEI", "HUST", "TONGJI"]


def find_cell_path(cid):
    for src in SOURCE_DIRS:
        p = os.path.join(DATA_ROOT, src, cid + ".pkl")
        if os.path.exists(p):
            return p
        # HUST cells are stored as HUST_<id>.pkl but referenced here by bare <id>
        p2 = os.path.join(DATA_ROOT, src, f"{src}_{cid}.pkl")
        if os.path.exists(p2):
            return p2
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
    with open(os.path.join(CACHE_DIR, "feature_cache_crush_hust10_calce15_fullval.pkl"), "rb") as f:
        C = pickle.load(f)

    for split in ["train", "val", "test"]:
        ids = C[f"{split}_ids"]
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
        C[f"{split}_soh"] = torch.tensor(np.stack(trajs), dtype=torch.float32)
        print(f"{split}_soh: {C[f'{split}_soh'].shape}", flush=True)

    out_pkl = os.path.join(CACHE_DIR, "feature_cache_crush_hust10_calce15_fullval_withsoh.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(C, f)
    print(f"saved -> {out_pkl}")
