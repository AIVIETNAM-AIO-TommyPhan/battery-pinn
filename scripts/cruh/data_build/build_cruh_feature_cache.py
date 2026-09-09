"""Build feature_cache_cruh_base.pkl for the CRUH benchmark.

CRUH pools 4 raw sources (CALCE, RWTH, UL_PUR, HNEI) via the official
CRUHTrainTestSplitter -- train=51, test=34 (README benchmark: 60 RMSE,
Discharge model). EOL threshold = 80% default (CRUH is not in the
SNL/CRUSH 90% exception list).

Unlike build_hust_feature_cache.py (which produced an immediate naive
train_base/val split), this script keeps the full 51-cell official train
pool intact as `train_pool_*` -- val construction (K-means clustering on
z-scored [non-degenerate features] + z-scored RUL, per the project's
established HUST methodology) happens in a separate downstream script
(build_cruh_cluster_val.py) that consumes this base cache.

None of CRUH's 4 sources have precomputed Qdlin or internal_resistance
(verified 2026-09-01) -- same situation as HUST, where those columns end
up constant/NaN-filled-to-0 after normalization (harmless, matches the
project's "pre-declared, not re-tuned per dataset" TOP3_FEATURES choice).
"""
import os
import sys
import time
import pickle

import numpy as np
import torch
from scipy.stats import skew, kurtosis

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
sys.path.insert(0, REPO)

from batteryml.builders import FEATURE_EXTRACTORS, LABEL_ANNOTATORS
from batteryml.data.battery_data import BatteryData
from batteryml.train_test_split.CRUH_split import CRUHTrainTestSplitter
import BatLiNet  # noqa: F401

DATA_ROOT = os.path.join(REPO, "batteryml", "data", "processed")
CACHE_DIR = os.path.join(REPO, "ipynb", "exp_v3", "cruh")
OUT_PKL = os.path.join(CACHE_DIR, "feature_cache_cruh_base.pkl")

DIFF_BASE = 9
CAP_CLIP_BOUND = 1.05

SCALAR_FEATURE_NAMES = [
    "charge_energy_mean", "discharge_energy_mean", "mean_charge_current", "mean_discharge_current",
    "discharge_time_mean", "mean_discharge_voltage", "voltage_soc_10", "voltage_soc_25",
    "voltage_soc_50", "voltage_soc_75", "voltage_soc_90",
    "voltage_slope_10_50", "voltage_slope_50_90", "voltage_std_discharge", "voltage_hysteresis",
    "coulombic_efficiency_mean", "temperature_mean_charge", "temperature_max_charge", "temperature_rise_charge",
    "temperature_mean_discharge", "temperature_max_discharge", "temperature_rise_discharge",
    "capacity_fade_slope",
]
NEW_FEATURE_NAMES = [
    "internal_resistance_mean", "internal_resistance_change",
    "qdlin_diff_min", "qdlin_diff_max", "qdlin_diff_mean", "qdlin_diff_std",
    "qdlin_diff_var", "qdlin_diff_skew", "qdlin_diff_kurt", "qdlin_diff_l1", "qdlin_diff_l2",
]
ALL_FEATURE_NAMES = SCALAR_FEATURE_NAMES + NEW_FEATURE_NAMES


def _scalar_cycle(cycle_data, nominal_cap):
    I = np.array(cycle_data.current_in_A)
    V = np.array(cycle_data.voltage_in_V)
    t = np.array(cycle_data.time_in_s)
    Qc_raw = np.clip(np.array(cycle_data.charge_capacity_in_Ah), None, CAP_CLIP_BOUND * nominal_cap)
    Qd_raw = np.array(cycle_data.discharge_capacity_in_Ah)
    T = np.array(cycle_data.temperature_in_C) if cycle_data.temperature_in_C else None
    charge_mask, discharge_mask = I > 0.1, I < -0.1
    if charge_mask.sum() < 3 or discharge_mask.sum() < 3:
        return None
    Vc, Ic, tc, Qc = V[charge_mask], I[charge_mask], t[charge_mask], Qc_raw[charge_mask]
    Vd, Id, td, Qd = V[discharge_mask], I[discharge_mask], t[discharge_mask], Qd_raw[discharge_mask]
    charge_energy = np.trapz(Vc * Ic, tc) / 3600.0
    discharge_energy = -np.trapz(Vd * Id, td) / 3600.0
    discharge_time = float(td.max() - td.min()) if len(td) > 1 else np.nan
    mean_discharge_voltage = float(Vd.mean())
    voltage_std_discharge = float(Vd.std())
    q_norm = Qd / nominal_cap
    order = np.argsort(q_norm)
    q_sorted, v_sorted = q_norm[order], Vd[order]

    def v_at_soc(soc):
        if q_sorted.max() < soc or q_sorted.min() > soc:
            return np.nan
        return float(np.interp(soc, q_sorted, v_sorted))

    v10, v25, v50, v75, v90 = [v_at_soc(s) for s in [0.1, 0.25, 0.5, 0.75, 0.9]]
    slope_10_50 = (v50 - v10) / 0.4 if np.isfinite(v10) and np.isfinite(v50) else np.nan
    slope_50_90 = (v90 - v50) / 0.4 if np.isfinite(v50) and np.isfinite(v90) else np.nan
    qc_norm = Qc / nominal_cap
    order_c = np.argsort(qc_norm)
    qc_sorted, vc_sorted = qc_norm[order_c], Vc[order_c]

    def vc_at_soc(soc):
        if qc_sorted.max() < soc or qc_sorted.min() > soc:
            return np.nan
        return float(np.interp(soc, qc_sorted, vc_sorted))

    hyst_val = [vc_at_soc(s) - v_at_soc(s) for s in [0.1, 0.25, 0.5, 0.75, 0.9]]
    hyst_val = [h for h in hyst_val if np.isfinite(h)]
    voltage_hysteresis = np.mean(hyst_val) if hyst_val else np.nan
    ce = float(Qd.max() / Qc.max()) if Qc.max() > 0 else np.nan
    if T is not None:
        Tc, Td = T[charge_mask], T[discharge_mask]
        temp_mean_c, temp_max_c = float(Tc.mean()), float(Tc.max())
        temp_rise_c = float(Tc.max() - Tc[0]) if len(Tc) else np.nan
        temp_mean_d, temp_max_d = float(Td.mean()), float(Td.max())
        temp_rise_d = float(Td.max() - Td[0]) if len(Td) else np.nan
    else:
        temp_mean_c = temp_max_c = temp_rise_c = np.nan
        temp_mean_d = temp_max_d = temp_rise_d = np.nan
    return dict(
        charge_energy=charge_energy, discharge_energy=discharge_energy,
        mean_charge_current=float(Ic.mean()), mean_discharge_current=float(Id.mean()),
        discharge_time=discharge_time, mean_discharge_voltage=mean_discharge_voltage,
        v10=v10, v25=v25, v50=v50, v75=v75, v90=v90,
        slope_10_50=slope_10_50, slope_50_90=slope_50_90,
        voltage_std_discharge=voltage_std_discharge,
        hysteresis=voltage_hysteresis, ce=ce,
        temp_mean_c=temp_mean_c, temp_max_c=temp_max_c, temp_rise_c=temp_rise_c,
        temp_mean_d=temp_mean_d, temp_max_d=temp_max_d, temp_rise_d=temp_rise_d,
        qd_max=float(Qd.max()),
    )


def extract_extended_scalars(cell, max_cycle_index=99):
    nominal_cap = cell.nominal_capacity_in_Ah
    rows = []
    for i, cycle_data in enumerate(cell.cycle_data):
        if i > max_cycle_index:
            break
        r = _scalar_cycle(cycle_data, nominal_cap)
        if r is not None:
            rows.append(r)
    if not rows:
        return np.full(len(SCALAR_FEATURE_NAMES), np.nan, dtype=np.float32)

    def col(key):
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        return float(np.mean(vals)) if len(vals) else np.nan

    qd_series = np.array([r["qd_max"] for r in rows], dtype=np.float64)
    capacity_fade_slope = float(np.polyfit(np.arange(len(qd_series)), qd_series, 1)[0]) if len(qd_series) > 1 else np.nan
    return np.array([
        col("charge_energy"), col("discharge_energy"), col("mean_charge_current"), col("mean_discharge_current"),
        col("discharge_time"), col("mean_discharge_voltage"),
        col('v10'), col('v25'), col('v50'), col('v75'), col('v90'),
        col("slope_10_50"), col("slope_50_90"),
        col("voltage_std_discharge"), col("hysteresis"), col("ce"),
        col("temp_mean_c"), col("temp_max_c"), col("temp_rise_c"),
        col("temp_mean_d"), col("temp_max_d"), col("temp_rise_d"),
        capacity_fade_slope,
    ])


def extract_new_scalars(cell, early_cycle=9, max_cycle_index=99):
    ir_vals = [c.internal_resistance_in_ohm for c in cell.cycle_data[:max_cycle_index + 1]
               if c.internal_resistance_in_ohm is not None]
    ir_mean = float(np.mean(ir_vals)) if len(ir_vals) else np.nan
    max_idx = min(max_cycle_index, len(cell.cycle_data) - 1)
    ir_early = cell.cycle_data[min(early_cycle, max_idx)].internal_resistance_in_ohm
    ir_late = cell.cycle_data[max_idx].internal_resistance_in_ohm

    def _finite_or_none(x):
        return x is not None and np.isfinite(x)

    ir_change = float(ir_late - ir_early) if _finite_or_none(ir_early) and _finite_or_none(ir_late) else np.nan
    qd_early = np.array(cell.cycle_data[min(early_cycle, max_idx)].additional_data.get("Qdlin", []))
    qd_late = np.array(cell.cycle_data[max_idx].additional_data.get("Qdlin", []))
    if len(qd_early) and len(qd_late) and len(qd_early) == len(qd_late):
        diff = qd_late - qd_early
        diff = diff[np.isfinite(diff)]
        if len(diff) > 1:
            vals = dict(
                min=float(np.min(diff)), max=float(np.max(diff)), mean=float(np.mean(diff)),
                std=float(np.std(diff)), var=float(np.var(diff)),
                skew=float(skew(diff, bias=False)), kurt=float(kurtosis(diff, bias=False)),
                l1=float(np.mean(np.abs(diff))), l2=float(np.sqrt(np.mean(diff ** 2))),
            )
        else:
            vals = dict.fromkeys(["min", "max", "mean", "std", "var", "skew", "kurt", "l1", "l2"], np.nan)
    else:
        vals = dict.fromkeys(["min", "max", "mean", "std", "var", "skew", "kurt", "l1", "l2"], np.nan)
    return np.array([
        ir_mean, ir_change, vals["min"], vals["max"], vals["mean"], vals["std"],
        vals["var"], vals["skew"], vals["kurt"], vals["l1"], vals["l2"],
    ], dtype=np.float32)


def full_scalar_vec(cell):
    return np.concatenate([extract_extended_scalars(cell), extract_new_scalars(cell)])


def build_pool(cells, tag):
    t0 = time.time()
    annot = LABEL_ANNOTATORS.build({"name": "RULLabelAnnotator"})
    labels = annot(cells).float()
    assert not torch.isnan(labels).any(), f"{tag}: NaN RUL label found"
    extractor = FEATURE_EXTRACTORS.build({
        "name": "BatLiNetFeatureExtractor", "smooth_features": False,
        "min_cycle_index": 0, "max_cycle_index": 99, "diff_base": DIFF_BASE})
    cnn_feature = extractor(cells).float()
    scalar_raw = torch.tensor(np.stack([full_scalar_vec(c) for c in cells]), dtype=torch.float32)
    print(f"  [{tag}] n={len(cells)}  cnn_feature={tuple(cnn_feature.shape)}  built in {time.time()-t0:.0f}s", flush=True)
    return cnn_feature, labels, scalar_raw


if __name__ == "__main__":
    roots = [
        os.path.join(DATA_ROOT, "CALCE"),
        os.path.join(DATA_ROOT, "RWTH"),
        os.path.join(DATA_ROOT, "UL_PUR"),
        os.path.join(DATA_ROOT, "HNEI"),
    ]
    splitter = CRUHTrainTestSplitter(roots)
    train_files, test_files = splitter.split()
    print(f"train_pool={len(train_files)}  test={len(test_files)}", flush=True)

    train_cell_ids = [f.stem for f in train_files]
    test_cell_ids = [f.stem for f in test_files]
    assert not (set(train_cell_ids) & set(test_cell_ids)), "leakage guard FAIL"

    train_cells = [BatteryData.load(str(f)) for f in train_files]
    test_cells = [BatteryData.load(str(f)) for f in test_files]

    train_pool_feature, train_pool_label, train_pool_scalar_raw = build_pool(train_cells, "train_pool")
    test_feature, test_label, test_scalar_raw = build_pool(test_cells, "test")

    _scalar_mean = train_pool_scalar_raw.nanmean(dim=0)
    _scalar_mean = torch.nan_to_num(_scalar_mean, nan=0.0)
    train_pool_scalar_raw = torch.where(torch.isnan(train_pool_scalar_raw), _scalar_mean.expand_as(train_pool_scalar_raw), train_pool_scalar_raw)
    test_scalar_raw = torch.where(torch.isnan(test_scalar_raw), _scalar_mean.expand_as(test_scalar_raw), test_scalar_raw)
    _scalar_std = train_pool_scalar_raw.std(dim=0).clamp_min(1e-6)

    train_pool_scalar = (train_pool_scalar_raw - _scalar_mean) / _scalar_std
    test_scalar = (test_scalar_raw - _scalar_mean) / _scalar_std

    C = {
        "train_pool_feature": train_pool_feature, "train_pool_label": train_pool_label,
        "test_feature": test_feature, "test_label": test_label,
        "train_pool_scalar": train_pool_scalar, "test_scalar": test_scalar,
        "train_pool_scalar_raw": train_pool_scalar_raw, "test_scalar_raw": test_scalar_raw,
        "feature_names": ALL_FEATURE_NAMES,
        "train_pool_ids": train_cell_ids, "test_ids": test_cell_ids,
    }
    with open(OUT_PKL, "wb") as f:
        pickle.dump(C, f)
    print(f"\nsaved -> {OUT_PKL}")
    print(f"train_pool RUL: mean={train_pool_label.mean():.1f} std={train_pool_label.std():.1f}")
    print(f"test RUL:       mean={test_label.mean():.1f} std={test_label.std():.1f}")
