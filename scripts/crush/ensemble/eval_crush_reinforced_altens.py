"""Post-hoc alternate ensemble methods on the already-saved CRUSH
(train=70 reinforced, val=20 random) Tier3 per-component predictions --
no retraining needed, since v2/v1/inter raw val+test predictions are
already saved per seed. Mirrors run_crush_v1v2inter_tier3_ensemble.py's own
affine_fit preprocessing exactly, then compares:
  - NNLS (already computed, saved as ens_test)
  - simple mean of the 3 affine-calibrated components
  - simple mean of just {V2, Inter} (drop V1, the consistently weakest arm)
"""
import glob
import os
import pickle
import re

import numpy as np

CACHE_DIR = os.path.join(
    r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML",
    "ipynb", "exp_v3", "crush",
)


def rmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)))


def affine_fit(val_p, val_t, test_p):
    A = np.vstack([val_p, np.ones_like(val_p)]).T
    a, b = np.linalg.lstsq(A, val_t, rcond=None)[0]
    return a * val_p + b, a * test_p + b


pattern = os.path.join(
    CACHE_DIR,
    "feature_cache_crush_hust10_calce15_fullval_withsoh_randval_reinforced_v1v2inter_tier3_pen1.4_ensemble_seed*.pkl",
)
files = sorted(glob.glob(pattern), key=lambda p: int(re.search(r"seed(\d+)", p).group(1)))

rows = []
for fp in files:
    seed = int(re.search(r"seed(\d+)", fp).group(1))
    with open(fp, "rb") as f:
        D = pickle.load(f)
    val_true, test_true = D["val_true"], D["test_true"]
    v2c_val, v2c_test = affine_fit(D["v2_val"], val_true, D["v2_test"])
    v1c_val, v1c_test = affine_fit(D["v1_val"], val_true, D["v1_test"])
    ic_val, ic_test = affine_fit(D["inter_val"], val_true, D["inter_test"])

    mean3_val = (v2c_val + v1c_val + ic_val) / 3
    mean3_test = (v2c_test + v1c_test + ic_test) / 3
    mean2_val = (v2c_val + ic_val) / 2
    mean2_test = (v2c_test + ic_test) / 2

    rows.append(dict(
        seed=seed,
        v2=rmse(D["v2_test"], test_true),
        v1=rmse(D["v1_test"], test_true),
        inter=rmse(D["inter_test"], test_true),
        nnls=rmse(D["ens_test"], test_true),
        nnls_val=rmse(D["ens_val"], val_true),
        mean3=rmse(mean3_test, test_true),
        mean3_val=rmse(mean3_val, val_true),
        mean2_v2inter=rmse(mean2_test, test_true),
        mean2_val=rmse(mean2_val, val_true),
        nnls_w=D["nnls_weights"].round(3).tolist(),
    ))

print(f"{'seed':>4} {'V2':>7} {'V1':>7} {'Inter':>7} {'NNLS':>7}(val={'':>0}) {'Mean3':>7} {'Mean2(V2+Inter)':>16}")
for r in rows:
    print(f"{r['seed']:>4} {r['v2']:>7.2f} {r['v1']:>7.2f} {r['inter']:>7.2f} "
          f"{r['nnls']:>7.2f}(v={r['nnls_val']:.1f}) {r['mean3']:>7.2f}(v={r['mean3_val']:.1f}) "
          f"{r['mean2_v2inter']:>7.2f}(v={r['mean2_val']:.1f})  w={r['nnls_w']}")

if len(rows) > 1:
    import statistics as st
    for key in ["v2", "v1", "inter", "nnls", "mean3", "mean2_v2inter"]:
        vals = [r[key] for r in rows]
        print(f"{key}: mean={st.mean(vals):.2f} std={st.pstdev(vals):.2f} (n={len(vals)})")
