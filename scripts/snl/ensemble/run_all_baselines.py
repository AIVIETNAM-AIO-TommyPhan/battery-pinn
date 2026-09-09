"""Run all 9 sklearn baseline model families on SNL, using the freshly
re-preprocessed-from-raw data (ipynb/exp_v3/snl/processed_fresh) instead of
the existing batteryml/data/processed/SNL, per explicit request to redo
from raw and check whether that resolves the XGBoost discrepancy
(local repro 545.11 vs the cited README-style number 215).

For each model family's official snl.yaml, this makes a modified copy with
cell_data_path repointed at the fresh directory, then runs Pipeline.train +
evaluate exactly as bin/batteryml.py's `run --train --eval` would.
"""
import os
import sys
import re
import glob
import pickle

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
sys.path.insert(0, REPO)

from batteryml.pipeline import Pipeline

FRESH_SNL_DIR = os.path.join(REPO, "ipynb", "exp_v3", "snl", "processed_fresh")
TMP_CONFIG_DIR = os.path.join(REPO, "ipynb", "exp_v3", "snl", "tmp_configs")
WORKSPACE_ROOT = os.path.join(REPO, "ipynb", "exp_v3", "snl", "baseline_workspaces")
os.makedirs(TMP_CONFIG_DIR, exist_ok=True)
os.makedirs(WORKSPACE_ROOT, exist_ok=True)

MODELS = [
    ("Dummy", "configs/baselines/sklearn/dummy/snl.yaml"),
    ("Variance", "configs/baselines/sklearn/variance_model/snl.yaml"),
    ("Discharge", "configs/baselines/sklearn/discharge_model/snl.yaml"),
    ("Full", "configs/baselines/sklearn/full_model/snl.yaml"),
    ("Ridge", "configs/baselines/sklearn/ridge/snl.yaml"),
    ("PCR", "configs/baselines/sklearn/pcr/snl.yaml"),
    ("PLSR", "configs/baselines/sklearn/plsr/snl.yaml"),
    ("GPR", "configs/baselines/sklearn/gpr/snl.yaml"),
    ("XGBoost", "configs/baselines/sklearn/xgb/snl.yaml"),
]

CELL_PATH_RE = re.compile(r"cell_data_path:\s*'.*?'")

results = {}
for name, rel_path in MODELS:
    config_path = os.path.join(REPO, rel_path)
    with open(config_path, "r") as f:
        text = f.read()
    fresh_path_escaped = FRESH_SNL_DIR.replace("\\", "\\\\")
    new_text = CELL_PATH_RE.sub(f"cell_data_path: '{fresh_path_escaped}'", text, count=1)
    tmp_config = os.path.join(TMP_CONFIG_DIR, f"snl_{name.lower()}.yaml")
    with open(tmp_config, "w") as f:
        f.write(new_text)

    workspace = os.path.join(WORKSPACE_ROOT, name.lower())
    print(f"\n=== {name} ===", flush=True)
    try:
        pipeline = Pipeline(tmp_config, workspace)
        model, dataset = pipeline.train(seed=0, device="cpu", dataset=None, skip_if_executed=False)
        pipeline.evaluate(seed=0, device="cpu", metric=["RMSE", "MAE", "MAPE"],
                           model=model, dataset=dataset, skip_if_executed=False)
        pred_files = sorted(glob.glob(os.path.join(workspace, "predictions_seed_0_*.pkl")))
        with open(pred_files[-1], "rb") as f:
            obj = pickle.load(f)
        results[name] = obj["scores"]
        print(f"{name}: {obj['scores']}", flush=True)
    except Exception as e:
        results[name] = f"ERROR: {e}"
        print(f"{name}: ERROR: {e}", flush=True)

print("\n\n=== SUMMARY (fresh-from-raw SNL data) ===")
for name, _ in MODELS:
    print(f"{name:12s}  {results.get(name)}")
