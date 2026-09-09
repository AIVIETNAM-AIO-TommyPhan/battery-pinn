"""Run all 9 sklearn baseline model families on CRUSH, using the existing
batteryml/data/processed/{CALCE,RWTH,UL_PUR,SNL,HNEI} folders directly (no
raw-reprocessing needed: SNL was already confirmed byte-identical when
rebuilt from raw for the SNL investigation, and UL_PUR/HNEI's "raw" zips
turned out to already be the same processed .pkl files -- verified via
md5sum match. CALCE/RWTH have no raw zip available locally at all.)
"""
import os
import sys
import glob
import pickle

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
sys.path.insert(0, REPO)

from batteryml.pipeline import Pipeline

WORKSPACE_ROOT = os.path.join(REPO, "ipynb", "exp_v3", "crush", "baseline_workspaces")
os.makedirs(WORKSPACE_ROOT, exist_ok=True)

MODELS = [
    ("Dummy", "configs/baselines/sklearn/dummy/crush.yaml"),
    ("Variance", "configs/baselines/sklearn/variance_model/crush.yaml"),
    ("Discharge", "configs/baselines/sklearn/discharge_model/crush.yaml"),
    ("Full", "configs/baselines/sklearn/full_model/crush.yaml"),
    ("Ridge", "configs/baselines/sklearn/ridge/crush.yaml"),
    ("PCR", "configs/baselines/sklearn/pcr/crush.yaml"),
    ("PLSR", "configs/baselines/sklearn/plsr/crush.yaml"),
    ("GPR", "configs/baselines/sklearn/gpr/crush.yaml"),
    ("XGBoost", "configs/baselines/sklearn/xgb/crush.yaml"),
]

results = {}
for name, rel_path in MODELS:
    config_path = os.path.join(REPO, rel_path)
    workspace = os.path.join(WORKSPACE_ROOT, name.lower())
    print(f"\n=== {name} ===", flush=True)
    try:
        pipeline = Pipeline(config_path, workspace)
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

print("\n\n=== SUMMARY (CRUSH, existing processed data) ===")
for name, _ in MODELS:
    print(f"{name:12s}  {results.get(name)}")
