"""Run the official BatteryML XGBoost baseline config directly via Pipeline
(bypassing bin/batteryml.py, which shadows the `batteryml` package name when
invoked as a script from within its own bin/ directory)."""
import sys

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
sys.path.insert(0, REPO)

from batteryml.pipeline import Pipeline

config_path = sys.argv[1]
workspace = sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0

pipeline = Pipeline(config_path, workspace)
model, dataset = pipeline.train(seed=seed, device="cpu", dataset=None, skip_if_executed=False)
pipeline.evaluate(seed=seed, device="cpu", metric=["RMSE", "MAE", "MAPE"], model=model, dataset=dataset, skip_if_executed=False)
