"""Stage 0 sanity baseline -- SmallCNNScalarBranch, day_0901 architecture-ladder track.

Fresh seed=0 verification run (patience=350, matching the exact protocol that produced
the historical 95.42 in ipynb/exp_v2/matr/matr1/variant_c_tier0_safety_loss_v1.ipynb,
confirmed via that notebook's source + report_expirement/day_0825/report.md -- NOT
patience=1e9, which was an error in Project_Architecture_Ladder_Spec.md's protocol pin).

If seed=0 reproduces 95.42 within +/-0.5, seeds 1-4 are reused from
top3_multiseed_results.pkl (same model/protocol/code path, deterministic seeding) instead
of retraining, to avoid ~1.5-2h of redundant GPU compute. Full metrics (RMSE/MAE/MAPE/R2 +
short/long bias) are recomputed uniformly from stored predictions for all 5 seeds so the
Stage 0 output matches the schema required by Project_Architecture_Ladder_Spec.md.
"""
import os
import sys
import time
import pickle

import numpy as np
import torch
import torch.nn as nn

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
sys.path.insert(0, REPO)

from batteryml.data.databundle import DataBundle
from batteryml.builders import DATA_TRANSFORMATIONS
import BatLiNet  # noqa: F401

CACHE_DIR = os.path.join(REPO, "ipynb", "exp_v3", "matr1_feature_cache")
FEATURE_CACHE_PKL = os.path.join(CACHE_DIR, "feature_cache.pkl")
OLD_MULTISEED_PKL = os.path.join(CACHE_DIR, "top3_multiseed_results.pkl")
OUT_PKL = os.path.join(CACHE_DIR, "stage0_smallcnn_scalar_branch_5seed.pkl")
SOURCE_NOTEBOOK = os.path.join(
    REPO, "ipynb", "exp_v2", "matr", "matr1", "variant_c_tier0_safety_loss_v1.ipynb")

DIFF_BASE = 9
EPOCHS = 1000
EVAL_EVERY = 50
PATIENCE = 350
REFERENCE_RMSE = 95.42
SANITY_TOLERANCE = 0.5
TOP3_FEATURES = ["qdlin_diff_std", "voltage_slope_50_90", "voltage_soc_90"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}  diff_base={DIFF_BASE}  patience={PATIENCE}")


class SmallCNNScalarBranch(nn.Module):
    def __init__(self, n_scalar):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.scalar = nn.Sequential(
            nn.Linear(n_scalar, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
        )
        self.head = nn.Linear(32 + 16, 1)

    def forward(self, x, scalar):
        x = self.features(x)
        x = x.flatten(1)
        scalar = self.scalar(scalar)
        return self.head(torch.cat([x, scalar], dim=1)).squeeze(1)


@torch.no_grad()
def smoothing(feature):
    med = feature.median(-1)[0].unsqueeze(-1).expand(*feature.shape)
    med_diff = (feature - med).abs()
    med_diff_std = med_diff.std(-1, keepdim=True).expand(*feature.shape)
    mask = med_diff > 3 * med_diff_std
    feature = feature.clone()
    feature[mask] = 0.
    return feature


@torch.no_grad()
def _remove_glitches(x, width=25, threshold=3):
    left = torch.roll(x, shifts=1, dims=-1)
    right = torch.roll(x, shifts=-1, dims=-1)
    diff_left = (left - x).abs()
    diff_right = (right - x).abs()
    non_smooth_left = diff_left > diff_left.std(-1, keepdim=True) * threshold
    non_smooth_right = diff_right > diff_right.std(-1, keepdim=True) * threshold
    for _ in range(width):
        non_smooth_left = non_smooth_left | torch.roll(non_smooth_left, shifts=1, dims=-1)
        non_smooth_right = non_smooth_right | torch.roll(non_smooth_right, shifts=-1, dims=-1)
    to_smooth = non_smooth_left & non_smooth_right
    x = x.clone()
    x[to_smooth] = 0.
    return x


@torch.no_grad()
def filter_cycles(feature, enable=True):
    if not enable:
        return feature
    feature = feature.clone()
    max_val = feature.abs().amax(-1)
    max_val_diff = (max_val - max_val.median(-1, keepdim=True)[0]).abs()
    mask = max_val_diff > max_val_diff.std(-1, keepdim=True) * 5
    mean_val = feature.mean(-1)
    mean_val_diff = (mean_val - mean_val.median(-1, keepdim=True)[0]).abs()
    mask |= mean_val_diff > mean_val_diff.std(-1, keepdim=True) * 5
    feature[mask] = 0.
    return feature


@torch.no_grad()
def clean_feature(diffed_feature, num_edge=50, filter_cycles_enable=True):
    feature = diffed_feature.clone()
    feature[..., :num_edge] = smoothing(feature[..., :num_edge])
    feature[..., -num_edge:] = smoothing(feature[..., -num_edge:])
    feature = _remove_glitches(feature)
    feature = filter_cycles(feature, enable=filter_cycles_enable)
    return feature


def run_one_scalar_safety(train_scalar, val_scalar_, test_scalar_, seed, safety_weight=1.0,
                           patience=PATIENCE, tag=""):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

    t0 = time.time()
    db = DataBundle(train_base_feature, train_base_label, test_feature, test_label,
                     feature_transformation=None,
                     label_transformation=DATA_TRANSFORMATIONS.build(
                         {"name": "SequentialDataTransformation", "transformations": [
                             {"name": "LogScaleDataTransformation"}, {"name": "ZScoreDataTransformation"}]}))
    model = SmallCNNScalarBranch(n_scalar=train_scalar.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    feat_dev = db.train_data.feature.to(device)
    label_dev = db.train_data.label.to(device)
    scalar_dev = train_scalar.to(device)
    diffed = clean_feature(feat_dev - feat_dev[:, :, [DIFF_BASE]])

    def predict_on(feature, scalar):
        feat = feature.to(device)
        diffed_e = clean_feature(feat - feat[:, :, [DIFF_BASE]])
        pred = model(diffed_e, scalar.to(device)).cpu()
        return db.label_transformation.inverse_transform(pred).numpy().astype(float).ravel()

    val_true = val_label.numpy().ravel()
    test_true_np = test_label.numpy().ravel()
    best_val_rmse, best_val_epoch, best_state = float("inf"), None, None
    best_val_pred, best_test_pred = None, None

    for epoch in range(EPOCHS):
        model.train(); optimizer.zero_grad()
        pred = model(diffed, scalar_dev)
        error = pred - label_dev
        weight = torch.ones_like(error)
        weight[error > 0] = safety_weight
        loss = (weight * error ** 2).mean()
        loss.backward(); optimizer.step()
        if (epoch + 1) % EVAL_EVERY == 0 or (epoch + 1) == EPOCHS:
            model.eval()
            with torch.no_grad():
                val_pred = predict_on(val_feature, val_scalar_)
                test_pred_demo = predict_on(test_feature, test_scalar_)
            val_rmse = float(np.sqrt(np.mean((val_true - val_pred) ** 2)))
            test_rmse_demo = float(np.sqrt(np.mean((test_true_np - test_pred_demo) ** 2)))
            improved = val_rmse < best_val_rmse
            if improved:
                best_val_rmse, best_val_epoch = val_rmse, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_val_pred, best_test_pred = val_pred.copy(), test_pred_demo.copy()
            print(f"  [{tag}] [{epoch+1:4d}/{EPOCHS}] loss={loss.item():.4f} "
                  f"val_RMSE={val_rmse:7.2f}  test_RMSE={test_rmse_demo:7.2f}"
                  f"{'  <- best val' if improved else ''}", flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
            if (epoch + 1) - best_val_epoch >= patience:
                print(f"  [{tag}] early stop at epoch {epoch+1} (no improvement for "
                      f"{(epoch+1)-best_val_epoch} epochs)", flush=True)
                break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        y_pred_test = predict_on(test_feature, test_scalar_)
    rmse = float(np.sqrt(np.mean((test_true_np - y_pred_test) ** 2)))
    train_seconds = time.time() - t0

    del model, optimizer, feat_dev, label_dev, scalar_dev, diffed, best_state
    if device == "cuda":
        torch.cuda.empty_cache()

    return dict(rmse=rmse, best_val_epoch=best_val_epoch, best_val_rmse=best_val_rmse,
                val_pred=best_val_pred, test_pred=best_test_pred, train_seconds=train_seconds,
                n_parameters=n_params)


def compute_full_metrics(test_pred, test_true):
    n = len(test_true)
    order = np.argsort(test_true)
    short_idx = order[: n // 3]
    long_idx = order[-(n // 3):]

    def _block(p, t):
        mae = float(np.mean(np.abs(p - t)))
        rmse_ = float(np.sqrt(np.mean((p - t) ** 2)))
        mape = float(np.mean(np.abs((p - t) / t))) * 100
        ss_res = np.sum((p - t) ** 2); ss_tot = np.sum((t - t.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        return mae, rmse_, mape, r2

    mae, rmse_, mape, r2 = _block(test_pred, test_true)
    short_bias = float(np.mean(test_pred[short_idx] - test_true[short_idx]))
    long_bias = float(np.mean(test_pred[long_idx] - test_true[long_idx]))
    return dict(rmse=rmse_, mae=mae, mape=mape, r2=r2, short_bias=short_bias, long_bias=long_bias)


STAGE0_ALREADY_DONE = os.path.exists(OUT_PKL)

if STAGE0_ALREADY_DONE:
    with open(OUT_PKL, "rb") as f:
        _out = pickle.load(f)
    print(f"Stage 0 already complete -- loading cached result from {OUT_PKL} (no retraining).")
    agg = _out["aggregate"]
    print(f"RMSE mean/std: {agg['rmse_mean']:.2f} / {agg['rmse_std']:.2f}")
    print(f"MAE  mean/std: {agg['mae_mean']:.2f} / {agg['mae_std']:.2f}")
    print(f"R2   mean/std: {agg['r2_mean']:.3f} / {agg['r2_std']:.3f}")
    print(f"short_bias mean/std: {agg['short_bias_mean']:+.2f} / {agg['short_bias_std']:.2f}")
    print(f"long_bias  mean/std: {agg['long_bias_mean']:+.2f} / {agg['long_bias_std']:.2f}")
    print(f"n_parameters: {agg['n_parameters']:,}")

if __name__ == "__main__" and not STAGE0_ALREADY_DONE:
    with open(FEATURE_CACHE_PKL, "rb") as f:
        _C = pickle.load(f)
    train_base_feature = _C["train_base_feature"]; train_base_label = _C["train_base_label"]
    val_feature = _C["val_feature"]; val_label = _C["val_label"]
    test_feature = _C["test_feature"]; test_label = _C["test_label"]
    train_base_scalar = _C["train_base_scalar"]; val_scalar = _C["val_scalar"]; test_scalar = _C["test_scalar"]
    ALL_FEATURE_NAMES = _C["feature_names"]

    _idxs = [ALL_FEATURE_NAMES.index(w) for w in TOP3_FEATURES]
    tr_s = train_base_scalar[:, _idxs]
    va_s = val_scalar[:, _idxs]
    te_s = test_scalar[:, _idxs]
    test_true_np = test_label.numpy().ravel()

    print(f"\n=== Stage 0: seed=0 fresh verification (patience={PATIENCE}) ===")
    seed0_result = run_one_scalar_safety(tr_s, va_s, te_s, seed=0, safety_weight=1.0, tag="seed0")
    diff = abs(seed0_result["rmse"] - REFERENCE_RMSE)
    sanity_ok = diff <= SANITY_TOLERANCE
    print(f"\nseed=0 fresh RMSE={seed0_result['rmse']:.4f}  vs reference={REFERENCE_RMSE}  "
          f"diff={diff:.4f}  -- {'PASS' if sanity_ok else 'FAIL'}")

    if not sanity_ok:
        print("\nSANITY GATE FAILED. Stopping before assembling Stage 0 artifact.")
        print("Audit order: clean_feature, DIFF_BASE, scalar feature ordering, split indices, "
              "label transform, AdamW instantiation/default decay, metric code, checkpoint "
              "restoration, deterministic/CUDA settings.")
        sys.exit(1)

    print(f"\nGate passed (diff={diff:.4f} <= {SANITY_TOLERANCE}). "
          f"Reusing seeds 1-4 from {os.path.basename(OLD_MULTISEED_PKL)} "
          f"(same model/protocol/code path, deterministic seeding) instead of retraining.")

    with open(OLD_MULTISEED_PKL, "rb") as f:
        old_multiseed = pickle.load(f)

    seeds_out = {}
    seeds_out[0] = dict(
        rmse=None, mae=None, mape=None, r2=None, short_bias=None, long_bias=None,
        best_epoch=seed0_result["best_val_epoch"], best_val_metric=seed0_result["best_val_rmse"],
        val_pred=seed0_result["val_pred"], test_pred=seed0_result["test_pred"],
        val_true=val_label.numpy().ravel(), test_true=test_true_np,
        train_seconds=seed0_result["train_seconds"], n_parameters=seed0_result["n_parameters"],
        source="fresh_run_this_session",
    )
    for s in [1, 2, 3, 4]:
        old = old_multiseed[s]
        seeds_out[s] = dict(
            rmse=None, mae=None, mape=None, r2=None, short_bias=None, long_bias=None,
            best_epoch=old["epoch"], best_val_metric=None,
            val_pred=old["val_pred"], test_pred=old["test_pred"],
            val_true=val_label.numpy().ravel(), test_true=test_true_np,
            train_seconds=None, n_parameters=seed0_result["n_parameters"],
            source="reused_top3_multiseed_results.pkl",
        )

    for s, d in seeds_out.items():
        m = compute_full_metrics(d["test_pred"], d["test_true"])
        d.update(m)

    rmses = np.array([seeds_out[s]["rmse"] for s in range(5)])
    maes = np.array([seeds_out[s]["mae"] for s in range(5)])
    r2s = np.array([seeds_out[s]["r2"] for s in range(5)])
    short_biases = np.array([seeds_out[s]["short_bias"] for s in range(5)])
    long_biases = np.array([seeds_out[s]["long_bias"] for s in range(5)])
    train_seconds_known = [seeds_out[s]["train_seconds"] for s in range(5) if seeds_out[s]["train_seconds"]]

    aggregate = dict(
        rmse_mean=float(rmses.mean()), rmse_std=float(rmses.std()),
        r2_mean=float(r2s.mean()), r2_std=float(r2s.std()),
        mae_mean=float(maes.mean()), mae_std=float(maes.std()),
        short_bias_mean=float(short_biases.mean()), short_bias_std=float(short_biases.std()),
        long_bias_mean=float(long_biases.mean()), long_bias_std=float(long_biases.std()),
        train_seconds_mean=float(np.mean(train_seconds_known)) if train_seconds_known else None,
        n_parameters=seed0_result["n_parameters"],
        unstable=None, unstable_reason=None,
    )

    out = dict(
        stage="stage0",
        model_name="SmallCNNScalarBranch",
        source_model_file=SOURCE_NOTEBOOK,
        source_runner="run_one_scalar_safety",
        reference_rmse=REFERENCE_RMSE,
        preprocessing=dict(
            diff_base=DIFF_BASE,
            operation="clean_feature(feat - feat[:, :, [9], :])",
            clean_feature_source=SOURCE_NOTEBOOK,
        ),
        protocol=dict(
            epochs=EPOCHS, eval_every=EVAL_EVERY, patience=PATIENCE,
            optimizer="AdamW", lr=0.001, weight_decay_argument=None,
            weight_decay_effective_default=0.01,
            label_transform="LogScaleDataTransformation+ZScoreDataTransformation",
            split_id="NEW_TRAIN_BASE_IDS/NEW_VAL_IDS/OFFICIAL_TEST_IDS (matr1_feature_cache/feature_cache.pkl)",
        ),
        seeds=seeds_out,
        aggregate=aggregate,
        note=(
            "seed=0 trained fresh this session (patience=350, verified vs historical 95.42). "
            "seeds 1-4 reused from top3_multiseed_results.pkl (same code path/protocol/seeds, "
            "deterministic) rather than retrained, to avoid redundant GPU compute -- see "
            "report_expirement/day_0901/plan.md for rationale. patience=350 (NOT 1e9) is the "
            "correct historical protocol; Project_Architecture_Ladder_Spec.md's PATIENCE=1e9 "
            "pin was an error, corrected this session."
        ),
    )

    with open(OUT_PKL, "wb") as f:
        pickle.dump(out, f)

    print(f"\n=== Stage 0 aggregate (5-seed) ===")
    print(f"RMSE mean/std: {aggregate['rmse_mean']:.2f} / {aggregate['rmse_std']:.2f}")
    print(f"MAE  mean/std: {aggregate['mae_mean']:.2f} / {aggregate['mae_std']:.2f}")
    print(f"R2   mean/std: {aggregate['r2_mean']:.3f} / {aggregate['r2_std']:.3f}")
    print(f"short_bias mean/std: {aggregate['short_bias_mean']:+.2f} / {aggregate['short_bias_std']:.2f}")
    print(f"long_bias  mean/std: {aggregate['long_bias_mean']:+.2f} / {aggregate['long_bias_std']:.2f}")
    print(f"n_parameters: {aggregate['n_parameters']:,}")
    print(f"\nSaved -> {OUT_PKL}")
