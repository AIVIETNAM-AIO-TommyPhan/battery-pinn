"""Re-opening PINN Task #7 (day_0901/plan.md): Tier0 (safety-weighted loss) on V2,
the best architecture found this session (axis-aware attention pooling, no PE).

Historical context (day_0827, closed track): Tier0 on the ORIGINAL SmallCNNScalarBranch
failed because short_bias(+99.84) and long_bias(-18.56) were OPPOSITE-signed --
penalizing over-prediction (pred>true) helped short-life but pushed long-life further
under (sw=2.0: long_bias -18.56 -> -67.09), net RMSE got worse (95.42 -> 103.37).

V2 is structurally different: short_bias is positive in 5/5 confirmed seeds, and
long_bias is SAME-signed (positive) in 3/5 seeds (0,1,2) -- only seeds 3,4 (the two
best-RMSE seeds) have negative long_bias. Hypothesis: safety-weight may help more of
V2's seeds without the old backfire, but seeds 3/4 could still see the old failure
mode since their long_bias already leans negative.

Tested at seed=0 first (short_bias=+55.37, long_bias=+31.32 -- same-signed, the
favorable case) per "run one seed first" convention. safety_weight swept over
[1.5, 2.0] (1.0 == plain V2, already have: RMSE=75.32).

Architecture: identical to Stage 1 V2 (attention pooling, no PE), top3 scalar
features, patience=200 (V2's confirmed protocol), EPOCHS=1000, EVAL_EVERY=50.
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
CONFIRM_PKL = os.path.join(CACHE_DIR, "stage1_axis_aware_confirm_5seed.pkl")
OUT_PKL = os.path.join(CACHE_DIR, "v2_tier0_safety_loss_seed0.pkl")

DIFF_BASE = 9
EPOCHS = 1000
EVAL_EVERY = 50
PATIENCE = 200
TOP3_FEATURES = ["qdlin_diff_std", "voltage_slope_50_90", "voltage_soc_90"]
SAFETY_WEIGHTS = [1.5, 2.0]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}  diff_base={DIFF_BASE}  patience={PATIENCE}")


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


class CycleAttentionPooling(nn.Module):
    def __init__(self, d_model=32, hidden_dim=16):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        logits = self.score(x)
        weights = torch.softmax(logits, dim=1)
        pooled = (weights * x).sum(dim=1)
        return pooled, weights.squeeze(-1)


class V2AttentionOnly(nn.Module):
    def __init__(self, n_scalar):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.pool = CycleAttentionPooling(d_model=32, hidden_dim=16)
        self.scalar = nn.Sequential(
            nn.Linear(n_scalar, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
        )
        self.head = nn.Linear(32 + 16, 1)

    def forward(self, x, scalar):
        conv3_out = self.backbone(x)
        cycle_tokens = conv3_out.mean(dim=-1).transpose(1, 2)
        signal_embedding, attn = self.pool(cycle_tokens)
        scalar_embedding = self.scalar(scalar)
        out = self.head(torch.cat([signal_embedding, scalar_embedding], dim=1))
        return out.squeeze(1)


def compute_full_metrics(test_pred, test_true):
    n = len(test_true)
    order = np.argsort(test_true)
    short_idx = order[: n // 3]
    long_idx = order[-(n // 3):]
    mae = float(np.mean(np.abs(test_pred - test_true)))
    rmse_ = float(np.sqrt(np.mean((test_pred - test_true) ** 2)))
    mape = float(np.mean(np.abs((test_pred - test_true) / test_true))) * 100
    ss_res = np.sum((test_pred - test_true) ** 2); ss_tot = np.sum((test_true - test_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    short_bias = float(np.mean(test_pred[short_idx] - test_true[short_idx]))
    long_bias = float(np.mean(test_pred[long_idx] - test_true[long_idx]))
    return dict(rmse=rmse_, mae=mae, mape=mape, r2=r2, short_bias=short_bias, long_bias=long_bias)


def run_one(safety_weight, seed=0):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

    t0 = time.time()
    db = DataBundle(train_base_feature, train_base_label, test_feature, test_label,
                     feature_transformation=None,
                     label_transformation=DATA_TRANSFORMATIONS.build(
                         {"name": "SequentialDataTransformation", "transformations": [
                             {"name": "LogScaleDataTransformation"}, {"name": "ZScoreDataTransformation"}]}))
    model = V2AttentionOnly(n_scalar=tr_s.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    feat_dev = db.train_data.feature.to(device)
    label_dev = db.train_data.label.to(device)
    scalar_dev = tr_s.to(device)
    diffed = clean_feature(feat_dev - feat_dev[:, :, [DIFF_BASE]])

    def predict_on(feature, scalar):
        feat = feature.to(device)
        diffed_e = clean_feature(feat - feat[:, :, [DIFF_BASE]])
        with torch.no_grad():
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
            val_pred = predict_on(val_feature, va_s)
            test_pred_demo = predict_on(test_feature, te_s)
            val_rmse = float(np.sqrt(np.mean((val_true - val_pred) ** 2)))
            test_rmse_demo = float(np.sqrt(np.mean((test_true_np - test_pred_demo) ** 2)))
            improved = val_rmse < best_val_rmse
            if improved:
                best_val_rmse, best_val_epoch = val_rmse, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_val_pred, best_test_pred = val_pred.copy(), test_pred_demo.copy()
            print(f"  [sw{safety_weight}] [{epoch+1:4d}/{EPOCHS}] loss={loss.item():.4f} "
                  f"val_RMSE={val_rmse:7.2f}  test_RMSE={test_rmse_demo:7.2f}"
                  f"{'  <- best val' if improved else ''}", flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
            if (epoch + 1) - best_val_epoch >= PATIENCE:
                print(f"  [sw{safety_weight}] early stop at epoch {epoch+1} (no improvement for "
                      f"{(epoch+1)-best_val_epoch} epochs)", flush=True)
                break

    model.load_state_dict(best_state); model.eval()
    train_seconds = time.time() - t0
    metrics = compute_full_metrics(best_test_pred, test_true_np)

    del model, optimizer, feat_dev, label_dev, scalar_dev, diffed, best_state
    if device == "cuda":
        torch.cuda.empty_cache()

    return dict(best_epoch=best_val_epoch, val_pred=best_val_pred, test_pred=best_test_pred,
                val_true=val_true, test_true=test_true_np, train_seconds=train_seconds,
                n_parameters=n_params, safety_weight=safety_weight, **metrics)


if __name__ == "__main__":
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

    with open(CONFIRM_PKL, "rb") as f:
        confirm = pickle.load(f)
    v2_baseline = confirm["V2"][0]

    results = {}
    if os.path.exists(OUT_PKL):
        with open(OUT_PKL, "rb") as f:
            results = pickle.load(f)
        print(f"resuming: {list(results.keys())} already done")

    for sw in SAFETY_WEIGHTS:
        if sw in results:
            print(f"sw={sw}: already done, rmse={results[sw]['rmse']:.2f} -- skipping")
            continue
        print(f"\n=== V2 + Tier0 safety_weight={sw}, seed=0, patience={PATIENCE} ===")
        res = run_one(sw, seed=0)
        results[sw] = res
        with open(OUT_PKL, "wb") as f:
            pickle.dump(results, f)
        print(f"sw={sw}: RMSE={res['rmse']:.2f}  MAE={res['mae']:.2f}  R2={res['r2']:.3f}  "
              f"short_bias={res['short_bias']:+.2f}  long_bias={res['long_bias']:+.2f}  best_epoch={res['best_epoch']}")

    print(f"\n=== Summary: V2 + Tier0 safety-weight vs plain V2 (RMSE={v2_baseline['rmse']:.2f}, "
          f"short_bias={v2_baseline['short_bias']:+.2f}, long_bias={v2_baseline['long_bias']:+.2f}) ===")
    print(f"sw=1.0 (plain V2): RMSE={v2_baseline['rmse']:.2f}  short_bias={v2_baseline['short_bias']:+.2f}  long_bias={v2_baseline['long_bias']:+.2f}")
    for sw in SAFETY_WEIGHTS:
        r = results[sw]
        delta = r["rmse"] - v2_baseline["rmse"]
        print(f"sw={sw}: RMSE={r['rmse']:.2f}  delta={delta:+.2f}  short_bias={r['short_bias']:+.2f}  long_bias={r['long_bias']:+.2f}")
