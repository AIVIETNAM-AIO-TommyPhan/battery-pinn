"""Ablation: entropy regularization on V2's attention pooling (no PE, attention).

Motivation (this session, day_0901): V2 seed=2 (RMSE=99.58, worst of 5 confirmed
seeds) was diagnosed via attention_weights -- its attention collapses onto a very
narrow band of end-cycles (argmax range [45,47], std=0.45 across test cells) vs
seed=0's broader spread (argmax range [43,49], std=1.41). Hypothesis: penalizing
low-entropy (over-concentrated) attention during training should reduce this
collapse and stabilize seed-to-seed variance.

Pre-declared BEFORE looking at any test result from this ablation (per project
rule: never pick hyperparameters by test-side peeking):
    ENTROPY_LAMBDA = 0.02
    loss = mse_loss - ENTROPY_LAMBDA * mean(entropy(attention_weights))
    (subtracting entropy means the optimizer is rewarded for higher entropy,
    i.e. penalized for collapsing onto few cycles)

Tested on seed=0 (known-good, RMSE=75.32 baseline) and seed=2 (known-bad,
RMSE=99.58 baseline) only -- directly checks whether the fix helps the failure
case without hurting the good case, before committing to a full 5-seed sweep.
Protocol otherwise identical to V2's confirmed protocol: patience=200, EPOCHS=1000,
EVAL_EVERY=50, AdamW(lr=1e-3) default decay, DIFF_BASE=9 + clean_feature.
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
OUT_PKL = os.path.join(CACHE_DIR, "v2_entropy_reg_ablation.pkl")

DIFF_BASE = 9
EPOCHS = 1000
EVAL_EVERY = 50
PATIENCE = 200
TOP3_FEATURES = ["qdlin_diff_std", "voltage_slope_50_90", "voltage_soc_90"]
ENTROPY_LAMBDA = 0.02  # pre-declared, see module docstring
TEST_SEEDS = [0, 2]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}  diff_base={DIFF_BASE}  patience={PATIENCE}  entropy_lambda={ENTROPY_LAMBDA}")


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
    """Exact V2 architecture: no PE, attention pooling."""

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
        self.last_attn = None

    def forward(self, x, scalar):
        conv3_out = self.backbone(x)
        cycle_tokens = conv3_out.mean(dim=-1).transpose(1, 2)  # [B,50,32]
        signal_embedding, attn = self.pool(cycle_tokens)
        self.last_attn = attn  # keep graph attached during training for the entropy term
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


def attn_diag(attn_np):
    entropy = -np.sum(attn_np * np.log(attn_np + 1e-12), axis=1)
    max_w = attn_np.max(axis=1)
    argmax = attn_np.argmax(axis=1)
    return dict(entropy_mean=float(entropy.mean()), max_w_mean=float(max_w.mean()),
                argmax_std=float(argmax.std()), argmax_range=[int(argmax.min()), int(argmax.max())])


def run_one_seed(seed, tag):
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

    def predict_on(feature, scalar, capture_attn=False):
        feat = feature.to(device)
        diffed_e = clean_feature(feat - feat[:, :, [DIFF_BASE]])
        with torch.no_grad():
            pred = model(diffed_e, scalar.to(device))
            attn = model.last_attn.cpu().numpy() if capture_attn else None
        pred = pred.cpu()
        return db.label_transformation.inverse_transform(pred).numpy().astype(float).ravel(), attn

    val_true = val_label.numpy().ravel()
    test_true_np = test_label.numpy().ravel()
    best_val_rmse, best_val_epoch, best_state = float("inf"), None, None
    best_val_pred, best_test_pred, best_attn = None, None, None

    for epoch in range(EPOCHS):
        model.train(); optimizer.zero_grad()
        pred = model(diffed, scalar_dev)
        mse = ((pred - label_dev) ** 2).mean()
        attn = model.last_attn
        entropy = -(attn * torch.log(attn + 1e-12)).sum(dim=1).mean()
        loss = mse - ENTROPY_LAMBDA * entropy
        loss.backward(); optimizer.step()
        if (epoch + 1) % EVAL_EVERY == 0 or (epoch + 1) == EPOCHS:
            model.eval()
            val_pred, _ = predict_on(val_feature, va_s)
            test_pred_demo, _ = predict_on(test_feature, te_s)
            val_rmse = float(np.sqrt(np.mean((val_true - val_pred) ** 2)))
            test_rmse_demo = float(np.sqrt(np.mean((test_true_np - test_pred_demo) ** 2)))
            improved = val_rmse < best_val_rmse
            if improved:
                best_val_rmse, best_val_epoch = val_rmse, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_val_pred, best_test_pred = val_pred.copy(), test_pred_demo.copy()
                _, best_attn = predict_on(test_feature, te_s, capture_attn=True)
            print(f"  [{tag}] [{epoch+1:4d}/{EPOCHS}] mse={mse.item():.4f} entropy={entropy.item():.3f} "
                  f"loss={loss.item():.4f} val_RMSE={val_rmse:7.2f}  test_RMSE={test_rmse_demo:7.2f}"
                  f"{'  <- best val' if improved else ''}", flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
            if (epoch + 1) - best_val_epoch >= PATIENCE:
                print(f"  [{tag}] early stop at epoch {epoch+1} (no improvement for "
                      f"{(epoch+1)-best_val_epoch} epochs)", flush=True)
                break

    model.load_state_dict(best_state); model.eval()
    train_seconds = time.time() - t0
    metrics = compute_full_metrics(best_test_pred, test_true_np)
    diag = attn_diag(best_attn)

    del model, optimizer, feat_dev, label_dev, scalar_dev, diffed, best_state
    if device == "cuda":
        torch.cuda.empty_cache()

    return dict(best_epoch=best_val_epoch, val_pred=best_val_pred, test_pred=best_test_pred,
                val_true=val_true, test_true=test_true_np, train_seconds=train_seconds,
                n_parameters=n_params, attention_weights=best_attn, attn_diag=diag, **metrics)


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

    results = {}
    if os.path.exists(OUT_PKL):
        with open(OUT_PKL, "rb") as f:
            results = pickle.load(f)
        print(f"resuming: {list(results.keys())} already done")

    for seed in TEST_SEEDS:
        if seed in results:
            print(f"seed{seed}: already done, rmse={results[seed]['rmse']:.2f} -- skipping")
            continue
        print(f"\n=== V2 + entropy_reg(lambda={ENTROPY_LAMBDA}): seed={seed}, patience={PATIENCE} ===")
        res = run_one_seed(seed, tag=f"entropy_reg_s{seed}")
        results[seed] = res
        with open(OUT_PKL, "wb") as f:
            pickle.dump(results, f)
        baseline_v2 = confirm["V2"][seed]
        print(f"seed{seed}: entropy_reg RMSE={res['rmse']:.2f}  (plain V2 was {baseline_v2['rmse']:.2f})  "
              f"attn_entropy={res['attn_diag']['entropy_mean']:.3f} (plain V2: "
              f"{attn_diag(np.asarray(baseline_v2['attention_weights']))['entropy_mean']:.3f} if available)  "
              f"argmax_std={res['attn_diag']['argmax_std']:.2f}")

    print(f"\n=== Entropy regularization ablation summary (lambda={ENTROPY_LAMBDA}) ===")
    for seed in TEST_SEEDS:
        r = results[seed]
        baseline_v2 = confirm["V2"][seed]
        delta = r["rmse"] - baseline_v2["rmse"]
        print(f"seed{seed}: plain_V2_rmse={baseline_v2['rmse']:.2f}  entropy_reg_rmse={r['rmse']:.2f}  "
              f"delta={delta:+.2f}  entropy={r['attn_diag']['entropy_mean']:.3f}  "
              f"argmax_std={r['attn_diag']['argmax_std']:.2f}  argmax_range={r['attn_diag']['argmax_range']}")
