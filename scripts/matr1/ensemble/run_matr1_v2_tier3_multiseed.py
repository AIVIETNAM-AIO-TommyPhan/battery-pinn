"""Tier3 (Wiener drift/diffusion), ported to the CURRENT session's V2 backbone
(SmallCNNScalarBranchAxisAware, axis-aware attention pooling) -- the historical
Tier3 result (81.69+/-4.62, 5-seed) used the OLD V2PINN class
(run_v2_min_tier3_multiseed.py), never the axis-aware architecture this session
actually uses for the SOH-head (Tier1) results. This script closes that gap.

Exact loss formulas reused verbatim from run_v2_pinn_tiers.py / the Wiener_PINN
checklist (unchanged): lambda_soh=0.01, lambda_mono=0.05, lambda_wiener=0.05,
mono_tolerance=0.002, patience=200 (V2's own protocol). V2 only (use_pe=False) --
Tier3 was never tried on V1 or Inter-Embedding, matching historical scope.

CLI: python run_matr1_v2_tier3_multiseed.py [seed]
"""
import os
import sys
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
sys.path.insert(0, REPO)

from batteryml.data.databundle import DataBundle
from batteryml.builders import DATA_TRANSFORMATIONS
import BatLiNet  # noqa: F401

CACHE_DIR = os.path.join(REPO, "ipynb", "exp_v3", "matr1_feature_cache")
FEATURE_CACHE_PKL = os.path.join(CACHE_DIR, "feature_cache.pkl")
SOH_CACHE_PKL = os.path.join(CACHE_DIR, "soh_trajectory_cache.pkl")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
OUT_PKL = os.path.join(CACHE_DIR, f"matr1_v2axisaware_tier3_seed{SEED}.pkl")

DIFF_BASE = 9
EPOCHS = 1000
EVAL_EVERY = 50
PATIENCE = 200
TOP3_FEATURES = ["qdlin_diff_std", "voltage_slope_50_90", "voltage_soc_90"]

LAMBDA_SOH = 0.01
LAMBDA_MONO = 0.05
LAMBDA_WIENER = 0.05
MONO_TOLERANCE = 0.002

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}  diff_base={DIFF_BASE}  seed={SEED}  tier3(axis-aware V2)", flush=True)


@torch.no_grad()
def smoothing(feature):
    med = feature.median(-1)[0].unsqueeze(-1).expand(*feature.shape)
    med_diff = (feature - med).abs()
    med_diff_std = med_diff.std(-1, keepdim=True).expand(*feature.shape)
    mask = med_diff > 3 * med_diff_std
    feature = feature.clone(); feature[mask] = 0.
    return feature


@torch.no_grad()
def _remove_glitches(x, width=25, threshold=3):
    left = torch.roll(x, shifts=1, dims=-1); right = torch.roll(x, shifts=-1, dims=-1)
    diff_left = (left - x).abs(); diff_right = (right - x).abs()
    nsl = diff_left > diff_left.std(-1, keepdim=True) * threshold
    nsr = diff_right > diff_right.std(-1, keepdim=True) * threshold
    for _ in range(width):
        nsl = nsl | torch.roll(nsl, shifts=1, dims=-1)
        nsr = nsr | torch.roll(nsr, shifts=-1, dims=-1)
    to_smooth = nsl & nsr
    x = x.clone(); x[to_smooth] = 0.
    return x


@torch.no_grad()
def filter_cycles(feature, enable=True):
    if not enable:
        return feature
    feature = feature.clone()
    max_val = feature.abs().amax(-1)
    mvd = (max_val - max_val.median(-1, keepdim=True)[0]).abs()
    mask = mvd > mvd.std(-1, keepdim=True) * 5
    mean_val = feature.mean(-1)
    mnd = (mean_val - mean_val.median(-1, keepdim=True)[0]).abs()
    mask |= mnd > mnd.std(-1, keepdim=True) * 5
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
        self.score = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        logits = self.score(x)
        weights = torch.softmax(logits, dim=1)
        pooled = (weights * x).sum(dim=1)
        return pooled, weights.squeeze(-1)


class V2AxisAwareTier3(nn.Module):
    """V2 backbone (axis-aware attention, no PE) + SOH/monotonicity/Wiener heads."""

    def __init__(self, n_scalar, n_cycles=100):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.pool = CycleAttentionPooling(d_model=32, hidden_dim=16)
        self.scalar = nn.Sequential(nn.Linear(n_scalar, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
        self.rul_head = nn.Linear(32 + 16, 1)
        self.soh_head = nn.Linear(32 + 16, n_cycles)
        self.mu_head = nn.Linear(32 + 16, n_cycles)
        self.sigma_head = nn.Linear(32 + 16, n_cycles)

    def forward(self, x, scalar):
        conv3_out = self.backbone(x)
        cycle_tokens = conv3_out.mean(dim=-1).transpose(1, 2)
        signal_embedding, _ = self.pool(cycle_tokens)
        scalar_embedding = self.scalar(scalar)
        h = torch.cat([signal_embedding, scalar_embedding], dim=1)
        rul = self.rul_head(h).squeeze(1)
        soh = self.soh_head(h)
        mu = self.mu_head(h)
        sigma = F.softplus(self.sigma_head(h)) + 1e-6
        return rul, soh, mu, sigma


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


def run_tier3(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

    db = DataBundle(train_base_feature, train_base_label, test_feature, test_label,
                     feature_transformation=None,
                     label_transformation=DATA_TRANSFORMATIONS.build(
                         {"name": "SequentialDataTransformation", "transformations": [
                             {"name": "LogScaleDataTransformation"}, {"name": "ZScoreDataTransformation"}]}))
    model = V2AxisAwareTier3(n_scalar=tr_s.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    feat_dev = db.train_data.feature.to(device)
    label_dev = db.train_data.label.to(device)
    scalar_dev = tr_s.to(device)
    diffed = clean_feature(feat_dev - feat_dev[:, :, [DIFF_BASE]])
    train_soh_dev = train_base_soh.to(device)

    def predict_on(feature, scalar):
        feat = feature.to(device)
        diffed_e = clean_feature(feat - feat[:, :, [DIFF_BASE]])
        with torch.no_grad():
            pred, _, _, _ = model(diffed_e, scalar.to(device))
            pred = pred.cpu()
        return db.label_transformation.inverse_transform(pred).numpy().astype(float).ravel()

    val_true = val_label.numpy().ravel()
    test_true_np = test_label.numpy().ravel()
    best_val_rmse, best_val_epoch, best_state = float("inf"), None, None
    best_val_pred, best_test_pred = None, None

    for epoch in range(EPOCHS):
        model.train(); optimizer.zero_grad()
        pred, soh_pred, mu, sigma = model(diffed, scalar_dev)
        loss_rul = ((pred - label_dev) ** 2).mean()
        loss_soh = F.smooth_l1_loss(soh_pred, train_soh_dev)
        loss_mono = F.relu(soh_pred[:, 1:] - soh_pred[:, :-1] - MONO_TOLERANCE).mean()
        # Wiener-process NLL: delta_D ~ N(mu, sigma^2), delta_t = 1 cycle (dropped from the
        # formulas below since delta_t=1 makes it a no-op; keep delta_t explicit only if
        # cycles are ever unevenly spaced)
        D = 1.0 - soh_pred
        delta_D = D[:, 1:] - D[:, :-1]
        residual = delta_D - mu[:, :-1]
        variance = sigma[:, :-1] ** 2 + 1e-6
        loss_wiener = (0.5 * residual ** 2 / variance + 0.5 * torch.log(variance)).mean()
        loss = loss_rul + LAMBDA_SOH * loss_soh + LAMBDA_MONO * loss_mono + LAMBDA_WIENER * loss_wiener
        loss.backward(); optimizer.step()

        if (epoch + 1) % EVAL_EVERY == 0 or (epoch + 1) == EPOCHS:
            model.eval()
            val_pred = predict_on(val_feature, va_s)
            test_pred = predict_on(test_feature, te_s)
            val_rmse = float(np.sqrt(np.mean((val_true - val_pred) ** 2)))
            test_rmse = float(np.sqrt(np.mean((test_true_np - test_pred) ** 2)))
            improved = val_rmse < best_val_rmse
            if improved:
                best_val_rmse, best_val_epoch = val_rmse, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_val_pred, best_test_pred = val_pred.copy(), test_pred.copy()
            print(f"  [tier3] [{epoch+1:4d}/{EPOCHS}] loss={loss.item():.4f} "
                  f"(rul={loss_rul.item():.4f} soh={loss_soh.item():.4f} "
                  f"mono={loss_mono.item():.4f} wiener={loss_wiener.item():.4f}) "
                  f"val_RMSE={val_rmse:7.2f}  test_RMSE={test_rmse:7.2f}"
                  f"{'  <- best val' if improved else ''}", flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
            if (epoch + 1) - best_val_epoch >= PATIENCE:
                print(f"  [tier3] early stop at epoch {epoch+1}", flush=True)
                break

    model.load_state_dict(best_state); model.eval()
    metrics = compute_full_metrics(best_test_pred, test_true_np)
    print(f"[tier3] FINAL test_RMSE={metrics['rmse']:.2f}\n", flush=True)

    del model, optimizer, feat_dev, label_dev, scalar_dev, diffed, best_state
    if device == "cuda":
        torch.cuda.empty_cache()

    return dict(best_epoch=best_val_epoch, val_pred=best_val_pred, test_pred=best_test_pred,
                val_true=val_true, test_true=test_true_np, n_parameters=n_params, **metrics)


if __name__ == "__main__":
    with open(FEATURE_CACHE_PKL, "rb") as f:
        _C = pickle.load(f)
    train_base_feature = _C["train_base_feature"]; train_base_label = _C["train_base_label"]
    val_feature = _C["val_feature"]; val_label = _C["val_label"]
    test_feature = _C["test_feature"]; test_label = _C["test_label"]
    train_base_scalar = _C["train_base_scalar"]; val_scalar = _C["val_scalar"]; test_scalar = _C["test_scalar"]
    ALL_FEATURE_NAMES = _C["feature_names"]
    _idxs = [ALL_FEATURE_NAMES.index(w) for w in TOP3_FEATURES]
    tr_s = train_base_scalar[:, _idxs]; va_s = val_scalar[:, _idxs]; te_s = test_scalar[:, _idxs]

    with open(SOH_CACHE_PKL, "rb") as f:
        _S = pickle.load(f)
    train_base_soh = _S["train_soh"]

    res = run_tier3(SEED)
    print(f"=== RESULT seed={SEED}: RMSE={res['rmse']:.2f}  MAE={res['mae']:.2f}  MAPE={res['mape']:.2f}%  "
          f"R2={res['r2']:.3f}  short_bias={res['short_bias']:+.2f}  long_bias={res['long_bias']:+.2f}")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(res, f)
    print(f"Saved -> {OUT_PKL}")
