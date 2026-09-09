"""Ablation: does adding a 4th scalar feature (qdlin_diff_l2, rank-4 from the
day_0825 single-feature ablation) improve Stage 2's best candidate (V4, ConvLSTM1D)?

All 34 candidate scalar features already live in feature_cache.pkl (built once,
no rebuild needed) -- see ipynb/exp_v2/matr/matr1/variant_c_feature_cache_ablation_v1.ipynb
cell-6 (extract_new_scalars) for how qdlin_diff_l2 is computed and cell-11 for how
top3 was derived (single-feature ablation vs CNN-only=130.83, not hand-picked).

Architecture identical to Stage 2 V4 (see run_stage2_percycle_screen.py) except
n_scalar=4 instead of 3. Seed=0 only, same protocol (patience=350), compared
directly against V4's confirmed screen result (RMSE=90.18).
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
STAGE2_SCREEN_PKL = os.path.join(CACHE_DIR, "stage2_per_cycle_temporal_screen_seed0.pkl")
OUT_PKL = os.path.join(CACHE_DIR, "stage2_v4_top4_test.pkl")

DIFF_BASE = 9
EPOCHS = 1000
EVAL_EVERY = 50
PATIENCE = 350
TOP4_FEATURES = ["qdlin_diff_std", "voltage_slope_50_90", "voltage_soc_90", "qdlin_diff_l2"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}  diff_base={DIFF_BASE}  patience={PATIENCE}  features={TOP4_FEATURES}")


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


class ConvLSTM1DCell(nn.Module):
    def __init__(self, in_channels=32, hidden_channels=32, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv1d(in_channels + hidden_channels, 4 * hidden_channels,
                                kernel_size=kernel_size, padding=padding)

    def forward(self, x_t, h_prev, c_prev):
        combined = torch.cat([x_t, h_prev], dim=1)
        gates = self.gates(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTM1D(nn.Module):
    def __init__(self, in_channels=32, hidden_channels=32, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTM1DCell(in_channels, hidden_channels, kernel_size)
        self.hidden_channels = hidden_channels

    def forward(self, x):
        assert x.ndim == 4
        B, L, C, W = x.shape
        h = x.new_zeros(B, self.hidden_channels, W)
        c = x.new_zeros(B, self.hidden_channels, W)
        for t in range(L):
            h, c = self.cell(x[:, t], h, c)
        return h


class CNNPerCycleConvLSTM(nn.Module):
    def __init__(self, n_scalar):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=7, padding=3), nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.convlstm = ConvLSTM1D(in_channels=32, hidden_channels=32, kernel_size=3)
        self.scalar = nn.Sequential(
            nn.Linear(n_scalar, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
        )
        self.head = nn.Linear(32 + 16, 1)

    def forward(self, x, scalar):
        B = x.shape[0]
        x_cycles = x.permute(0, 2, 1, 3).reshape(B * 100, 6, 1000)
        enc = self.encoder(x_cycles)  # [B*100, 32, 250]
        enc_seq = enc.reshape(B, 100, 32, 250)
        hidden = self.convlstm(enc_seq)  # [B,32,250]
        signal_embedding = nn.functional.adaptive_avg_pool1d(hidden, 1).squeeze(-1)
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


def run(seed=0):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

    t0 = time.time()
    db = DataBundle(train_base_feature, train_base_label, test_feature, test_label,
                     feature_transformation=None,
                     label_transformation=DATA_TRANSFORMATIONS.build(
                         {"name": "SequentialDataTransformation", "transformations": [
                             {"name": "LogScaleDataTransformation"}, {"name": "ZScoreDataTransformation"}]}))
    model = CNNPerCycleConvLSTM(n_scalar=tr_s.shape[1]).to(device)
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
        loss = ((pred - label_dev) ** 2).mean()
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
            print(f"  [top4] [{epoch+1:4d}/{EPOCHS}] loss={loss.item():.4f} "
                  f"val_RMSE={val_rmse:7.2f}  test_RMSE={test_rmse_demo:7.2f}"
                  f"{'  <- best val' if improved else ''}", flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
            if (epoch + 1) - best_val_epoch >= PATIENCE:
                print(f"  [top4] early stop at epoch {epoch+1} (no improvement for "
                      f"{(epoch+1)-best_val_epoch} epochs)", flush=True)
                break

    model.load_state_dict(best_state); model.eval()
    train_seconds = time.time() - t0
    metrics = compute_full_metrics(best_test_pred, test_true_np)

    return dict(best_epoch=best_val_epoch, val_pred=best_val_pred, test_pred=best_test_pred,
                val_true=val_true, test_true=test_true_np, train_seconds=train_seconds,
                n_parameters=n_params, features=TOP4_FEATURES, **metrics)


if __name__ == "__main__":
    with open(FEATURE_CACHE_PKL, "rb") as f:
        _C = pickle.load(f)
    train_base_feature = _C["train_base_feature"]; train_base_label = _C["train_base_label"]
    val_feature = _C["val_feature"]; val_label = _C["val_label"]
    test_feature = _C["test_feature"]; test_label = _C["test_label"]
    train_base_scalar = _C["train_base_scalar"]; val_scalar = _C["val_scalar"]; test_scalar = _C["test_scalar"]
    ALL_FEATURE_NAMES = _C["feature_names"]

    _idxs = [ALL_FEATURE_NAMES.index(w) for w in TOP4_FEATURES]
    tr_s = train_base_scalar[:, _idxs]
    va_s = val_scalar[:, _idxs]
    te_s = test_scalar[:, _idxs]

    with open(STAGE2_SCREEN_PKL, "rb") as f:
        stage2 = pickle.load(f)
    v4_top3_rmse = stage2["V4"]["rmse"]

    if os.path.exists(OUT_PKL):
        with open(OUT_PKL, "rb") as f:
            res = pickle.load(f)
        print(f"already done: rmse={res['rmse']:.2f}")
    else:
        print(f"\n=== Stage 2 V4 + top4 (qdlin_diff_l2 added) seed=0, patience={PATIENCE} ===")
        res = run(seed=0)
        with open(OUT_PKL, "wb") as f:
            pickle.dump(res, f)

    delta = res["rmse"] - v4_top3_rmse
    print(f"\n=== Result ===")
    print(f"V4 top3 (baseline): RMSE={v4_top3_rmse:.2f}")
    print(f"V4 top4 (+qdlin_diff_l2): RMSE={res['rmse']:.2f}  MAE={res['mae']:.2f}  R2={res['r2']:.3f}  "
          f"short_bias={res['short_bias']:+.2f}  long_bias={res['long_bias']:+.2f}  best_epoch={res['best_epoch']}")
    print(f"delta={delta:+.2f}  ({'IMPROVED' if delta < 0 else 'WORSE'})")
