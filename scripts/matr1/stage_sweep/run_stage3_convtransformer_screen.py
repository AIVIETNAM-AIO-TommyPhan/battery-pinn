"""Stage 3 -- ConvTransformer-inspired screen, seed=0. day_0901 architecture-ladder track.

V1-V3 ablation (see Project_Architecture_Ladder_Spec.md, Giai doan 3): reuses the
Stage-2 shared Conv1d per-cycle encoder (collapsed to cycle_tokens [B,100,32]), adds
a learnable cycle PE, then a lightweight block: {causal Conv1d k=5, 4-head
self-attention, both} + a pointwise FFN, each with residual + LayerNorm, mean-pooled
over the 100 cycles for the signal embedding. Fixed capacity: d_model=32,
num_heads=4, ffn_dim=64, n_blocks=1, temporal kernel_size=5.

Protocol: EPOCHS=1000, EVAL_EVERY=50, PATIENCE=350, AdamW(lr=1e-3) default decay,
DIFF_BASE=9 + clean_feature, same top3 scalar / split / feature cache as Stage 0-2.
"""
import os
import sys
import time
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
STAGE0_PKL = os.path.join(CACHE_DIR, "stage0_smallcnn_scalar_branch_5seed.pkl")
OUT_PKL = os.path.join(CACHE_DIR, "stage3_convtransformer_screen_seed0.pkl")

DIFF_BASE = 9
EPOCHS = 1000
EVAL_EVERY = 50
PATIENCE = 350
TOP3_FEATURES = ["qdlin_diff_std", "voltage_slope_50_90", "voltage_soc_90"]
D_MODEL = 32
NUM_HEADS = 4
FFN_DIM = 64
TEMPORAL_KERNEL = 5

RMSE_TOLERANCE = 1.10
RMSE_STRONG_IMPROVEMENT = 0.95
BIAS_IMPROVEMENT_CYCLES = 15.0

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


class LearnableCyclePositionalEncoding(nn.Module):
    def __init__(self, num_cycles=100, d_model=32):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, num_cycles, d_model))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    def forward(self, x):
        assert x.ndim == 3 and x.shape[1:] == self.pos.shape[1:]
        return x + self.pos


class PerCycleConvTransformerScalarBranch(nn.Module):
    """use_conv/use_attn toggle the causal-conv and self-attention sub-blocks;
    FFN + residual/LayerNorm always applied (per spec diagram)."""

    def __init__(self, n_scalar, use_conv: bool, use_attn: bool):
        super().__init__()
        self.use_conv, self.use_attn = use_conv, use_attn
        self.encoder = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=7, padding=3), nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.pe = LearnableCyclePositionalEncoding(num_cycles=100, d_model=D_MODEL)
        if use_conv:
            self.tconv = nn.Conv1d(D_MODEL, D_MODEL, kernel_size=TEMPORAL_KERNEL, padding=0)
            self.ln_conv = nn.LayerNorm(D_MODEL)
        if use_attn:
            self.mha = nn.MultiheadAttention(D_MODEL, NUM_HEADS, batch_first=True)
            self.ln_attn = nn.LayerNorm(D_MODEL)
        self.ffn = nn.Sequential(nn.Linear(D_MODEL, FFN_DIM), nn.ReLU(), nn.Linear(FFN_DIM, D_MODEL))
        self.ln_ffn = nn.LayerNorm(D_MODEL)
        self.scalar = nn.Sequential(
            nn.Linear(n_scalar, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
        )
        self.head = nn.Linear(D_MODEL + 16, 1)
        self.last_attn_map = None

    def forward(self, x, scalar):
        B = x.shape[0]
        x_cycles = x.permute(0, 2, 1, 3).reshape(B * 100, 6, 1000)
        enc = self.encoder(x_cycles)  # [B*100, 32, 250]
        cycle_embedding = F.adaptive_avg_pool1d(enc, 1).squeeze(-1)  # [B*100, 32]
        z = cycle_embedding.reshape(B, 100, D_MODEL)
        z = self.pe(z)

        if self.use_conv:
            zt = z.transpose(1, 2)  # [B,32,100]
            zt_padded = F.pad(zt, (TEMPORAL_KERNEL - 1, 0))  # causal left-pad
            tconv_out = self.tconv(zt_padded).transpose(1, 2)  # [B,100,32]
            z = self.ln_conv(z + tconv_out)

        if self.use_attn:
            causal_mask = torch.triu(torch.ones(100, 100, device=z.device, dtype=torch.bool), diagonal=1)
            attn_out, attn_map = self.mha(z, z, z, attn_mask=causal_mask, need_weights=True, average_attn_weights=True)
            self.last_attn_map = attn_map.detach()
            z = self.ln_attn(z + attn_out)

        ffn_out = self.ffn(z)
        z = self.ln_ffn(z + ffn_out)

        signal_embedding = z.mean(dim=1)
        scalar_embedding = self.scalar(scalar)
        out = self.head(torch.cat([signal_embedding, scalar_embedding], dim=1))
        return out.squeeze(1)


VARIANTS = {
    "V1": dict(use_conv=True, use_attn=False, note="Causal Conv1d k=5 only"),
    "V2": dict(use_conv=False, use_attn=True, note="4-head self-attention only"),
    "V3": dict(use_conv=True, use_attn=True, note="Causal Conv1d + self-attention"),
}


def passes_primary_gate(rmse_new, short_bias_new, long_bias_new,
                         rmse_base, short_bias_base, long_bias_base):
    short_gain = short_bias_base - short_bias_new
    long_gain = abs(long_bias_base) - abs(long_bias_new)
    rmse_safe = rmse_new <= rmse_base * RMSE_TOLERANCE
    rmse_strong = rmse_new <= rmse_base * RMSE_STRONG_IMPROVEMENT
    bias_gain = (short_gain >= BIAS_IMPROVEMENT_CYCLES) or (long_gain >= BIAS_IMPROVEMENT_CYCLES)
    return rmse_strong or (rmse_safe and bias_gain)


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


def run_one_variant(variant_id, use_conv, use_attn, seed=0, patience=PATIENCE):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

    t0 = time.time()
    db = DataBundle(train_base_feature, train_base_label, test_feature, test_label,
                     feature_transformation=None,
                     label_transformation=DATA_TRANSFORMATIONS.build(
                         {"name": "SequentialDataTransformation", "transformations": [
                             {"name": "LogScaleDataTransformation"}, {"name": "ZScoreDataTransformation"}]}))
    model = PerCycleConvTransformerScalarBranch(n_scalar=tr_s.shape[1], use_conv=use_conv, use_attn=use_attn).to(device)
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
            print(f"  [{variant_id}] [{epoch+1:4d}/{EPOCHS}] loss={loss.item():.4f} "
                  f"val_RMSE={val_rmse:7.2f}  test_RMSE={test_rmse_demo:7.2f}"
                  f"{'  <- best val' if improved else ''}", flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
            if (epoch + 1) - best_val_epoch >= patience:
                print(f"  [{variant_id}] early stop at epoch {epoch+1} (no improvement for "
                      f"{(epoch+1)-best_val_epoch} epochs)", flush=True)
                break

    model.load_state_dict(best_state); model.eval()
    train_seconds = time.time() - t0
    metrics = compute_full_metrics(best_test_pred, test_true_np)

    del model, optimizer, feat_dev, label_dev, scalar_dev, diffed, best_state
    if device == "cuda":
        torch.cuda.empty_cache()

    return dict(
        variant_id=variant_id, use_conv=use_conv, use_attn=use_attn,
        best_epoch=best_val_epoch, best_val_metric=best_val_rmse,
        val_pred=best_val_pred, test_pred=best_test_pred,
        val_true=val_true, test_true=test_true_np,
        train_seconds=train_seconds, n_parameters=n_params,
        diff_base=DIFF_BASE, d_model=D_MODEL, num_heads=NUM_HEADS, n_blocks=1,
        temporal_conv=use_conv,
        **metrics,
    )


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

    with open(STAGE0_PKL, "rb") as f:
        stage0 = pickle.load(f)
    base_rmse = stage0["aggregate"]["rmse_mean"]
    base_short_bias = stage0["aggregate"]["short_bias_mean"]
    base_long_bias = stage0["aggregate"]["long_bias_mean"]
    print(f"Stage 0 baseline (5-seed): rmse_mean={base_rmse:.2f}  "
          f"short_bias_mean={base_short_bias:+.2f}  long_bias_mean={base_long_bias:+.2f}")

    results = {}
    if os.path.exists(OUT_PKL):
        with open(OUT_PKL, "rb") as f:
            results = pickle.load(f)
        print(f"resuming: {list(results.keys())} already done")

    for vid, cfg in VARIANTS.items():
        if vid in results:
            print(f"{vid}: already done, rmse={results[vid]['rmse']:.2f} -- skipping")
            continue
        print(f"\n=== Stage 3 screen: {vid} ({cfg['note']}) seed=0, patience={PATIENCE} ===")
        res = run_one_variant(vid, cfg["use_conv"], cfg["use_attn"], seed=0)
        res["screen_pass"] = passes_primary_gate(
            res["rmse"], res["short_bias"], res["long_bias"],
            base_rmse, base_short_bias, base_long_bias)
        results[vid] = res
        with open(OUT_PKL, "wb") as f:
            pickle.dump(results, f)
        print(f"{vid}: test_RMSE={res['rmse']:.2f}  MAE={res['mae']:.2f}  R2={res['r2']:.3f}  "
              f"short_bias={res['short_bias']:+.2f}  long_bias={res['long_bias']:+.2f}  "
              f"best_epoch={res['best_epoch']}  screen_pass={res['screen_pass']}")

    print(f"\n=== Stage 3 screen summary (vs Stage 0 baseline rmse_mean={base_rmse:.2f}) ===")
    best_vid, best_rmse = None, float("inf")
    for vid, cfg in VARIANTS.items():
        r = results[vid]
        print(f"{vid:4s} conv={str(r['use_conv']):5s} attn={str(r['use_attn']):5s} rmse={r['rmse']:7.2f}  "
              f"mae={r['mae']:7.2f}  r2={r['r2']:.3f}  short_bias={r['short_bias']:+8.2f}  "
              f"long_bias={r['long_bias']:+8.2f}  pass={r['screen_pass']}")
        if r["screen_pass"] and r["rmse"] < best_rmse:
            best_vid, best_rmse = vid, r["rmse"]
    if best_vid:
        print(f"\nBest screen-passing candidate: {best_vid} (rmse={best_rmse:.2f})")
    else:
        print(f"\nNo variant passed the screen gate vs Stage 0 baseline ({base_rmse:.2f}).")
