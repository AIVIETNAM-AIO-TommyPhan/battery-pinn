"""Stage 1 -- axis-aware pooling head, 5-seed confirmation for V1 and V2.

Both V1 (PE+attention, full axis-aware head) and V2 (no PE, attention fixed) passed
the seed=0 screen gate vs Stage 0 baseline (rmse_mean=101.18). User approved
confirming BOTH (not just the single best-RMSE screen candidate) because V1 has a
materially better bias profile (short_bias +40.75, long_bias -3.82) than V2
(short_bias +55.37, long_bias +31.32, worse long_bias than the Stage 0 baseline
itself) despite V2's lower raw RMSE (75.32 vs 81.77) -- see
report_expirement/day_0901/report.md for the screen writeup.

Seed=0 is reused from stage1_axis_aware_screen_seed0.pkl (already trained,
deterministic, same code path) instead of retraining. Seeds 1-4 trained fresh here,
sequentially (one job at a time, 4GB GPU), for both variants -- 8 runs total.

Protocol: EPOCHS=1000, EVAL_EVERY=50, PATIENCE=350, AdamW(lr=1e-3) default decay,
DIFF_BASE=9 + clean_feature, same top3 scalar features / split / feature cache as
Stage 0/1 screen.
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
STAGE0_PKL = os.path.join(CACHE_DIR, "stage0_smallcnn_scalar_branch_5seed.pkl")
SCREEN_PKL = os.path.join(CACHE_DIR, "stage1_axis_aware_screen_seed0.pkl")
OUT_PKL = os.path.join(CACHE_DIR, "stage1_axis_aware_confirm_5seed.pkl")

DIFF_BASE = 9
EPOCHS = 1000
EVAL_EVERY = 50
PATIENCE = 350
TOP3_FEATURES = ["qdlin_diff_std", "voltage_slope_50_90", "voltage_soc_90"]

RMSE_TOLERANCE = 1.10
RMSE_STRONG_IMPROVEMENT = 0.95
BIAS_IMPROVEMENT_CYCLES = 15.0

CONFIRM_VARIANTS = {
    "V1": dict(use_pe=True, aggregation="attention"),
    "V2": dict(use_pe=False, aggregation="attention"),
}
CONFIRM_SEEDS = [1, 2, 3, 4]  # seed 0 reused from screen pkl

# PATIENCE deviation for V2, user-directed speed tradeoff (2026-09-01): V1 confirmed
# at the pinned patience=350; V2 seeds 1-4 use patience=200 instead to go faster.
# Documented explicitly here and in report.md -- not a silent protocol change.
PATIENCE_BY_VARIANT = {"V1": 350, "V2": 200}

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
    def __init__(self, num_cycles=50, d_model=32):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, num_cycles, d_model))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    def forward(self, x):
        assert x.ndim == 3 and x.shape[1:] == self.pos.shape[1:]
        return x + self.pos


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


class SmallCNNScalarBranchAxisAware(nn.Module):
    def __init__(self, n_scalar, use_pe: bool, aggregation: str):
        super().__init__()
        assert aggregation == "attention"
        self.aggregation = aggregation
        self.backbone = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.pe = LearnableCyclePositionalEncoding(num_cycles=50, d_model=32) if use_pe else None
        self.pool = CycleAttentionPooling(d_model=32, hidden_dim=16)
        self.scalar = nn.Sequential(
            nn.Linear(n_scalar, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
        )
        self.head = nn.Linear(32 + 16, 1)
        self.last_attn = None  # captured on every forward; caller snapshots at best-val checkpoints

    def forward(self, x, scalar):
        conv3_out = self.backbone(x)
        assert conv3_out.shape[1] == 32 and conv3_out.shape[2] == 50
        cycle_tokens = conv3_out.mean(dim=-1).transpose(1, 2)
        if self.pe is not None:
            cycle_tokens = self.pe(cycle_tokens)
        signal_embedding, attn = self.pool(cycle_tokens)
        self.last_attn = attn.detach()
        scalar_embedding = self.scalar(scalar)
        out = self.head(torch.cat([signal_embedding, scalar_embedding], dim=1))
        return out.squeeze(1)


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


def run_one_seed(variant_id, use_pe, seed, tag, patience):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

    t0 = time.time()
    db = DataBundle(train_base_feature, train_base_label, test_feature, test_label,
                     feature_transformation=None,
                     label_transformation=DATA_TRANSFORMATIONS.build(
                         {"name": "SequentialDataTransformation", "transformations": [
                             {"name": "LogScaleDataTransformation"}, {"name": "ZScoreDataTransformation"}]}))
    model = SmallCNNScalarBranchAxisAware(n_scalar=tr_s.shape[1], use_pe=use_pe, aggregation="attention").to(device)
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
    best_val_pred, best_test_pred, best_attn = None, None, None

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
                best_attn = model.last_attn.cpu().numpy() if model.last_attn is not None else None
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
    train_seconds = time.time() - t0
    metrics = compute_full_metrics(best_test_pred, test_true_np)

    del model, optimizer, feat_dev, label_dev, scalar_dev, diffed, best_state
    if device == "cuda":
        torch.cuda.empty_cache()

    return dict(
        best_epoch=best_val_epoch, best_val_metric=best_val_rmse,
        val_pred=best_val_pred, test_pred=best_test_pred,
        val_true=val_true, test_true=test_true_np,
        train_seconds=train_seconds, n_parameters=n_params,
        attention_weights=best_attn,
        source="fresh_run_this_session",
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

    with open(SCREEN_PKL, "rb") as f:
        screen = pickle.load(f)

    all_results = {}
    if os.path.exists(OUT_PKL):
        with open(OUT_PKL, "rb") as f:
            all_results = pickle.load(f)
        print(f"resuming: {list(all_results.keys())} already have entries")

    for vid, cfg in CONFIRM_VARIANTS.items():
        if vid not in all_results:
            all_results[vid] = {}
        # seed 0 reused from screen
        if 0 not in all_results[vid]:
            s0 = screen[vid]
            all_results[vid][0] = dict(
                best_epoch=s0["best_epoch"], best_val_metric=None,
                val_pred=s0["val_pred"], test_pred=s0["test_pred"],
                val_true=s0["val_true"], test_true=s0["test_true"],
                train_seconds=s0["train_seconds"], n_parameters=s0["n_parameters"],
                attention_weights=s0.get("attention_weights"),
                source="reused_stage1_screen_seed0",
                rmse=s0["rmse"], mae=s0["mae"], mape=s0["mape"], r2=s0["r2"],
                short_bias=s0["short_bias"], long_bias=s0["long_bias"],
            )
            with open(OUT_PKL, "wb") as f:
                pickle.dump(all_results, f)
        elif "attention_weights" not in all_results[vid][0]:
            # backfill: earlier run of this script didn't carry attention_weights over
            all_results[vid][0]["attention_weights"] = screen[vid].get("attention_weights")
            with open(OUT_PKL, "wb") as f:
                pickle.dump(all_results, f)

        for seed in CONFIRM_SEEDS:
            if seed in all_results[vid]:
                print(f"{vid} seed{seed}: already done, rmse={all_results[vid][seed]['rmse']:.2f} -- skipping")
                continue
            tag = f"{vid}_s{seed}"
            patience = PATIENCE_BY_VARIANT[vid]
            print(f"\n=== Stage 1 confirm: {vid} seed={seed}, patience={patience} ===")
            res = run_one_seed(vid, cfg["use_pe"], seed, tag, patience)
            all_results[vid][seed] = res
            with open(OUT_PKL, "wb") as f:
                pickle.dump(all_results, f)
            print(f"{vid} seed{seed}: test_RMSE={res['rmse']:.2f}  MAE={res['mae']:.2f}  R2={res['r2']:.3f}  "
                  f"short_bias={res['short_bias']:+.2f}  long_bias={res['long_bias']:+.2f}  "
                  f"best_epoch={res['best_epoch']}")

    print(f"\n=== Stage 1 confirm summary (vs Stage 0 baseline rmse_mean={base_rmse:.2f}) ===")
    for vid in CONFIRM_VARIANTS:
        rmses = np.array([all_results[vid][s]["rmse"] for s in range(5)])
        maes = np.array([all_results[vid][s]["mae"] for s in range(5)])
        r2s = np.array([all_results[vid][s]["r2"] for s in range(5)])
        short_biases = np.array([all_results[vid][s]["short_bias"] for s in range(5)])
        long_biases = np.array([all_results[vid][s]["long_bias"] for s in range(5)])
        confirm_pass = passes_primary_gate(
            float(rmses.mean()), float(short_biases.mean()), float(long_biases.mean()),
            base_rmse, base_short_bias, base_long_bias)
        unstable = bool(rmses.std() > 0.5 * rmses.mean())  # flag high-variance candidates
        print(f"\n{vid}: per-seed RMSE = {np.round(rmses,2).tolist()}")
        print(f"  RMSE {rmses.mean():.2f} +/- {rmses.std():.2f}   MAE {maes.mean():.2f} +/- {maes.std():.2f}   "
              f"R2 {r2s.mean():.3f} +/- {r2s.std():.3f}")
        print(f"  short_bias {short_biases.mean():+.2f} +/- {short_biases.std():.2f}   "
              f"long_bias {long_biases.mean():+.2f} +/- {long_biases.std():.2f}")
        print(f"  confirm_pass={confirm_pass}  unstable={unstable}  "
              f"stage1_promote={confirm_pass and not unstable}")

    print(f"\nSaved -> {OUT_PKL}")
