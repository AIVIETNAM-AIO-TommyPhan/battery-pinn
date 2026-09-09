"""Re-run the original PINN tiers (0.5, 1, 2, 3) on V2 (axis-aware attention, no PE) --
the Stage 1 winner this session -- instead of the original SmallCNNScalarBranch they were
tried on in day_0825/0827 (that track closed negative on the old backbone).

PROCESS NOTE (added after review): this is a SCREENING run, run ahead of the formal
Stage 4 sign-off in day_0901/plan.md Task #6 -- per user's explicit direction this
session to run all tiers now rather than wait. day_0901/plan.md Task #7 states PINN
work is "Gated on Task 6" (winning backbone from Stages 0-3 picked); Stage 4 itself
was completed retroactively alongside this run (see plan.md/report.md) once Stage 0-3
results already in hand this session were aggregated -- V2 (Stage 1) is confirmed the
selected backbone, so this run's target is not in question, but the gate was technically
open when this script started. V2's long_bias (5-seed: -0.75+/-45.44, driven by a seed-4
outlier) is worse than Stage 0 baseline's on that one axis even though V2 wins on RMSE --
"best architecture" here means lowest 5-seed RMSE, not uniformly best on every metric.

Deliberate deviation from the checklist: loss_rul below is plain MSE (matching V2's own
training, so the comparison is apples-to-apples against the 75.32 baseline), not the
checklist's smooth_l1 for Tier 1's loss_rul -- everything else (loss_soh, loss_mono,
loss_wiener, loss_rank) is verbatim.

Tier 0 (safety-weighted loss) is intentionally excluded from TIERS below -- already
tested separately in run_v2_tier0_safety_loss.py and confirmed FAILED on V2 (sw=1.5:
RMSE 75.32->93.48; sw=2.0: 75.32->100.93, long_bias flipped to -25.86) before this
script was written.

Sanity check performed (not just architectural inspection): instantiating
V2PINN(use_soh=False, use_wiener=False) and V2AttentionOnly with the same seed produces
BIT-IDENTICAL initial weights for every shared layer (verified via state_dict comparison,
all 16 tensors match) -- confirms tier=="0.5" (or any tier with lambda=0) reduces exactly
to V2's own training loop before the tier-specific loss terms are added.

Exact loss formulas reused verbatim from report_expirement/day_0825/Wiener_PINN_Project_Checklist.md
(the only surviving spec for these tiers -- the original .py/.ipynb training code no longer
exists, only .pkl results do):

Tier 0.5 (cross-cell ranking loss, checklist L216-291):
    margin = 0.0
    diff_true = true[:,None] - true[None,:]
    diff_pred = pred[:,None] - pred[None,:]
    sign_true = sign(diff_true)
    loss_rank = mean(pair_weight * relu(margin - sign_true*diff_pred))
    pair_weight = 1.0 for short-short or long-long pairs, 0.0 for any pair touching mid-life
    loss_total = loss_rul + lambda_rank * loss_rank
    Applied here in the transformed (log+zscore) label space (same space as loss_rul) for
    consistency -- monotonic transform preserves the sign_true relationships used by the hinge.

Tier 1 (auxiliary SOH head, checklist L317-365):
    loss_soh = smooth_l1(soh_pred, soh_true)
    loss_total = loss_rul + lambda_soh * loss_soh
    KNOWN RISK (day_0827 Task #12, root-caused): any nonzero lambda_soh corrupts long-life
    extrapolation via the shared backbone (long_bias jumped -18.6 -> -41.4 the moment SOH
    loss turned on, on the OLD architecture). Run anyway per user request for completeness/
    comparison on the new architecture, but this negative prior is expected to still apply
    since V2 shares the same backbone-sharing structure between RUL and SOH heads.

Tier 2 (Tier 1 + monotonicity, checklist L369-401):
    tolerance = 0.002
    loss_mono = relu(soh_pred[:,1:] - soh_pred[:,:-1] - tolerance).mean()
    loss_total = loss_rul + lambda_soh*loss_soh + lambda_mono*loss_mono

Tier 3 (Tier 1/2 + Wiener drift/diffusion, checklist L429-496):
    D = 1 - soh_pred (soh_pred here reuses the Tier1/2 SOH head output)
    sigma = softplus(raw_sigma) + 1e-6
    delta_D = D[:,1:] - D[:,:-1]; residual = delta_D - mu[:,:-1]*delta_t  (delta_t=1)
    variance = sigma[:,:-1]**2 * delta_t + 1e-6
    loss_wiener = mean(0.5*residual**2/variance + 0.5*log(variance))
    loss_total = loss_rul + lambda_soh*loss_soh + lambda_mono*loss_mono + lambda_wiener*loss_wiener

Lambda values: single run per tier at the best value found in the OLD track (day_0827),
not re-swept here (already know the productive region): lambda_rank=0.01, lambda_soh=0.01,
lambda_mono=0.05, lambda_wiener=0.05. Seed=0 only (screen), patience=200 (V2's protocol).

soh_true reused as-is from ipynb/exp_v3/matr1_feature_cache/soh_trajectory_cache.pkl
(already computed, matching shapes: train=91x100, val=41x100, test=42x100).
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
SOH_CACHE_PKL = os.path.join(CACHE_DIR, "soh_trajectory_cache.pkl")
CONFIRM_PKL = os.path.join(CACHE_DIR, "stage1_axis_aware_confirm_5seed.pkl")
OUT_PKL = os.path.join(CACHE_DIR, "v2_pinn_tiers_seed0.pkl")

DIFF_BASE = 9
EPOCHS = 1000
EVAL_EVERY = 50
PATIENCE = 200
TOP3_FEATURES = ["qdlin_diff_std", "voltage_slope_50_90", "voltage_soc_90"]

LAMBDA_RANK = 0.01
LAMBDA_SOH = 0.01
LAMBDA_MONO = 0.05
LAMBDA_WIENER = 0.05
MONO_TOLERANCE = 0.002

TIERS = ["0.5", "1", "2", "3"]

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


class V2PINN(nn.Module):
    """V2 backbone (no PE, attention pooling) + optional SOH/drift/diffusion heads."""

    def __init__(self, n_scalar, use_soh=False, use_wiener=False, n_cycles=100):
        super().__init__()
        self.use_soh, self.use_wiener = use_soh, use_wiener
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
        self.rul_head = nn.Linear(32 + 16, 1)
        if use_soh:
            self.soh_head = nn.Linear(32 + 16, n_cycles)
        if use_wiener:
            self.mu_head = nn.Linear(32 + 16, n_cycles)
            self.sigma_head = nn.Linear(32 + 16, n_cycles)

    def forward(self, x, scalar):
        conv3_out = self.backbone(x)
        cycle_tokens = conv3_out.mean(dim=-1).transpose(1, 2)
        signal_embedding, _ = self.pool(cycle_tokens)
        scalar_embedding = self.scalar(scalar)
        z = torch.cat([signal_embedding, scalar_embedding], dim=1)
        rul = self.rul_head(z).squeeze(1)
        soh = self.soh_head(z) if self.use_soh else None
        mu = self.mu_head(z) if self.use_wiener else None
        sigma = F.softplus(self.sigma_head(z)) + 1e-6 if self.use_wiener else None
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


def make_pair_weight(true_rul_np):
    n = len(true_rul_np)
    order = np.argsort(true_rul_np)
    tercile = np.zeros(n, dtype=int)  # 0=short, 1=mid, 2=long
    tercile[order[: n // 3]] = 0
    tercile[order[n // 3: 2 * (n // 3)]] = 1
    tercile[order[2 * (n // 3):]] = 2
    w = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if tercile[i] == tercile[j] and tercile[i] in (0, 2):
                w[i, j] = 1.0
    return torch.from_numpy(w)


def run_one_tier(tier, seed=0):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

    use_soh = tier in ("1", "2", "3")
    use_wiener = tier == "3"

    t0 = time.time()
    db = DataBundle(train_base_feature, train_base_label, test_feature, test_label,
                     feature_transformation=None,
                     label_transformation=DATA_TRANSFORMATIONS.build(
                         {"name": "SequentialDataTransformation", "transformations": [
                             {"name": "LogScaleDataTransformation"}, {"name": "ZScoreDataTransformation"}]}))
    model = V2PINN(n_scalar=tr_s.shape[1], use_soh=use_soh, use_wiener=use_wiener).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    feat_dev = db.train_data.feature.to(device)
    label_dev = db.train_data.label.to(device)
    scalar_dev = tr_s.to(device)
    diffed = clean_feature(feat_dev - feat_dev[:, :, [DIFF_BASE]])
    train_soh_dev = train_soh.to(device) if use_soh else None
    pair_weight_dev = make_pair_weight(train_base_label.numpy()).to(device) if tier == "0.5" else None

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
        loss = loss_rul

        if tier == "0.5":
            diff_true = label_dev.unsqueeze(1) - label_dev.unsqueeze(0)
            diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)
            sign_true = torch.sign(diff_true)
            loss_rank = (pair_weight_dev * F.relu(-sign_true * diff_pred)).mean()
            loss = loss + LAMBDA_RANK * loss_rank

        if use_soh:
            loss_soh = F.smooth_l1_loss(soh_pred, train_soh_dev)
            loss = loss + LAMBDA_SOH * loss_soh

        if tier == "2" or tier == "3":
            loss_mono = F.relu(soh_pred[:, 1:] - soh_pred[:, :-1] - MONO_TOLERANCE).mean()
            loss = loss + LAMBDA_MONO * loss_mono

        if use_wiener:
            D = 1.0 - soh_pred
            delta_D = D[:, 1:] - D[:, :-1]
            residual = delta_D - mu[:, :-1] * 1.0
            variance = sigma[:, :-1] ** 2 * 1.0 + 1e-6
            loss_wiener = (0.5 * residual ** 2 / variance + 0.5 * torch.log(variance)).mean()
            loss = loss + LAMBDA_WIENER * loss_wiener

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
            print(f"  [tier{tier}] [{epoch+1:4d}/{EPOCHS}] loss={loss.item():.4f} "
                  f"val_RMSE={val_rmse:7.2f}  test_RMSE={test_rmse_demo:7.2f}"
                  f"{'  <- best val' if improved else ''}", flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
            if (epoch + 1) - best_val_epoch >= PATIENCE:
                print(f"  [tier{tier}] early stop at epoch {epoch+1} (no improvement for "
                      f"{(epoch+1)-best_val_epoch} epochs)", flush=True)
                break

    model.load_state_dict(best_state); model.eval()
    train_seconds = time.time() - t0
    metrics = compute_full_metrics(best_test_pred, test_true_np)

    del model, optimizer, feat_dev, label_dev, scalar_dev, diffed, best_state
    if device == "cuda":
        torch.cuda.empty_cache()

    return dict(tier=tier, best_epoch=best_val_epoch, val_pred=best_val_pred, test_pred=best_test_pred,
                val_true=val_true, test_true=test_true_np, train_seconds=train_seconds,
                n_parameters=n_params, **metrics)


if __name__ == "__main__":
    with open(FEATURE_CACHE_PKL, "rb") as f:
        _C = pickle.load(f)
    train_base_feature = _C["train_base_feature"]; train_base_label = _C["train_base_label"]
    val_feature = _C["val_feature"]; val_label = _C["val_label"]
    test_feature = _C["test_feature"]; test_label = _C["test_label"]
    train_base_scalar = _C["train_base_scalar"]; val_scalar = _C["val_scalar"]; test_scalar = _C["test_scalar"]
    ALL_FEATURE_NAMES = _C["feature_names"]

    with open(SOH_CACHE_PKL, "rb") as f:
        _S = pickle.load(f)
    train_soh = _S["train_soh"]

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

    for tier in TIERS:
        if tier in results:
            print(f"tier{tier}: already done, rmse={results[tier]['rmse']:.2f} -- skipping")
            continue
        print(f"\n=== V2 + PINN Tier {tier}, seed=0, patience={PATIENCE} ===")
        res = run_one_tier(tier, seed=0)
        results[tier] = res
        with open(OUT_PKL, "wb") as f:
            pickle.dump(results, f)
        print(f"tier{tier}: RMSE={res['rmse']:.2f}  MAE={res['mae']:.2f}  R2={res['r2']:.3f}  "
              f"short_bias={res['short_bias']:+.2f}  long_bias={res['long_bias']:+.2f}  best_epoch={res['best_epoch']}")

    print(f"\n=== Summary: V2 + PINN tiers vs plain V2 (RMSE={v2_baseline['rmse']:.2f}) ===")
    for tier in TIERS:
        r = results[tier]
        delta = r["rmse"] - v2_baseline["rmse"]
        print(f"tier{tier}: RMSE={r['rmse']:.2f}  delta={delta:+.2f}  short_bias={r['short_bias']:+.2f}  long_bias={r['long_bias']:+.2f}")
