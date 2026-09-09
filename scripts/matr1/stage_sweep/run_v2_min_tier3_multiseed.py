"""5-seed confirmation for the best ensemble found this session: V2-architecture
`min` (qdlin_diff_min single-feature scalar branch) + V2-architecture Tier3 (Wiener),
combined via a FROZEN NNLS weight (fit once on seed=0's val, w_min=0.186,
w_tier3=0.794 -- reused as-is for every seed, not refit per seed).

Rationale for frozen weight (not per-seed NNLS): day_0827 Task #14 found dynamic
per-seed NNLS weight search improves RAW substantially but that gain does not
survive calibration ("double-dips" with calibration correcting the same bias) --
frozen weight + calibration was the more robust headline pattern in that track.

Calibration policy: "no default method" (day_0827 standing rule) -- for each seed,
pick whichever of {raw, isotonic, affine} has the lowest metric on THAT seed's own
val set (never chosen by peeking at test), report that as the seed's number.

Trains seeds 1-4 for both `min` and `tier3` (seed=0 already done, reused from
v2_single_feature_test.pkl and v2_pinn_tiers_seed0.pkl). Protocol matches each
model's already-established one: min uses patience=200 (V2 protocol), tier3 uses
patience=200 + the exact Tier3 Wiener loss from run_v2_pinn_tiers.py (lambda_soh=0.01,
lambda_mono=0.05, lambda_wiener=0.05, all pre-declared, not re-tuned per seed).
"""
import os
import sys
import time
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
sys.path.insert(0, REPO)

from batteryml.data.databundle import DataBundle
from batteryml.builders import DATA_TRANSFORMATIONS
import BatLiNet  # noqa: F401

CACHE_DIR = os.path.join(REPO, "ipynb", "exp_v3", "matr1_feature_cache")
FEATURE_CACHE_PKL = os.path.join(CACHE_DIR, "feature_cache.pkl")
SOH_CACHE_PKL = os.path.join(CACHE_DIR, "soh_trajectory_cache.pkl")
SINGLE_FEAT_PKL = os.path.join(CACHE_DIR, "v2_single_feature_test.pkl")
PINN_TIERS_PKL = os.path.join(CACHE_DIR, "v2_pinn_tiers_seed0.pkl")
OUT_PKL = os.path.join(CACHE_DIR, "v2_min_tier3_multiseed.pkl")

DIFF_BASE = 9
EPOCHS = 1000
EVAL_EVERY = 50
PATIENCE = 200
LAMBDA_SOH = 0.01
LAMBDA_MONO = 0.05
LAMBDA_WIENER = 0.05
MONO_TOLERANCE = 0.002
SEEDS_TO_RUN = [1, 2, 3, 4]
W_MIN, W_TIER3 = 0.186, 0.794  # frozen, from seed0 NNLS fit

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
        signal_embedding, _ = self.pool(cycle_tokens)
        scalar_embedding = self.scalar(scalar)
        out = self.head(torch.cat([signal_embedding, scalar_embedding], dim=1))
        return out.squeeze(1)


class V2PINN(nn.Module):
    def __init__(self, n_scalar, use_soh=True, use_wiener=True, n_cycles=100):
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


def rmse(t, p):
    return float(np.sqrt(np.mean((p - t) ** 2)))


def affine_calibrate(vp, vt, tp):
    A = np.vstack([vp, np.ones_like(vp)]).T
    a, b = np.linalg.lstsq(A, vt, rcond=None)[0]
    return a * tp + b


def train_min(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

    t0 = time.time()
    db = DataBundle(train_base_feature, train_base_label, test_feature, test_label,
                     feature_transformation=None,
                     label_transformation=DATA_TRANSFORMATIONS.build(
                         {"name": "SequentialDataTransformation", "transformations": [
                             {"name": "LogScaleDataTransformation"}, {"name": "ZScoreDataTransformation"}]}))
    model = V2AttentionOnly(n_scalar=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    feat_dev = db.train_data.feature.to(device)
    label_dev = db.train_data.label.to(device)
    scalar_dev = min_tr_s.to(device)
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
            val_pred = predict_on(val_feature, min_va_s)
            test_pred_demo = predict_on(test_feature, min_te_s)
            val_rmse = float(np.sqrt(np.mean((val_true - val_pred) ** 2)))
            test_rmse_demo = float(np.sqrt(np.mean((test_true_np - test_pred_demo) ** 2)))
            improved = val_rmse < best_val_rmse
            if improved:
                best_val_rmse, best_val_epoch = val_rmse, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_val_pred, best_test_pred = val_pred.copy(), test_pred_demo.copy()
            print(f"  [min_s{seed}] [{epoch+1:4d}/{EPOCHS}] val_RMSE={val_rmse:7.2f}  "
                  f"test_RMSE={test_rmse_demo:7.2f}"
                  f"{'  <- best val' if improved else ''}", flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
            if (epoch + 1) - best_val_epoch >= PATIENCE:
                print(f"  [min_s{seed}] early stop at epoch {epoch+1}", flush=True)
                break

    model.load_state_dict(best_state); model.eval()
    del model, optimizer, feat_dev, label_dev, scalar_dev, diffed, best_state
    if device == "cuda":
        torch.cuda.empty_cache()
    return dict(val_pred=best_val_pred, test_pred=best_test_pred, val_true=val_true,
                test_true=test_true_np, best_epoch=best_val_epoch, train_seconds=time.time() - t0)


def train_tier3(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

    t0 = time.time()
    db = DataBundle(train_base_feature, train_base_label, test_feature, test_label,
                     feature_transformation=None,
                     label_transformation=DATA_TRANSFORMATIONS.build(
                         {"name": "SequentialDataTransformation", "transformations": [
                             {"name": "LogScaleDataTransformation"}, {"name": "ZScoreDataTransformation"}]}))
    model = V2PINN(n_scalar=tr_s.shape[1], use_soh=True, use_wiener=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    feat_dev = db.train_data.feature.to(device)
    label_dev = db.train_data.label.to(device)
    scalar_dev = tr_s.to(device)
    diffed = clean_feature(feat_dev - feat_dev[:, :, [DIFF_BASE]])
    train_soh_dev = train_soh.to(device)

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
        loss = ((pred - label_dev) ** 2).mean()
        loss_soh = F.smooth_l1_loss(soh_pred, train_soh_dev)
        loss = loss + LAMBDA_SOH * loss_soh
        loss_mono = F.relu(soh_pred[:, 1:] - soh_pred[:, :-1] - MONO_TOLERANCE).mean()
        loss = loss + LAMBDA_MONO * loss_mono
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
            print(f"  [tier3_s{seed}] [{epoch+1:4d}/{EPOCHS}] loss={loss.item():.4f} "
                  f"val_RMSE={val_rmse:7.2f}  test_RMSE={test_rmse_demo:7.2f}"
                  f"{'  <- best val' if improved else ''}", flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
            if (epoch + 1) - best_val_epoch >= PATIENCE:
                print(f"  [tier3_s{seed}] early stop at epoch {epoch+1}", flush=True)
                break

    model.load_state_dict(best_state); model.eval()
    del model, optimizer, feat_dev, label_dev, scalar_dev, diffed, best_state
    if device == "cuda":
        torch.cuda.empty_cache()
    return dict(val_pred=best_val_pred, test_pred=best_test_pred, val_true=val_true,
                test_true=test_true_np, best_epoch=best_val_epoch, train_seconds=time.time() - t0)


def best_of_three(val_pred, val_true, test_pred, test_true):
    """Pick whichever of {raw, isotonic, affine} has the lowest VAL rmse (never test)."""
    iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(val_pred, val_true)
    val_iso_rmse = rmse(val_true, iso.predict(val_pred))
    aff_val_pred = affine_calibrate(val_pred, val_true, val_pred)
    val_aff_rmse = rmse(val_true, aff_val_pred)
    val_raw_rmse = rmse(val_true, val_pred)
    candidates = {"raw": val_raw_rmse, "isotonic": val_iso_rmse, "affine": val_aff_rmse}
    best_method = min(candidates, key=candidates.get)
    if best_method == "raw":
        test_final = test_pred
    elif best_method == "isotonic":
        test_final = iso.predict(test_pred)
    else:
        test_final = affine_calibrate(val_pred, val_true, test_pred)
    return best_method, test_final


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

    top3_idxs = [ALL_FEATURE_NAMES.index(w) for w in ["qdlin_diff_std", "voltage_slope_50_90", "voltage_soc_90"]]
    tr_s = train_base_scalar[:, top3_idxs]
    va_s = val_scalar[:, top3_idxs]
    te_s = test_scalar[:, top3_idxs]

    min_idx = ALL_FEATURE_NAMES.index("qdlin_diff_min")
    min_tr_s = train_base_scalar[:, [min_idx]]
    min_va_s = val_scalar[:, [min_idx]]
    min_te_s = test_scalar[:, [min_idx]]

    results = {}
    if os.path.exists(OUT_PKL):
        with open(OUT_PKL, "rb") as f:
            results = pickle.load(f)
        print(f"resuming: {list(results.keys())} already done")

    # seed 0: reuse from existing screens
    if 0 not in results:
        with open(SINGLE_FEAT_PKL, "rb") as f:
            sf = pickle.load(f)
        with open(PINN_TIERS_PKL, "rb") as f:
            pt = pickle.load(f)
        results[0] = dict(min=sf["min"], tier3=pt["3"])
        with open(OUT_PKL, "wb") as f:
            pickle.dump(results, f)

    for seed in SEEDS_TO_RUN:
        if seed in results:
            print(f"seed{seed}: already done -- skipping")
            continue
        print(f"\n=== seed={seed}: training min ===")
        min_res = train_min(seed)
        print(f"\n=== seed={seed}: training tier3 ===")
        tier3_res = train_tier3(seed)
        results[seed] = dict(min=min_res, tier3=tier3_res)
        with open(OUT_PKL, "wb") as f:
            pickle.dump(results, f)
        print(f"seed{seed} done: min_test_rmse={rmse(min_res['test_true'], min_res['test_pred']):.2f}  "
              f"tier3_test_rmse={rmse(tier3_res['test_true'], tier3_res['test_pred']):.2f}")

    print(f"\n=== Ensemble (frozen w_min={W_MIN}, w_tier3={W_TIER3}) + best-of-3 calibration per seed ===")
    rmses = []
    for seed in range(5):
        min_r, t3_r = results[seed]["min"], results[seed]["tier3"]
        val_true, test_true = min_r["val_true"], min_r["test_true"]
        ens_val = W_MIN * min_r["val_pred"] + W_TIER3 * t3_r["val_pred"]
        ens_test = W_MIN * min_r["test_pred"] + W_TIER3 * t3_r["test_pred"]
        method, final_test = best_of_three(ens_val, val_true, ens_test, test_true)
        r = rmse(test_true, final_test)
        rmses.append(r)
        m = compute_full_metrics(final_test, test_true)
        print(f"seed{seed}: method={method:9s} rmse={r:7.2f}  short_bias={m['short_bias']:+.2f}  long_bias={m['long_bias']:+.2f}")

    rmses = np.array(rmses)
    print(f"\n5-seed ensemble (min+tier3, best-of-3 calib): mean={rmses.mean():.2f}  std={rmses.std():.2f}")
    print(f"(cf. V2 alone 5-seed: raw=82.85+/-11.43, affine=74.10+/-9.46)")
