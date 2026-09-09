"""MATR1 -- V2 + V1 + Inter-Embedding + Affine/NNLS ensemble, full Tier3
variant: SOH auxiliary head + monotonicity + Wiener drift/diffusion (SDE),
ported to the CURRENT axis-aware backbone for BOTH V1 (use_pe=True) and V2
(use_pe=False) -- historical Tier3 (81.69+/-4.88, 5-seed) only ever used V2 on
the OLD V2PINN class; this closes the V1 + Inter-Embedding gap.

Exact loss formulas reused verbatim from run_v2_pinn_tiers.py / the Wiener_PINN
checklist: lambda_soh=0.01, lambda_mono=0.05, lambda_wiener=0.05,
mono_tolerance=0.002, patience=200 (matches this session's SOH-head convention
for both V1 and V2). RUL loss is plain MSE (matches V2's own training,
apples-to-apples against the no-Tier3 baseline).

Wiener-process NLL: delta_D ~ N(mu, sigma^2), delta_t = 1 cycle (dropped from
the formulas since delta_t=1 is a no-op).

CLI: python run_matr1_v1v2inter_tier3_ensemble.py [seed]
"""
import os
import sys
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import nnls

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
sys.path.insert(0, REPO)

from batteryml.data.databundle import DataBundle
from batteryml.builders import DATA_TRANSFORMATIONS
import BatLiNet  # noqa: F401

CACHE_DIR = os.path.join(REPO, "ipynb", "exp_v3", "matr1_feature_cache")
FEATURE_CACHE_PKL = os.path.join(CACHE_DIR, "feature_cache.pkl")
SOH_CACHE_PKL = os.path.join(CACHE_DIR, "soh_trajectory_cache.pkl")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
OUT_PKL = os.path.join(CACHE_DIR, f"matr1_v1v2inter_tier3_ensemble_seed{SEED}.pkl")

DIFF_BASE = 9
EPOCHS = 1000
EVAL_EVERY = 50
PATIENCE_BY_VARIANT = {"V1": 350, "V2": 200}  # user-directed: try longer patience for V1 (2026-09-06)
TOP3_FEATURES = ["qdlin_diff_std", "voltage_slope_50_90", "voltage_soc_90"]

LAMBDA_SOH = 0.01
LAMBDA_MONO = 0.05
LAMBDA_WIENER = 0.05
MONO_TOLERANCE = 0.002

# Inter-Embedding config (mirrors run_inter_embedding_final.py, MATR1 work)
MARGIN_M = 0.1
REF_SEED = 0
ANCHORS_PER_EPOCH = 32
TARGETS_PER_ANCHOR = 4
IE_EPOCHS = 1000
IE_PATIENCE = 200
IE_EVAL_EVERY = 25
IE_HIDDEN = 64
N_DELTA_SEEDS = 8

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}  diff_base={DIFF_BASE}  seed={SEED}  tier3(V1+V2+Inter)", flush=True)


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


class LearnableCyclePositionalEncoding(nn.Module):
    def __init__(self, num_cycles=50, d_model=32):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, num_cycles, d_model))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    def forward(self, x):
        return x + self.pos


class CycleAttentionPooling(nn.Module):
    def __init__(self, d_model=32, hidden_dim=16):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        logits = self.score(x)
        weights = torch.softmax(logits, dim=1)
        pooled = (weights * x).sum(dim=1)
        return pooled, weights.squeeze(-1)


class SmallCNNScalarBranchAxisAwareTier3(nn.Module):
    def __init__(self, n_scalar, use_pe: bool, n_cycles_soh=100):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.pe = LearnableCyclePositionalEncoding(num_cycles=50, d_model=32) if use_pe else None
        self.pool = CycleAttentionPooling(d_model=32, hidden_dim=16)
        self.scalar = nn.Sequential(nn.Linear(n_scalar, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
        self.head = nn.Linear(32 + 16, 1)
        self.soh_head = nn.Linear(32 + 16, n_cycles_soh)
        self.mu_head = nn.Linear(32 + 16, n_cycles_soh)
        self.sigma_head = nn.Linear(32 + 16, n_cycles_soh)

    def forward(self, x, scalar, return_embedding=False, return_tier3=False):
        conv3_out = self.backbone(x)
        cycle_tokens = conv3_out.mean(dim=-1).transpose(1, 2)
        if self.pe is not None:
            cycle_tokens = self.pe(cycle_tokens)
        signal_embedding, attn = self.pool(cycle_tokens)
        scalar_embedding = self.scalar(scalar)
        h = torch.cat([signal_embedding, scalar_embedding], dim=1)
        out = self.head(h).squeeze(1)
        if return_tier3:
            soh = self.soh_head(h)
            mu = self.mu_head(h)
            sigma = F.softplus(self.sigma_head(h)) + 1e-6
            if return_embedding:
                return out, h, soh, mu, sigma
            return out, soh, mu, sigma
        if return_embedding:
            return out, h
        return out


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def train_cnn(use_pe, tag):
    """Trains V1 (use_pe=True) or V2 (use_pe=False), full-batch (91 train cells
    fit on GPU without chunking), full Tier3 loss (SOH + mono + Wiener)."""
    import random
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

    db = DataBundle(train_base_feature, train_base_label, test_feature, test_label,
                     feature_transformation=None,
                     label_transformation=DATA_TRANSFORMATIONS.build(
                         {"name": "SequentialDataTransformation", "transformations": [
                             {"name": "LogScaleDataTransformation"}, {"name": "ZScoreDataTransformation"}]}))
    model = SmallCNNScalarBranchAxisAwareTier3(n_scalar=tr_s.shape[1], use_pe=use_pe).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    feat_dev = db.train_data.feature.to(device)
    label_dev = db.train_data.label.to(device)
    scalar_dev = tr_s.to(device)
    soh_dev = train_base_soh.to(device)
    diffed = clean_feature(feat_dev - feat_dev[:, :, [DIFF_BASE]])

    def predict_on(feature, scalar, return_h=False):
        feat = feature.to(device)
        diffed_e = clean_feature(feat - feat[:, :, [DIFF_BASE]])
        with torch.no_grad():
            out = model(diffed_e, scalar.to(device), return_embedding=return_h)
        if return_h:
            pred, h = out
            h = h.cpu().numpy()
        else:
            pred, h = out, None
        pred_np = db.label_transformation.inverse_transform(pred.cpu()).numpy().astype(float).ravel()
        if return_h:
            return pred_np, h
        return pred_np

    val_true = val_label.numpy().ravel()
    test_true_np = test_label.numpy().ravel()
    best_val_rmse, best_val_epoch, best_state = float("inf"), None, None
    patience = PATIENCE_BY_VARIANT[tag]

    for epoch in range(EPOCHS):
        model.train(); optimizer.zero_grad()
        pred, soh_pred, mu, sigma = model(diffed, scalar_dev, return_tier3=True)
        loss_rul = ((pred - label_dev) ** 2).mean()
        loss_soh = F.smooth_l1_loss(soh_pred, soh_dev)
        loss_mono = F.relu(soh_pred[:, 1:] - soh_pred[:, :-1] - MONO_TOLERANCE).mean()
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
            val_rmse = rmse(val_pred, val_true)
            test_rmse = rmse(test_pred, test_true_np)
            improved = val_rmse < best_val_rmse
            if improved:
                best_val_rmse, best_val_epoch = val_rmse, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  [{tag}] [{epoch+1:4d}/{EPOCHS}] loss={loss.item():.4f} "
                  f"(rul={loss_rul.item():.4f} soh={loss_soh.item():.4f} "
                  f"mono={loss_mono.item():.4f} wiener={loss_wiener.item():.4f}) "
                  f"val_RMSE={val_rmse:7.2f}  test_RMSE={test_rmse:7.2f}"
                  f"{'  <- best val' if improved else ''}", flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
            if (epoch + 1) - best_val_epoch >= patience:
                print(f"  [{tag}] early stop at epoch {epoch+1}", flush=True)
                break

    model.load_state_dict(best_state); model.eval()
    val_pred = predict_on(val_feature, va_s)
    test_pred = predict_on(test_feature, te_s)
    h_train = h_val = h_test = None
    if not use_pe:  # only V2 feeds the Inter-Embedding model
        _, h_train = predict_on(train_base_feature, tr_s, return_h=True)
        _, h_val = predict_on(val_feature, va_s, return_h=True)
        _, h_test = predict_on(test_feature, te_s, return_h=True)
    print(f"[{tag}] FINAL test_RMSE={rmse(test_pred, test_true_np):.2f}\n", flush=True)

    del model, optimizer, feat_dev, label_dev, scalar_dev, soh_dev, diffed, best_state
    if device == "cuda":
        torch.cuda.empty_cache()
    return val_pred, test_pred, h_train, h_val, h_test


class InterEmbeddingModel(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, delta_h):
        return self.net(delta_h).squeeze(-1)


def build_reference_set(z, K, seed=REF_SEED):
    n = len(z)
    if K >= n:
        return np.arange(n)
    order = np.argsort(z)
    short_pool = order[: n // 3]; mid_pool = order[n // 3: 2 * n // 3]; long_pool = order[2 * n // 3:]
    rng = np.random.RandomState(seed)
    k_short = max(1, round(K * 3 / 8)); k_long = max(1, round(K * 2 / 8))
    k_mid = max(1, K - k_short - k_long)
    k_short = min(k_short, len(short_pool)); k_mid = min(k_mid, len(mid_pool)); k_long = min(k_long, len(long_pool))
    refs = list(rng.choice(short_pool, size=k_short, replace=False)) + \
           list(rng.choice(mid_pool, size=k_mid, replace=False)) + \
           list(rng.choice(long_pool, size=k_long, replace=False))
    return np.array(refs)


def run_inter_embedding(h_train, h_val, h_test, train_label_np, val_label_np, test_label_np):
    log_train = np.log(train_label_np)
    mu, sigma = log_train.mean(), log_train.std()
    z_train = (log_train - mu) / sigma
    n_train = len(z_train)
    dim = h_train.shape[1]
    K_final = n_train

    ref_idx_8 = build_reference_set(z_train, min(8, n_train))
    ref_idx_final = build_reference_set(z_train, K_final)
    h_train_t = torch.tensor(h_train, dtype=torch.float32, device=device)
    z_train_t = torch.tensor(z_train, dtype=torch.float32, device=device)

    def infer_k8(model, h_target):
        h_refs = h_train[ref_idx_8]; z_refs = z_train[ref_idx_8]
        preds = []
        with torch.no_grad():
            for h_r, z_r in zip(h_refs, z_refs):
                dh = torch.tensor(h_target - h_r[None, :], dtype=torch.float32, device=device)
                dz_pred = model(dh).cpu().numpy()
                preds.append(z_r + dz_pred)
        return np.exp(np.mean(np.stack(preds, axis=0), axis=0) * sigma + mu)

    def train_one(seed):
        rng = np.random.RandomState(seed)

        def sample_pairs():
            anchors = rng.choice(n_train, size=min(ANCHORS_PER_EPOCH, n_train), replace=False)
            pi, pj = [], []
            for a in anchors:
                cands = [j for j in range(n_train) if j != a and abs(z_train[a] - z_train[j]) > MARGIN_M]
                if not cands:
                    continue
                chosen = rng.choice(cands, size=min(TARGETS_PER_ANCHOR, len(cands)), replace=False)
                for j in chosen:
                    pi.append(a); pj.append(j)
            return np.array(pi), np.array(pj)

        torch.manual_seed(seed)
        model = InterEmbeddingModel(dim=dim, hidden=IE_HIDDEN).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        best_val_rmse, best_epoch, best_state = float("inf"), 0, None
        for epoch in range(IE_EPOCHS):
            model.train()
            pi, pj = sample_pairs()
            dh = h_train_t[pi] - h_train_t[pj]
            dz_true = z_train_t[pi] - z_train_t[pj]
            pred = model(dh)
            loss = ((pred - dz_true) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if (epoch + 1) % IE_EVAL_EVERY == 0 or (epoch + 1) == IE_EPOCHS:
                model.eval()
                val_rmse = rmse(infer_k8(model, h_val), val_label_np)
                if val_rmse < best_val_rmse:
                    best_val_rmse, best_epoch = val_rmse, epoch + 1
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if (epoch + 1) - best_epoch >= IE_PATIENCE:
                    break
        model.load_state_dict(best_state); model.eval()
        return model

    def predict_median(model, h_target, ref_idx):
        h_refs = h_train[ref_idx]; z_refs = z_train[ref_idx]
        preds = []
        with torch.no_grad():
            for h_r, z_r in zip(h_refs, z_refs):
                dh = torch.tensor(h_target - h_r[None, :], dtype=torch.float32, device=device)
                dz_pred = model(dh).cpu().numpy()
                preds.append(z_r + dz_pred)
        z_final = np.median(np.stack(preds, axis=0), axis=0)
        return np.exp(z_final * sigma + mu)

    print(f"Training {N_DELTA_SEEDS} Inter-Embedding delta-model seeds (K_final={K_final})...", flush=True)
    models = [train_one(s) for s in range(N_DELTA_SEEDS)]
    val_preds = np.stack([predict_median(m, h_val, ref_idx_final) for m in models], axis=0)
    test_preds = np.stack([predict_median(m, h_test, ref_idx_final) for m in models], axis=0)
    return val_preds.mean(axis=0), test_preds.mean(axis=0)


def affine_fit(val_p, val_t, test_p):
    A = np.vstack([val_p, np.ones_like(val_p)]).T
    a, b = np.linalg.lstsq(A, val_t, rcond=None)[0]
    return a * val_p + b, a * test_p + b


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

    val_true = val_label.numpy().ravel()
    test_true = test_label.numpy().ravel()
    train_true = train_base_label.numpy().ravel()

    print("=== Training V2 (use_pe=False), Tier3 ===", flush=True)
    v2_val, v2_test, h_train, h_val, h_test = train_cnn(use_pe=False, tag="V2")

    print("=== Training V1 (use_pe=True), Tier3 ===", flush=True)
    v1_val, v1_test, _, _, _ = train_cnn(use_pe=True, tag="V1")

    print("=== Training Inter-Embedding (on V2 Tier3 embeddings) ===", flush=True)
    inter_val, inter_test = run_inter_embedding(h_train, h_val, h_test, train_true, val_true, test_true)
    print(f"Inter-Embedding: val_RMSE={rmse(inter_val, val_true):.2f}  test_RMSE={rmse(inter_test, test_true):.2f}\n")

    print("=== Affine-calibrate each component on val, then NNLS-ensemble ===", flush=True)
    v2c_val, v2c_test = affine_fit(v2_val, val_true, v2_test)
    v1c_val, v1c_test = affine_fit(v1_val, val_true, v1_test)
    interc_val, interc_test = affine_fit(inter_val, val_true, inter_test)

    V3v = np.stack([v2c_val, v1c_val, interc_val], axis=1)
    w3, _ = nnls(V3v, val_true)
    V3t = np.stack([v2c_test, v1c_test, interc_test], axis=1)
    ens_test = V3t @ w3
    ens_val = V3v @ w3

    print(f"\n=== RESULTS (seed={SEED}, Tier3) ===")
    print(f"V2 solo:            val={rmse(v2_val,val_true):.2f}  test={rmse(v2_test,test_true):.2f}")
    print(f"V1 solo:            val={rmse(v1_val,val_true):.2f}  test={rmse(v1_test,test_true):.2f}")
    print(f"Inter-Embedding:    val={rmse(inter_val,val_true):.2f}  test={rmse(inter_test,test_true):.2f}")
    print(f"Affine-3way NNLS:   val={rmse(ens_val,val_true):.2f}  test={rmse(ens_test,test_true):.2f}  weights={w3.round(3)}")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(dict(
            v2_val=v2_val, v2_test=v2_test, v1_val=v1_val, v1_test=v1_test,
            inter_val=inter_val, inter_test=inter_test,
            ens_val=ens_val, ens_test=ens_test, nnls_weights=w3,
            val_true=val_true, test_true=test_true,
        ), f)
    print(f"\nSaved -> {OUT_PKL}")
