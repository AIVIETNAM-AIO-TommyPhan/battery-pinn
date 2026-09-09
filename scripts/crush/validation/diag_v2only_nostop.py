"""CRUSH -- V2 + V1 + Inter-Embedding + Affine/NNLS ensemble, full Tier3
variant: SOH auxiliary head + monotonicity + Wiener drift/diffusion (SDE) on
top of the already-validated asymmetric RUL loss (UNDER_PENALTY=1.4).
Mirrors run_matr1_v1v2inter_tier3_ensemble.py / run_hust_v1v2inter_tier3_ensemble.py.

Exact loss formulas reused verbatim: lambda_soh=0.01, lambda_mono=0.05,
lambda_wiener=0.05, mono_tolerance=0.002 (pre-declared, not re-tuned per
dataset). CHUNK-based mini-batch training (CHUNK=48).

CLI: python run_crush_v1v2inter_tier3_ensemble.py <cache_name> [seed] [under_penalty]
  e.g. python run_crush_v1v2inter_tier3_ensemble.py feature_cache_crush_hust10_calce15_fullval_withsoh 0 1.4
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

CACHE_DIR = os.path.join(REPO, "ipynb", "exp_v3", "crush")
CACHE_NAME = sys.argv[1] if len(sys.argv) > 1 else "feature_cache_crush_hust10_calce15_fullval_withsoh"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
UNDER_PENALTY = float(sys.argv[3]) if len(sys.argv) > 3 else 1.4
FEATURE_CACHE_PKL = os.path.join(CACHE_DIR, f"{CACHE_NAME}.pkl")
OUT_PKL = os.path.join(CACHE_DIR, f"{CACHE_NAME}_v1v2inter_tier3_pen{UNDER_PENALTY:.1f}_ensemble_seed{SEED}.pkl")

DIFF_BASE = 9
EPOCHS = 1000
EVAL_EVERY = 50
PATIENCE = 100000  # diagnostic: no early stop
CHUNK = 48
TOP3_FEATURES = ["qdlin_diff_std", "voltage_slope_50_90", "voltage_soc_90"]

LAMBDA_SOH = 0.01
LAMBDA_MONO = 0.05
LAMBDA_WIENER = 0.05
MONO_TOLERANCE = 0.002

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
print(f"device={device}  diff_base={DIFF_BASE}  chunk={CHUNK}  seed={SEED}  pen={UNDER_PENALTY}  tier3", flush=True)


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
    train_feature_cpu = db.train_data.feature
    train_label_cpu = db.train_data.label
    train_soh_cpu = train_base_soh
    n_train = train_feature_cpu.shape[0]

    def predict_on(feature, scalar, return_h=False):
        preds, hs = [], []
        for s in range(0, feature.shape[0], CHUNK):
            feat = feature[s:s+CHUNK].to(device)
            diffed_e = clean_feature(feat - feat[:, :, [DIFF_BASE]])
            with torch.no_grad():
                out = model(diffed_e, scalar[s:s+CHUNK].to(device), return_embedding=return_h)
            if return_h:
                pred, h = out
                hs.append(h.cpu())
            else:
                pred = out
            preds.append(pred.cpu())
            del feat, diffed_e
            if device == "cuda":
                torch.cuda.empty_cache()
        pred = torch.cat(preds)
        pred_np = db.label_transformation.inverse_transform(pred).numpy().astype(float).ravel()
        if return_h:
            return pred_np, torch.cat(hs).numpy()
        return pred_np

    val_true = val_label.numpy().ravel()
    test_true_np = test_label.numpy().ravel()
    best_val_rmse, best_val_epoch, best_state = float("inf"), None, None

    for epoch in range(EPOCHS):
        model.train(); optimizer.zero_grad()
        for s in range(0, n_train, CHUNK):
            feat_c = train_feature_cpu[s:s+CHUNK].to(device)
            label_c = train_label_cpu[s:s+CHUNK].to(device)
            scalar_c = tr_s[s:s+CHUNK].to(device)
            soh_c = train_soh_cpu[s:s+CHUNK].to(device)
            diffed_c = clean_feature(feat_c - feat_c[:, :, [DIFF_BASE]])
            pred_c, soh_pred_c, mu_c, sigma_c = model(diffed_c, scalar_c, return_tier3=True)
            diff_c = pred_c - label_c
            weight_c = torch.where(diff_c < 0, UNDER_PENALTY, 1.0)
            loss_rul_c = (weight_c * diff_c ** 2).mean()
            loss_soh_c = F.smooth_l1_loss(soh_pred_c, soh_c)
            loss_mono_c = F.relu(soh_pred_c[:, 1:] - soh_pred_c[:, :-1] - MONO_TOLERANCE).mean()
            D_c = 1.0 - soh_pred_c
            delta_D_c = D_c[:, 1:] - D_c[:, :-1]
            residual_c = delta_D_c - mu_c[:, :-1]
            variance_c = sigma_c[:, :-1] ** 2 + 1e-6
            loss_wiener_c = (0.5 * residual_c ** 2 / variance_c + 0.5 * torch.log(variance_c)).mean()
            chunk_loss = (loss_rul_c + LAMBDA_SOH * loss_soh_c + LAMBDA_MONO * loss_mono_c
                          + LAMBDA_WIENER * loss_wiener_c) * (feat_c.shape[0] / n_train)
            chunk_loss.backward()
            del feat_c, label_c, scalar_c, soh_c, diffed_c, pred_c, soh_pred_c, mu_c, sigma_c, chunk_loss
        optimizer.step()
        if device == "cuda":
            torch.cuda.empty_cache()
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
            print(f"  [{tag}] [{epoch+1:4d}/{EPOCHS}] val_RMSE={val_rmse:7.2f}  test_RMSE={test_rmse:7.2f}"
                  f"{'  <- best val' if improved else ''}", flush=True)
            if (epoch + 1) - best_val_epoch >= PATIENCE:
                print(f"  [{tag}] early stop at epoch {epoch+1}", flush=True)
                break

    model.load_state_dict(best_state); model.eval()
    val_pred = predict_on(val_feature, va_s)
    test_pred = predict_on(test_feature, te_s)
    h_train = h_val = h_test = None
    if not use_pe:
        _, h_train = predict_on(train_base_feature, tr_s, return_h=True)
        _, h_val = predict_on(val_feature, va_s, return_h=True)
        _, h_test = predict_on(test_feature, te_s, return_h=True)
    print(f"[{tag}] FINAL test_RMSE={rmse(test_pred, test_true_np):.2f}\n", flush=True)
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
            diff = pred - dz_true
            weight = torch.where(diff < 0, UNDER_PENALTY, 1.0)
            loss = (weight * diff ** 2).mean()
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
    train_base_soh = _C["train_soh"]
    val_feature = _C["val_feature"]; val_label = _C["val_label"]
    test_feature = _C["test_feature"]; test_label = _C["test_label"]
    train_base_scalar = _C["train_base_scalar"]; val_scalar = _C["val_scalar"]; test_scalar = _C["test_scalar"]
    ALL_FEATURE_NAMES = _C["feature_names"]
    _idxs = [ALL_FEATURE_NAMES.index(w) for w in TOP3_FEATURES]
    tr_s = train_base_scalar[:, _idxs]; va_s = val_scalar[:, _idxs]; te_s = test_scalar[:, _idxs]

    val_true = val_label.numpy().ravel()
    test_true = test_label.numpy().ravel()
    train_true = train_base_label.numpy().ravel()

    print("=== Training V2 (use_pe=False), Tier3 ===", flush=True)
    v2_val, v2_test, h_train, h_val, h_test = train_cnn(use_pe=False, tag="V2")
    import sys as _sys; _sys.exit(0)  # diagnostic: V2 only

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
