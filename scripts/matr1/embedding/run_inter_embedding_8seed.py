"""Extend the 5-seed final pipeline (run_inter_embedding_multiseed_final.py)
to 8 seeds (0-7), matching this project's BatLiNet 8-seed convention. Seeds
0-4 are reused from inter_embedding_multiseed_final.pkl; seeds 5-7 are
trained fresh here: V2 (embedding-exposing, no PE, patience=350), V1 (with
PE, patience=350, matching the original confirm protocol), and the
Inter-Embedding 8-seed delta-model ensemble on V2's embeddings.

Resumable: saves after each seed completes.
"""
import os
import sys
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
FIVESEED_PKL = os.path.join(CACHE_DIR, "inter_embedding_multiseed_final.pkl")
OUT_PKL = os.path.join(CACHE_DIR, "inter_embedding_8seed.pkl")

DIFF_BASE = 9
EPOCHS_V = 1000
EVAL_EVERY = 50
PATIENCE_V1 = 350
PATIENCE_V2 = 350
TOP3_FEATURES = ["qdlin_diff_std", "voltage_slope_50_90", "voltage_soc_90"]

MARGIN_M = 0.1
REF_SEED = 0
ANCHORS_PER_EPOCH = 32
TARGETS_PER_ANCHOR = 4
EPOCHS_INTER = 1000
INTER_PATIENCE = 200
HIDDEN = 64
N_MODEL_SEEDS = 8
K_FINAL = 91
NEW_SEEDS = [5, 6, 7]

device = "cuda" if torch.cuda.is_available() else "cpu"


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


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


class V1PE(nn.Module):
    """V1: attention pooling + learnable cycle positional encoding.
    Matches SmallCNNScalarBranchAxisAware(use_pe=True) exactly."""

    def __init__(self, n_scalar):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.pe = LearnableCyclePositionalEncoding(num_cycles=50, d_model=32)
        self.pool = CycleAttentionPooling(d_model=32, hidden_dim=16)
        self.scalar = nn.Sequential(nn.Linear(n_scalar, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
        self.head = nn.Linear(32 + 16, 1)

    def forward(self, x, scalar):
        conv3_out = self.backbone(x)
        cycle_tokens = conv3_out.mean(dim=-1).transpose(1, 2)
        cycle_tokens = self.pe(cycle_tokens)
        signal_embedding, _ = self.pool(cycle_tokens)
        scalar_embedding = self.scalar(scalar)
        out = self.head(torch.cat([signal_embedding, scalar_embedding], dim=1))
        return out.squeeze(1)


class V2WithEmbedding(nn.Module):
    def __init__(self, n_scalar):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.pool = CycleAttentionPooling(d_model=32, hidden_dim=16)
        self.scalar = nn.Sequential(nn.Linear(n_scalar, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
        self.head = nn.Linear(32 + 16, 1)

    def forward(self, x, scalar, return_embedding=False):
        conv3_out = self.backbone(x)
        cycle_tokens = conv3_out.mean(dim=-1).transpose(1, 2)
        signal_embedding, _ = self.pool(cycle_tokens)
        scalar_embedding = self.scalar(scalar)
        h = torch.cat([signal_embedding, scalar_embedding], dim=1)
        out = self.head(h).squeeze(1)
        if return_embedding:
            return out, h
        return out


class InterEmbeddingModel(nn.Module):
    def __init__(self, dim=48, hidden=64):
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


def train_v1(seed, train_base_feature, train_base_label, test_feature, test_label,
             val_feature, val_label, tr_s, va_s, te_s):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

    db = DataBundle(train_base_feature, train_base_label, test_feature, test_label,
                     feature_transformation=None,
                     label_transformation=DATA_TRANSFORMATIONS.build(
                         {"name": "SequentialDataTransformation", "transformations": [
                             {"name": "LogScaleDataTransformation"}, {"name": "ZScoreDataTransformation"}]}))
    model = V1PE(n_scalar=tr_s.shape[1]).to(device)
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
    best_val_rmse, best_val_epoch, best_state = float("inf"), 0, None

    for epoch in range(EPOCHS_V):
        model.train(); optimizer.zero_grad()
        pred = model(diffed, scalar_dev)
        loss = ((pred - label_dev) ** 2).mean()
        loss.backward(); optimizer.step()
        if (epoch + 1) % EVAL_EVERY == 0 or (epoch + 1) == EPOCHS_V:
            model.eval()
            val_pred = predict_on(val_feature, va_s)
            val_rmse = float(np.sqrt(np.mean((val_true - val_pred) ** 2)))
            improved = val_rmse < best_val_rmse
            if improved:
                best_val_rmse, best_val_epoch = val_rmse, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  [V1 s{seed}] [{epoch+1:4d}/{EPOCHS_V}] val_RMSE={val_rmse:7.2f}"
                  f"{'  <- best val' if improved else ''}", flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
            if (epoch + 1) - best_val_epoch >= PATIENCE_V1:
                print(f"  [V1 s{seed}] early stop at epoch {epoch+1}", flush=True)
                break

    model.load_state_dict(best_state); model.eval()
    val_pred = predict_on(val_feature, va_s)
    test_pred = predict_on(test_feature, te_s)
    print(f"[s{seed}] V1 test_RMSE={rmse(test_pred, test_true_np):.2f}")
    return val_pred, test_pred


def train_v2_embed(seed, train_base_feature, train_base_label, test_feature, test_label,
                    val_feature, val_label, tr_s, va_s, te_s):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

    db = DataBundle(train_base_feature, train_base_label, test_feature, test_label,
                     feature_transformation=None,
                     label_transformation=DATA_TRANSFORMATIONS.build(
                         {"name": "SequentialDataTransformation", "transformations": [
                             {"name": "LogScaleDataTransformation"}, {"name": "ZScoreDataTransformation"}]}))
    model = V2WithEmbedding(n_scalar=tr_s.shape[1]).to(device)
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
    best_val_rmse, best_val_epoch, best_state = float("inf"), 0, None

    for epoch in range(EPOCHS_V):
        model.train(); optimizer.zero_grad()
        pred = model(diffed, scalar_dev)
        loss = ((pred - label_dev) ** 2).mean()
        loss.backward(); optimizer.step()
        if (epoch + 1) % EVAL_EVERY == 0 or (epoch + 1) == EPOCHS_V:
            model.eval()
            val_pred = predict_on(val_feature, va_s)
            test_pred_demo = predict_on(test_feature, te_s)
            val_rmse = float(np.sqrt(np.mean((val_true - val_pred) ** 2)))
            test_rmse_demo = float(np.sqrt(np.mean((test_true_np - test_pred_demo) ** 2)))
            improved = val_rmse < best_val_rmse
            if improved:
                best_val_rmse, best_val_epoch = val_rmse, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  [V2-embed s{seed}] [{epoch+1:4d}/{EPOCHS_V}] val_RMSE={val_rmse:7.2f} "
                  f"test_RMSE={test_rmse_demo:7.2f}{'  <- best val' if improved else ''}", flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
            if (epoch + 1) - best_val_epoch >= PATIENCE_V2:
                print(f"  [V2-embed s{seed}] early stop at epoch {epoch+1}", flush=True)
                break

    model.load_state_dict(best_state); model.eval()
    correct_val_pred = predict_on(val_feature, va_s)
    correct_test_pred = predict_on(test_feature, te_s)

    def extract(feature, scalar):
        feat = feature.to(device)
        diffed_e = clean_feature(feat - feat[:, :, [DIFF_BASE]])
        with torch.no_grad():
            _, h = model(diffed_e, scalar.to(device), return_embedding=True)
        return h.cpu().numpy()

    h_train = extract(train_base_feature, tr_s)
    h_val = extract(val_feature, va_s)
    h_test = extract(test_feature, te_s)
    print(f"[s{seed}] V2 test_RMSE={rmse(correct_test_pred, test_true_np):.2f}")
    return h_train, h_val, h_test, correct_val_pred, correct_test_pred, val_true, test_true_np


def run_delta_ensemble(seed_v2, h_train, h_val, h_test, train_label_np, val_label_np, test_label_np):
    log_train = np.log(train_label_np)
    mu, sigma = log_train.mean(), log_train.std()
    z_train = (log_train - mu) / sigma
    n_train = len(z_train)
    h_train_t = torch.tensor(h_train, dtype=torch.float32, device=device)
    z_train_t = torch.tensor(z_train, dtype=torch.float32, device=device)
    REF_IDX_8 = build_reference_set(z_train, 8)
    REF_IDX_FINAL = build_reference_set(z_train, K_FINAL)

    def train_one(seed):
        rng = np.random.RandomState(seed)

        def sample_pairs():
            anchors = rng.choice(n_train, size=ANCHORS_PER_EPOCH, replace=False)
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
        model = InterEmbeddingModel(dim=48, hidden=HIDDEN).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        def infer_val_k8():
            h_refs = h_train[REF_IDX_8]; z_refs = z_train[REF_IDX_8]
            preds = []
            with torch.no_grad():
                for h_r, z_r in zip(h_refs, z_refs):
                    dh = torch.tensor(h_val - h_r[None, :], dtype=torch.float32, device=device)
                    dz_pred = model(dh).cpu().numpy()
                    preds.append(z_r + dz_pred)
            return np.exp(np.mean(np.stack(preds, axis=0), axis=0) * sigma + mu)

        best_val_rmse, best_epoch, best_state = float("inf"), 0, None
        for epoch in range(EPOCHS_INTER):
            model.train()
            pi, pj = sample_pairs()
            dh = h_train_t[pi] - h_train_t[pj]
            dz_true = z_train_t[pi] - z_train_t[pj]
            pred = model(dh)
            loss = ((pred - dz_true) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if (epoch + 1) % EVAL_EVERY == 0 or (epoch + 1) == EPOCHS_INTER:
                model.eval()
                val_rmse = rmse(infer_val_k8(), val_label_np)
                improved = val_rmse < best_val_rmse
                if improved:
                    best_val_rmse, best_epoch = val_rmse, epoch + 1
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if (epoch + 1) - best_epoch >= INTER_PATIENCE:
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

    models = [train_one(s) for s in range(N_MODEL_SEEDS)]
    val_ens = np.mean([predict_median(m, h_val, REF_IDX_FINAL) for m in models], axis=0)
    test_ens = np.mean([predict_median(m, h_test, REF_IDX_FINAL) for m in models], axis=0)
    print(f"[s{seed_v2}] Inter-Embedding (8-seed, K=91, median): "
          f"val={rmse(val_ens,val_label_np):.2f} test={rmse(test_ens,test_label_np):.2f}")
    return val_ens, test_ens


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

    train_label_np = train_base_label.numpy().ravel()
    val_label_np = val_label.numpy().ravel()
    test_label_np = test_label.numpy().ravel()

    all_results = {}
    if os.path.exists(OUT_PKL):
        with open(OUT_PKL, "rb") as f:
            all_results = pickle.load(f)
        print(f"resuming: seeds {list(all_results.keys())} already done")

    if not set([0, 1, 2, 3, 4]).issubset(all_results.keys()) and os.path.exists(FIVESEED_PKL):
        with open(FIVESEED_PKL, "rb") as f:
            five = pickle.load(f)
        for s in [0, 1, 2, 3, 4]:
            if s not in all_results:
                all_results[s] = five[s]
        with open(OUT_PKL, "wb") as f:
            pickle.dump(all_results, f)
        print("seeds 0-4 reused from inter_embedding_multiseed_final.pkl")

    for seed in NEW_SEEDS:
        if seed in all_results:
            print(f"seed{seed}: already done -- skipping")
            continue
        print(f"\n=== 8-seed extension: seed={seed} ===")
        v1_val_pred, v1_test_pred = train_v1(
            seed, train_base_feature, train_base_label, test_feature, test_label,
            val_feature, val_label, tr_s, va_s, te_s)
        h_train, h_val, h_test, v2_val_pred, v2_test_pred, val_true, test_true = train_v2_embed(
            seed, train_base_feature, train_base_label, test_feature, test_label,
            val_feature, val_label, tr_s, va_s, te_s)
        inter_val_pred, inter_test_pred = run_delta_ensemble(
            seed, h_train, h_val, h_test, train_label_np, val_label_np, test_label_np)

        all_results[seed] = dict(
            v2_val_pred=v2_val_pred, v2_test_pred=v2_test_pred,
            v1_val_pred=v1_val_pred, v1_test_pred=v1_test_pred,
            inter_val_pred=inter_val_pred, inter_test_pred=inter_test_pred,
            val_true=val_true, test_true=test_true, source="fresh_8seed_extension",
        )
        with open(OUT_PKL, "wb") as f:
            pickle.dump(all_results, f)

    print("\n=== 8-seed summary ===")
    from scipy.optimize import nnls

    def affine_fit(val_p, val_t, test_p):
        A = np.vstack([val_p, np.ones_like(val_p)]).T
        a, b = np.linalg.lstsq(A, val_t, rcond=None)[0]
        return a * val_p + b, a * test_p + b

    SEEDS_ALL = [0, 1, 2, 3, 4, 5, 6, 7]
    v2_t, v1_t, inter_t, v2v1_t, aff3_t = [], [], [], [], []
    for seed in SEEDS_ALL:
        r = all_results[seed]
        val_true, test_true = r["val_true"], r["test_true"]
        v2_val, v2_test = r["v2_val_pred"], r["v2_test_pred"]
        v1_val, v1_test = r["v1_val_pred"], r["v1_test_pred"]
        inter_val, inter_test = r["inter_val_pred"], r["inter_test_pred"]

        v2_t.append(rmse(v2_test, test_true)); v1_t.append(rmse(v1_test, test_true)); inter_t.append(rmse(inter_test, test_true))

        V2v1 = np.stack([v2_val, v1_val], axis=1)
        w, _ = nnls(V2v1, val_true)
        v2v1_t.append(rmse(np.stack([v2_test, v1_test], axis=1) @ w, test_true))

        v2c_val, v2c_test = affine_fit(v2_val, val_true, v2_test)
        v1c_val, v1c_test = affine_fit(v1_val, val_true, v1_test)
        interc_val, interc_test = affine_fit(inter_val, val_true, inter_test)
        V3 = np.stack([v2c_val, v1c_val, interc_val], axis=1)
        w3, _ = nnls(V3, val_true)
        aff3_t.append(rmse(np.stack([v2c_test, v1c_test, interc_test], axis=1) @ w3, test_true))

    def ms(x):
        return f"{np.mean(x):.2f} +- {np.std(x):.2f}  per-seed={[round(v,2) for v in x]}"

    print("V2 solo:     ", ms(v2_t))
    print("V1 solo:     ", ms(v1_t))
    print("Inter-final: ", ms(inter_t))
    print("V2+V1:       ", ms(v2v1_t))
    print("Affine-3way: ", ms(aff3_t))
    print(f"\nSaved to {OUT_PKL}")
