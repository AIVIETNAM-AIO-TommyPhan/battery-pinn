"""Multi-seed confirmation of the Inter-Embedding Model AS A STANDALONE
MODEL (not as a V2 ensemble partner -- Stage 1 seed=0 failed the ensemble
decorrelation gate, corr=0.923, because it is built from V2's own
embeddings and inherits most of V2's error structure).

Seed 0's result (already computed, inter_embedding_stage1_seed0.pkl):
  V2 solo test=75.32, Inter-Embedding solo test=68.41 (9.2% better, single
  seed -- not yet a claim per project convention, hence this confirmation).

This script reproduces the same pipeline for seeds 0-4: retrain V2-embed,
sanity-gate check, extract embeddings, train Inter-Embedding Model, report
solo RMSE (val+test) per seed. Resumable: saves after each seed completes.
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
OUT_PKL = os.path.join(CACHE_DIR, "inter_embedding_multiseed.pkl")
SEED0_PKL = os.path.join(CACHE_DIR, "inter_embedding_stage1_seed0.pkl")

DIFF_BASE = 9
EPOCHS_V2 = 1000
EVAL_EVERY = 50
PATIENCE_V2 = 350
TOP3_FEATURES = ["qdlin_diff_std", "voltage_slope_50_90", "voltage_soc_90"]

MARGIN_M = 0.1
ANCHORS_PER_EPOCH = 32
TARGETS_PER_ANCHOR = 4
EPOCHS_INTER = 1000
INTER_PATIENCE = 200
REF_SEED = 0
K_NN = 8
SEEDS = [0, 1, 2, 3, 4]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}")


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
    non_smooth_left = diff_left > diff_left.std(-1, keepdim=True) * threshold
    non_smooth_right = diff_right > diff_right.std(-1, keepdim=True) * threshold
    for _ in range(width):
        non_smooth_left = non_smooth_left | torch.roll(non_smooth_left, shifts=1, dims=-1)
        non_smooth_right = non_smooth_right | torch.roll(non_smooth_right, shifts=-1, dims=-1)
    to_smooth = non_smooth_left & non_smooth_right
    x = x.clone(); x[to_smooth] = 0.
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
        self.score = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        logits = self.score(x)
        weights = torch.softmax(logits, dim=1)
        pooled = (weights * x).sum(dim=1)
        return pooled, weights.squeeze(-1)


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
        self.last_attn = None

    def forward(self, x, scalar, return_embedding=False):
        conv3_out = self.backbone(x)
        cycle_tokens = conv3_out.mean(dim=-1).transpose(1, 2)
        signal_embedding, attn = self.pool(cycle_tokens)
        self.last_attn = attn.detach()
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


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


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
    best_val_rmse, best_val_epoch, best_state = float("inf"), None, None
    best_val_pred, best_test_pred = None, None

    for epoch in range(EPOCHS_V2):
        model.train(); optimizer.zero_grad()
        pred = model(diffed, scalar_dev)
        loss = ((pred - label_dev) ** 2).mean()
        loss.backward(); optimizer.step()
        if (epoch + 1) % EVAL_EVERY == 0 or (epoch + 1) == EPOCHS_V2:
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
            print(f"  [V2-embed s{seed}] [{epoch+1:4d}/{EPOCHS_V2}] val_RMSE={val_rmse:7.2f} "
                  f"test_RMSE={test_rmse_demo:7.2f}{'  <- best val' if improved else ''}", flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
            if (epoch + 1) - best_val_epoch >= PATIENCE_V2:
                print(f"  [V2-embed s{seed}] early stop at epoch {epoch+1}", flush=True)
                break

    model.load_state_dict(best_state); model.eval()
    return model, db, best_val_pred, best_test_pred, val_true, test_true_np


def extract_embeddings(model, feature, scalar):
    feat = feature.to(device)
    diffed_e = clean_feature(feat - feat[:, :, [DIFF_BASE]])
    with torch.no_grad():
        _, h = model(diffed_e, scalar.to(device), return_embedding=True)
    return h.cpu().numpy()


def fixed_references(z, seed=REF_SEED):
    n = len(z)
    order = np.argsort(z)
    short_pool = order[: n // 3]; mid_pool = order[n // 3: 2 * n // 3]; long_pool = order[2 * n // 3:]
    rng = np.random.RandomState(seed)
    refs = list(rng.choice(short_pool, size=3, replace=False)) + \
           list(rng.choice(mid_pool, size=3, replace=False)) + \
           list(rng.choice(long_pool, size=2, replace=False))
    return np.array(refs)


def run_one_seed(seed, train_base_feature, train_base_label, test_feature, test_label,
                  val_feature, val_label, tr_s, va_s, te_s, train_label_np, val_label_np, test_label_np):
    t0 = time.time()
    model, db, v2_val_pred, v2_test_pred, val_true, test_true = train_v2_embed(
        seed, train_base_feature, train_base_label, test_feature, test_label,
        val_feature, val_label, tr_s, va_s, te_s)
    v2_test_rmse = rmse(v2_test_pred, test_true)
    print(f"[s{seed}] V2-embed test_RMSE={v2_test_rmse:.2f}")

    h_train = extract_embeddings(model, train_base_feature, tr_s)
    h_val = extract_embeddings(model, val_feature, va_s)
    h_test = extract_embeddings(model, test_feature, te_s)

    log_train = np.log(train_label_np)
    mu, sigma = log_train.mean(), log_train.std()
    z_train = (np.log(train_label_np) - mu) / sigma
    n_train = len(z_train)

    ref_idx = fixed_references(z_train)
    rng = np.random.RandomState(seed)  # pair-sampling RNG varies by seed (independent training run)

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
    inter_model = InterEmbeddingModel(dim=48, hidden=64).to(device)
    inter_opt = torch.optim.Adam(inter_model.parameters(), lr=1e-3)
    h_train_t = torch.tensor(h_train, dtype=torch.float32, device=device)
    z_train_t = torch.tensor(z_train, dtype=torch.float32, device=device)

    def infer_inter(h_target, model_):
        h_refs = h_train[ref_idx]; z_refs = z_train[ref_idx]
        preds = []
        with torch.no_grad():
            for h_r, z_r in zip(h_refs, z_refs):
                dh = torch.tensor(h_target - h_r[None, :], dtype=torch.float32, device=device)
                dz_pred = model_(dh).cpu().numpy()
                preds.append(z_r + dz_pred)
        z_inter = np.mean(np.stack(preds, axis=0), axis=0)
        return np.exp(z_inter * sigma + mu)

    best_val_inter_rmse, best_epoch, best_state = float("inf"), None, None
    for epoch in range(EPOCHS_INTER):
        inter_model.train()
        pi, pj = sample_pairs()
        dh = h_train_t[pi] - h_train_t[pj]
        dz_true = z_train_t[pi] - z_train_t[pj]
        pred = inter_model(dh)
        loss = ((pred - dz_true) ** 2).mean()
        inter_opt.zero_grad(); loss.backward(); inter_opt.step()
        if (epoch + 1) % EVAL_EVERY == 0 or (epoch + 1) == EPOCHS_INTER:
            inter_model.eval()
            val_pred_rul = infer_inter(h_val, inter_model)
            val_inter_rmse = rmse(val_pred_rul, val_true)
            improved = val_inter_rmse < best_val_inter_rmse
            if improved:
                best_val_inter_rmse, best_epoch = val_inter_rmse, epoch + 1
                best_state = {k: v.detach().clone() for k, v in inter_model.state_dict().items()}
            print(f"  [inter s{seed}] [{epoch+1:4d}/{EPOCHS_INTER}] RMSE_inter_val={val_inter_rmse:7.2f}"
                  f"{'  <- best val' if improved else ''}", flush=True)
            if (epoch + 1) - best_epoch >= INTER_PATIENCE:
                print(f"  [inter s{seed}] early stop at epoch {epoch+1}", flush=True)
                break

    inter_model.load_state_dict(best_state); inter_model.eval()
    inter_val_pred = infer_inter(h_val, inter_model)
    inter_test_pred = infer_inter(h_test, inter_model)
    inter_val_rmse = rmse(inter_val_pred, val_true)
    inter_test_rmse = rmse(inter_test_pred, test_true)
    corr_test = float(np.corrcoef(v2_test_pred - test_true, inter_test_pred - test_true)[0, 1])

    print(f"[s{seed}] DONE: V2={v2_test_rmse:.2f} Inter={inter_test_rmse:.2f} "
          f"beats_V2={inter_test_rmse < v2_test_rmse} corr={corr_test:.3f}")

    del model, inter_model
    if device == "cuda":
        torch.cuda.empty_cache()

    return dict(v2_val_pred=v2_val_pred, v2_test_pred=v2_test_pred, v2_test_rmse=v2_test_rmse,
                inter_val_pred=inter_val_pred, inter_test_pred=inter_test_pred,
                inter_val_rmse=inter_val_rmse, inter_test_rmse=inter_test_rmse,
                val_true=val_true, test_true=test_true, corr_test=corr_test, ref_idx=ref_idx.tolist())


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

    # reuse seed 0 from the already-completed single-seed run
    if 0 not in all_results and os.path.exists(SEED0_PKL):
        with open(SEED0_PKL, "rb") as f:
            s0 = pickle.load(f)
        all_results[0] = dict(
            v2_val_pred=s0["v2_val_pred"], v2_test_pred=s0["v2_test_pred"], v2_test_rmse=s0["v2_test_rmse"],
            inter_val_pred=s0["inter_val_pred"], inter_test_pred=s0["inter_test_pred"],
            inter_val_rmse=s0["inter_val_rmse"], inter_test_rmse=s0["inter_test_rmse"],
            val_true=s0["val_true"], test_true=s0["test_true"], corr_test=s0["corr_test"],
            ref_idx=s0["ref_idx"], source="reused_stage1_seed0",
        )
        with open(OUT_PKL, "wb") as f:
            pickle.dump(all_results, f)
        print("seed 0 reused from inter_embedding_stage1_seed0.pkl")

    for seed in SEEDS:
        if seed in all_results:
            print(f"seed{seed}: already done, V2={all_results[seed]['v2_test_rmse']:.2f} "
                  f"Inter={all_results[seed]['inter_test_rmse']:.2f} -- skipping")
            continue
        print(f"\n=== Inter-Embedding multiseed: seed={seed} ===")
        res = run_one_seed(seed, train_base_feature, train_base_label, test_feature, test_label,
                            val_feature, val_label, tr_s, va_s, te_s,
                            train_label_np, val_label_np, test_label_np)
        all_results[seed] = res
        with open(OUT_PKL, "wb") as f:
            pickle.dump(all_results, f)

    v2_rmses = np.array([all_results[s]["v2_test_rmse"] for s in SEEDS])
    inter_rmses = np.array([all_results[s]["inter_test_rmse"] for s in SEEDS])
    print("\n=== 5-seed summary ===")
    print(f"V2 solo:    mean={v2_rmses.mean():.2f} std={v2_rmses.std():.2f}  per-seed={[round(x,2) for x in v2_rmses]}")
    print(f"Inter solo: mean={inter_rmses.mean():.2f} std={inter_rmses.std():.2f}  per-seed={[round(x,2) for x in inter_rmses]}")
    print(f"Inter beats V2 in {(inter_rmses < v2_rmses).sum()}/5 seeds")
