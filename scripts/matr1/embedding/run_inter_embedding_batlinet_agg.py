"""BatLiNet-inspired reference-aggregation ablation, on top of the ORIGINAL
(best-known) Inter-Embedding delta model: raw/unstandardized V2 embeddings,
plain 2-layer MLP (Linear(48,64)->ReLU->Linear(64,1)), no antisymmetric
wrapper -- matching run_inter_embedding_stage1.py's recipe exactly, since
that is the only configuration in this whole search that has beaten V2
solo (68.41 vs 75.32).

Only the AGGREGATION step is varied here, inspired directly by BatLiNet's
own design (BatLiNet/ARCHITECTURE.md):
  - K (reference set size): 8 (current), 20, 40, 91 (all train cells)
  - Aggregation: mean (current), median (BatLiNet's actual eval-time
    choice -- robust to outlier references), inverse-distance-weighted
    (non-learned weighting by embedding L2 distance -- a fixed scheme,
    not a learned attention head, matching BatLiNet's philosophy of
    pre-declared/non-learned combination rules on tiny data)

The delta model itself is trained ONCE per K (K only changes which cells
are candidates, not the pair-sampling training procedure, so one trained
delta model can be reused across all aggregation methods and all K choices
via reference subsetting) -- see main() for the exact reuse structure.
"""
import pickle
import numpy as np
import torch
import torch.nn as nn

CACHE_DIR = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML\ipynb\exp_v3\matr1_feature_cache"
EMB_PKL = f"{CACHE_DIR}\\v2_embeddings_seed0.pkl"
OUT_PKL = f"{CACHE_DIR}\\inter_embedding_batlinet_agg.pkl"

MARGIN_M = 0.1
REF_SEED = 0
ANCHORS_PER_EPOCH = 32
TARGETS_PER_ANCHOR = 4
EPOCHS = 1000
PATIENCE = 200
EVAL_EVERY = 25
HIDDEN = 64

device = "cuda" if torch.cuda.is_available() else "cpu"


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


class InterEmbeddingModel(nn.Module):
    """Exact match to run_inter_embedding_stage1.py's delta model."""

    def __init__(self, dim=48, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, delta_h):
        return self.net(delta_h).squeeze(-1)


with open(EMB_PKL, "rb") as f:
    E = pickle.load(f)
h_train = E["h_train"]; h_val = E["h_val"]; h_test = E["h_test"]  # RAW, unstandardized
train_label = E["train_label"]; val_label = E["val_label"]; test_label = E["test_label"]
v2_test_rmse = E["v2_test_rmse"]

log_train = np.log(train_label)
mu, sigma = log_train.mean(), log_train.std()
z_train = (log_train - mu) / sigma
n_train = len(z_train)


def build_reference_set(z, K, seed=REF_SEED):
    n = len(z)
    if K >= n:
        return np.arange(n)  # all train cells
    order = np.argsort(z)
    short_pool = order[: n // 3]; mid_pool = order[n // 3: 2 * n // 3]; long_pool = order[2 * n // 3:]
    rng = np.random.RandomState(seed)
    # split K proportionally across the 3 buckets (same 3/8, 3/8, 2/8 ratio as the original 3/3/2-of-8 rule)
    k_short = max(1, round(K * 3 / 8)); k_long = max(1, round(K * 2 / 8))
    k_mid = max(1, K - k_short - k_long)
    k_short = min(k_short, len(short_pool)); k_mid = min(k_mid, len(mid_pool)); k_long = min(k_long, len(long_pool))
    refs = list(rng.choice(short_pool, size=k_short, replace=False)) + \
           list(rng.choice(mid_pool, size=k_mid, replace=False)) + \
           list(rng.choice(long_pool, size=k_long, replace=False))
    return np.array(refs)


REF_IDX_8 = build_reference_set(z_train, 8)  # matches the original fixed set exactly (3/3/2)
print(f"K=8 reference set: {REF_IDX_8.tolist()} (should match original run_inter_embedding_stage1.py)")

h_train_t = torch.tensor(h_train, dtype=torch.float32, device=device)
z_train_t = torch.tensor(z_train, dtype=torch.float32, device=device)


def train_delta_model(seed=0):
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

    # use the K=8 reference set for checkpoint selection (matches original protocol)
    def infer_val_k8():
        h_refs = h_train[REF_IDX_8]; z_refs = z_train[REF_IDX_8]
        preds = []
        with torch.no_grad():
            for h_r, z_r in zip(h_refs, z_refs):
                dh = torch.tensor(h_val - h_r[None, :], dtype=torch.float32, device=device)
                dz_pred = model(dh).cpu().numpy()
                preds.append(z_r + dz_pred)
        z_final = np.mean(np.stack(preds, axis=0), axis=0)
        return np.exp(z_final * sigma + mu)

    best_val_rmse, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(EPOCHS):
        model.train()
        pi, pj = sample_pairs()
        dh = h_train_t[pi] - h_train_t[pj]
        dz_true = z_train_t[pi] - z_train_t[pj]
        pred = model(dh)
        loss = ((pred - dz_true) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % EVAL_EVERY == 0 or (epoch + 1) == EPOCHS:
            model.eval()
            val_rmse = rmse(infer_val_k8(), val_label)
            improved = val_rmse < best_val_rmse
            if improved:
                best_val_rmse, best_epoch = val_rmse, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) - best_epoch >= PATIENCE:
                break

    model.load_state_dict(best_state); model.eval()
    print(f"[delta model] best_epoch={best_epoch} val_RMSE(K=8,mean)={best_val_rmse:.2f} "
          f"(cf. original Stage-1: val=84.90 test=68.41)")
    return model


def predict_all_deltas(model, h_target):
    """Delta_z(target, r) for every train cell r at once -- reused across K/agg choices."""
    preds = np.zeros((n_train, len(h_target)))
    with torch.no_grad():
        for r in range(n_train):
            dh = torch.tensor(h_target - h_train[r][None, :], dtype=torch.float32, device=device)
            preds[r] = model(dh).cpu().numpy()
    return preds  # [n_train, n_target], Delta_z(target_i, ref_r)


def aggregate(dz_all_refs, ref_idx, method, h_target=None, h_train_refs=None):
    """dz_all_refs: [n_train, n_target] Delta_z predictions vs every train cell.
    ref_idx: which train cells to use as the reference set for this K."""
    z_refs = z_train[ref_idx]  # [K]
    z_candidates = z_refs[:, None] + dz_all_refs[ref_idx]  # [K, n_target], each ref's implied z estimate

    if method == "mean":
        z_final = z_candidates.mean(axis=0)
    elif method == "median":
        z_final = np.median(z_candidates, axis=0)
    elif method == "invdist":
        # non-learned inverse-distance weighting on RAW embedding L2 distance
        d = np.linalg.norm(h_train[ref_idx][:, None, :] - h_target[None, :, :], axis=-1)  # [K, n_target]
        w = 1.0 / (d + 1e-2)
        w = w / w.sum(axis=0, keepdims=True)
        z_final = (w * z_candidates).sum(axis=0)
    else:
        raise ValueError(method)
    return np.exp(z_final * sigma + mu)


if __name__ == "__main__":
    print(f"cf. V2 solo test_RMSE={v2_test_rmse:.2f}")
    print(f"cf. original Stage-1 (K=8, mean): val=84.90 test=68.41\n")

    model = train_delta_model(seed=0)

    dz_val_all = predict_all_deltas(model, h_val)   # [91, 41]
    dz_test_all = predict_all_deltas(model, h_test)  # [91, 42]

    results = {}
    print("\n=== BatLiNet-inspired aggregation sweep (val RMSE; test logged for the record) ===")
    print(f"{'K':>4s} {'method':10s} {'val_RMSE':>9s} {'test_RMSE':>10s}")
    for K in [8, 20, 40, 91]:
        ref_idx = build_reference_set(z_train, K)
        for method in ["mean", "median", "invdist"]:
            val_pred = aggregate(dz_val_all, ref_idx, method, h_target=h_val)
            test_pred = aggregate(dz_test_all, ref_idx, method, h_target=h_test)
            val_r = rmse(val_pred, val_label)
            test_r = rmse(test_pred, test_label)
            results[(K, method)] = dict(val_rmse=val_r, test_rmse=test_r)
            print(f"{K:4d} {method:10s} {val_r:9.2f} {test_r:10.2f}")

    best_key = min(results.keys(), key=lambda k: results[k]["val_rmse"])
    print(f"\nBest by VAL: K={best_key[0]} method={best_key[1]}  "
          f"val={results[best_key]['val_rmse']:.2f} test={results[best_key]['test_rmse']:.2f}")
    print(f"cf. original (K=8, mean): val=84.90 test=68.41")
    print(f"cf. V2 solo test={v2_test_rmse:.2f}")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(dict(results={f"K{k[0]}_{k[1]}": v for k, v in results.items()},
                          best_key=f"K{best_key[0]}_{best_key[1]}", v2_test_rmse=v2_test_rmse), f)
    print(f"\nSaved to {OUT_PKL}")
