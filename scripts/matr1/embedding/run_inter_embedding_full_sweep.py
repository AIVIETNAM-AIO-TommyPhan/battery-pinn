"""Comprehensive final sweep: representation x reference-selection x K x
aggregation, all combinations, using 8-seed delta-model ensembles for both
representations. Consolidates everything tried piecemeal in this session's
final rounds into one complete table.

Axes:
  Representation: diff (48-dim), sqdiff_concat (144-dim)
  Reference selection: global (RUL-tercile stratified, one fixed set for
                        everyone), knn (per-target, nearest by embedding L2)
  K: 15, 20, 30, 40, 60, 91
  Aggregation: mean, median

96 combinations total (2x2x6x2), each using the same 8-seed ensemble
(trained once per representation, reused across all K/ref-selection/agg
combinations at inference time -- no retraining per combination).
"""
import pickle
import numpy as np
import torch
import torch.nn as nn

CACHE_DIR = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML\ipynb\exp_v3\matr1_feature_cache"
EMB_PKL = f"{CACHE_DIR}/v2_embeddings_seed0.pkl"
OUT_PKL = f"{CACHE_DIR}/inter_embedding_full_sweep.pkl"

MARGIN_M = 0.1
REF_SEED = 0
ANCHORS_PER_EPOCH = 32
TARGETS_PER_ANCHOR = 4
EPOCHS = 1000
PATIENCE = 200
EVAL_EVERY = 25
HIDDEN = 64
N_MODEL_SEEDS = 8
K_GRID = [15, 20, 30, 40, 60, 91]

device = "cuda" if torch.cuda.is_available() else "cpu"


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def combine(name, hi, hj):
    if name == "diff":
        return hi - hj
    if name == "sqdiff_concat":
        return np.concatenate([hi, hj, (hi - hj) ** 2], axis=-1)
    raise ValueError(name)


DIMS = {"diff": 48, "sqdiff_concat": 144}


class PlainMLP(nn.Module):
    def __init__(self, dim, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


with open(EMB_PKL, "rb") as f:
    E = pickle.load(f)
h_train = E["h_train"]; h_val = E["h_val"]; h_test = E["h_test"]
train_label = E["train_label"]; val_label = E["val_label"]; test_label = E["test_label"]
v2_test_rmse = E["v2_test_rmse"]

log_train = np.log(train_label)
mu, sigma = log_train.mean(), log_train.std()
z_train = (log_train - mu) / sigma
n_train = len(z_train)


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


REF_IDX_8 = build_reference_set(z_train, 8)


def train_one(name, seed):
    dim = DIMS[name]
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
    model = PlainMLP(dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    h_refs8 = h_train[REF_IDX_8]; z_refs8 = z_train[REF_IDX_8]

    def infer_val_k8():
        preds = []
        with torch.no_grad():
            for h_r, z_r in zip(h_refs8, z_refs8):
                x = combine(name, h_val, np.broadcast_to(h_r, h_val.shape))
                dz_pred = model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
                preds.append(z_r + dz_pred)
        return np.exp(np.mean(np.stack(preds, axis=0), axis=0) * sigma + mu)

    best_val_rmse, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(EPOCHS):
        model.train()
        pi, pj = sample_pairs()
        x = combine(name, h_train[pi], h_train[pj])
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        dz_true = torch.tensor(z_train[pi] - z_train[pj], dtype=torch.float32, device=device)
        pred = model(x_t)
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
    return model


def predict_delta_all(name, model, h_target):
    preds = np.zeros((n_train, len(h_target)))
    with torch.no_grad():
        for r in range(n_train):
            x = combine(name, h_target, np.broadcast_to(h_train[r], h_target.shape))
            preds[r] = model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
    return preds


def agg_global(dz_all, ref_idx, method):
    z_refs = z_train[ref_idx]
    z_cand = z_refs[:, None] + dz_all[ref_idx]
    z_final = np.median(z_cand, axis=0) if method == "median" else np.mean(z_cand, axis=0)
    return np.exp(z_final * sigma + mu)


def agg_knn(dz_all, h_target, K, method):
    n_target = h_target.shape[0]
    z_final = np.zeros(n_target)
    for t in range(n_target):
        d = np.linalg.norm(h_train - h_target[t][None, :], axis=1)
        nn_idx = np.argsort(d)[:K]
        z_cand = z_train[nn_idx] + dz_all[nn_idx, t]
        z_final[t] = np.median(z_cand) if method == "median" else np.mean(z_cand)
    return np.exp(z_final * sigma + mu)


if __name__ == "__main__":
    print(f"cf. V2 solo test={v2_test_rmse:.2f}")
    print(f"cf. established final (diff, global K=91, median, 8-seed): val=83.77 test=63.87\n")

    all_results = {}
    for rep in ["diff", "sqdiff_concat"]:
        print(f"=== Training 8-seed ensemble: {rep} ===")
        models = [train_one(rep, s) for s in range(N_MODEL_SEEDS)]
        dz_val_all = [predict_delta_all(rep, m, h_val) for m in models]
        dz_test_all = [predict_delta_all(rep, m, h_test) for m in models]

        for ref_method in ["global", "knn"]:
            for K in K_GRID:
                for agg_method in ["mean", "median"]:
                    if ref_method == "global":
                        ref_idx = build_reference_set(z_train, K)
                        val_ens = np.mean([agg_global(dz_val_all[i], ref_idx, agg_method) for i in range(N_MODEL_SEEDS)], axis=0)
                        test_ens = np.mean([agg_global(dz_test_all[i], ref_idx, agg_method) for i in range(N_MODEL_SEEDS)], axis=0)
                    else:
                        val_ens = np.mean([agg_knn(dz_val_all[i], h_val, K, agg_method) for i in range(N_MODEL_SEEDS)], axis=0)
                        test_ens = np.mean([agg_knn(dz_test_all[i], h_test, K, agg_method) for i in range(N_MODEL_SEEDS)], axis=0)
                    val_r = rmse(val_ens, val_label); test_r = rmse(test_ens, test_label)
                    key = f"{rep}__{ref_method}__K{K}__{agg_method}"
                    all_results[key] = dict(val_rmse=val_r, test_rmse=test_r)
                    print(f"{rep:14s} {ref_method:7s} K={K:3d} {agg_method:7s} val={val_r:7.2f} test={test_r:7.2f}")

    print("\n=== Stability summary per (representation, ref_method, aggregation) across K=15..91 ===")
    print(f"{'config':40s} {'test_mean':>10s} {'test_std':>9s} {'val_mean':>9s}")
    summary = {}
    for rep in ["diff", "sqdiff_concat"]:
        for ref_method in ["global", "knn"]:
            for agg_method in ["mean", "median"]:
                vals = [all_results[f"{rep}__{ref_method}__K{K}__{agg_method}"] for K in K_GRID]
                test_vals = [v["test_rmse"] for v in vals]; val_vals = [v["val_rmse"] for v in vals]
                key = f"{rep}__{ref_method}__{agg_method}"
                summary[key] = dict(test_mean=np.mean(test_vals), test_std=np.std(test_vals), val_mean=np.mean(val_vals))
                print(f"{key:40s} {np.mean(test_vals):10.2f} {np.std(test_vals):9.2f} {np.mean(val_vals):9.2f}")

    most_stable = min(summary.keys(), key=lambda k: summary[k]["test_std"])
    print(f"\nMost stable overall: {most_stable}  {summary[most_stable]}")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(dict(all_results=all_results, summary=summary, v2_test_rmse=v2_test_rmse), f)
    print(f"\nSaved to {OUT_PKL}")
