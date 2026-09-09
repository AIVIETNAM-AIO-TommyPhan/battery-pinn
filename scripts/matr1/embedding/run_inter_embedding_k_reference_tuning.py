"""K/reference-selection tuning on top of the 8-seed delta-model ensemble
(winning config from run_inter_embedding_seed_ensemble.py: K=91, median,
test=63.87). Two reference-selection families compared:

  A. Global fixed set (current): RUL-tercile-stratified, same set of K cells
     used for every target cell -- fine-grained K sweep.
  B. Per-target k-NN (NEW): for each target cell individually, pick its own
     K nearest TRAIN cells by embedding distance (L2 or cosine) -- a
     "similarity score"-based reference set, personalized per target,
     instead of one fixed global set.

Both swept over K in {5,8,15,20,30,40,60,91}, both mean and median
aggregation, using the SAME 8 delta-model seed-ensemble throughout (no
retraining -- purely an inference-time comparison).
"""
import pickle
import numpy as np
import torch
import torch.nn as nn

CACHE_DIR = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML\ipynb\exp_v3\matr1_feature_cache"
EMB_PKL = f"{CACHE_DIR}\\v2_embeddings_seed0.pkl"
OUT_PKL = f"{CACHE_DIR}\\inter_embedding_k_reference_tuning.pkl"

MARGIN_M = 0.1
REF_SEED = 0
ANCHORS_PER_EPOCH = 32
TARGETS_PER_ANCHOR = 4
EPOCHS = 1000
PATIENCE = 200
EVAL_EVERY = 25
HIDDEN = 64
N_MODEL_SEEDS = 8
K_GRID = [5, 8, 15, 20, 30, 40, 60, 91]

device = "cuda" if torch.cuda.is_available() else "cpu"


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


class InterEmbeddingModel(nn.Module):
    def __init__(self, dim=48, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, delta_h):
        return self.net(delta_h).squeeze(-1)


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


h_train_t = torch.tensor(h_train, dtype=torch.float32, device=device)
z_train_t = torch.tensor(z_train, dtype=torch.float32, device=device)
REF_IDX_8 = build_reference_set(z_train, 8)


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
    return model


def predict_delta_all(model, h_target):
    """Delta_z(target, r) vs every train cell r -- reused across all K/methods."""
    preds = np.zeros((n_train, len(h_target)))
    with torch.no_grad():
        for r in range(n_train):
            dh = torch.tensor(h_target - h_train[r][None, :], dtype=torch.float32, device=device)
            preds[r] = model(dh).cpu().numpy()
    return preds  # [n_train, n_target]


def agg_global(dz_all, ref_idx, method):
    z_refs = z_train[ref_idx]
    z_cand = z_refs[:, None] + dz_all[ref_idx]  # [K, n_target]
    z_final = np.median(z_cand, axis=0) if method == "median" else np.mean(z_cand, axis=0)
    return np.exp(z_final * sigma + mu)


def agg_per_target_knn(dz_all, h_target, K, method, metric="l2"):
    """Personalized reference set per target cell, by embedding distance."""
    n_target = h_target.shape[0]
    z_final = np.zeros(n_target)
    for t in range(n_target):
        if metric == "l2":
            d = np.linalg.norm(h_train - h_target[t][None, :], axis=1)
        else:  # cosine distance
            a = h_train / (np.linalg.norm(h_train, axis=1, keepdims=True) + 1e-8)
            b = h_target[t] / (np.linalg.norm(h_target[t]) + 1e-8)
            d = 1 - a @ b
        nn_idx = np.argsort(d)[:K]
        z_cand = z_train[nn_idx] + dz_all[nn_idx, t]
        z_final[t] = np.median(z_cand) if method == "median" else np.mean(z_cand)
    return np.exp(z_final * sigma + mu)


if __name__ == "__main__":
    print(f"cf. V2 solo test={v2_test_rmse:.2f}")
    print(f"cf. winning config so far (K=91 global, median, 8-seed ensemble): val=83.77 test=63.87\n")

    models = [train_one(s) for s in range(N_MODEL_SEEDS)]
    print(f"Trained {N_MODEL_SEEDS} delta-model seeds.\n")

    dz_val_all = [predict_delta_all(m, h_val) for m in models]
    dz_test_all = [predict_delta_all(m, h_test) for m in models]

    results = {}

    print("=== A. Global fixed reference set (RUL-tercile), K sweep ===")
    for K in K_GRID:
        ref_idx = build_reference_set(z_train, K)
        for method in ["mean", "median"]:
            val_ens = np.mean([agg_global(dz_val_all[i], ref_idx, method) for i in range(N_MODEL_SEEDS)], axis=0)
            test_ens = np.mean([agg_global(dz_test_all[i], ref_idx, method) for i in range(N_MODEL_SEEDS)], axis=0)
            val_r = rmse(val_ens, val_label); test_r = rmse(test_ens, test_label)
            results[f"global_K{K}_{method}"] = dict(val_rmse=val_r, test_rmse=test_r)
            print(f"K={K:3d} {method:8s} val={val_r:7.2f} test={test_r:7.2f}")

    print("\n=== B. Per-target k-NN reference set (L2 embedding distance), K sweep ===")
    for K in K_GRID:
        for method in ["mean", "median"]:
            val_ens = np.mean([agg_per_target_knn(dz_val_all[i], h_val, K, method, "l2") for i in range(N_MODEL_SEEDS)], axis=0)
            test_ens = np.mean([agg_per_target_knn(dz_test_all[i], h_test, K, method, "l2") for i in range(N_MODEL_SEEDS)], axis=0)
            val_r = rmse(val_ens, val_label); test_r = rmse(test_ens, test_label)
            results[f"knnL2_K{K}_{method}"] = dict(val_rmse=val_r, test_rmse=test_r)
            print(f"K={K:3d} {method:8s} val={val_r:7.2f} test={test_r:7.2f}")

    print("\n=== C. Per-target k-NN reference set (cosine distance), K sweep ===")
    for K in K_GRID:
        for method in ["mean", "median"]:
            val_ens = np.mean([agg_per_target_knn(dz_val_all[i], h_val, K, method, "cosine") for i in range(N_MODEL_SEEDS)], axis=0)
            test_ens = np.mean([agg_per_target_knn(dz_test_all[i], h_test, K, method, "cosine") for i in range(N_MODEL_SEEDS)], axis=0)
            val_r = rmse(val_ens, val_label); test_r = rmse(test_ens, test_label)
            results[f"knnCos_K{K}_{method}"] = dict(val_rmse=val_r, test_rmse=test_r)
            print(f"K={K:3d} {method:8s} val={val_r:7.2f} test={test_r:7.2f}")

    best_key = min(results.keys(), key=lambda k: results[k]["val_rmse"])
    print(f"\n=== Best by VAL: {best_key}  val={results[best_key]['val_rmse']:.2f} test={results[best_key]['test_rmse']:.2f} ===")
    print(f"cf. previous best (global K=91 median): val=83.77 test=63.87")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(dict(results=results, best_key=best_key, v2_test_rmse=v2_test_rmse), f)
    print(f"\nSaved to {OUT_PKL}")
