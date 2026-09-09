"""Compare more robust central-tendency aggregation methods (beyond mean vs.
median) for combining the K reference-implied z estimates, on top of the
same 8-seed delta-model ensemble. Reports BOTH the point RMSE per K AND the
stability across K (std of test RMSE across the K grid) -- stability across
K is the criterion that distinguished a real effect (median) from noise
(mean) in the previous sweep, so it is reported explicitly here rather than
just picking the single best number.

Methods tested, all applied to the K reference-implied z candidates:
  mean, median (reference points from the previous sweep)
  trimmed_mean(10%), trimmed_mean(20%), trimmed_mean(30%)
  huber (robust M-estimator location, iteratively reweighted)
  geomean (geometric mean, computed in RUL space -- z-space is already
           log-scaled, so geometric mean in RUL space corresponds to
           arithmetic mean in an even more compressed log-log space;
           included as a cheap alternative worth checking empirically)
"""
import pickle
import numpy as np
import torch
import torch.nn as nn

CACHE_DIR = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML\ipynb\exp_v3\matr1_feature_cache"
EMB_PKL = f"{CACHE_DIR}\\v2_embeddings_seed0.pkl"
OUT_PKL = f"{CACHE_DIR}\\inter_embedding_agg_methods.pkl"

MARGIN_M = 0.1
REF_SEED = 0
ANCHORS_PER_EPOCH = 32
TARGETS_PER_ANCHOR = 4
EPOCHS = 1000
PATIENCE = 200
EVAL_EVERY = 25
HIDDEN = 64
N_MODEL_SEEDS = 8
K_GRID = [15, 20, 30, 40, 60, 91]  # the range where median was stable

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
    preds = np.zeros((n_train, len(h_target)))
    with torch.no_grad():
        for r in range(n_train):
            dh = torch.tensor(h_target - h_train[r][None, :], dtype=torch.float32, device=device)
            preds[r] = model(dh).cpu().numpy()
    return preds


def huber_location(x, axis=0, c=1.345, n_iter=20):
    """Iteratively reweighted robust location (Huber M-estimator), applied
    along `axis`. x: array with the aggregation axis first."""
    x = np.moveaxis(x, axis, 0)
    loc = np.median(x, axis=0)
    for _ in range(n_iter):
        resid = x - loc[None, ...]
        mad = np.median(np.abs(resid), axis=0) + 1e-8
        scaled = resid / (1.4826 * mad[None, ...])
        w = np.where(np.abs(scaled) <= c, 1.0, c / (np.abs(scaled) + 1e-8))
        loc = (w * x).sum(axis=0) / w.sum(axis=0)
    return loc


def geomean_rul(z_cand):
    """Geometric mean computed directly in RUL space (exponentiate each
    candidate z first, then geometric-mean the resulting RUL values)."""
    rul_cand = np.exp(z_cand * sigma + mu)
    log_rul = np.log(np.clip(rul_cand, 1e-3, None))
    return np.exp(log_rul.mean(axis=0))  # returns RUL directly, not z


def agg(dz_all, ref_idx, method):
    z_refs = z_train[ref_idx]
    z_cand = z_refs[:, None] + dz_all[ref_idx]  # [K, n_target]
    if method == "mean":
        z_final = z_cand.mean(axis=0)
    elif method == "median":
        z_final = np.median(z_cand, axis=0)
    elif method.startswith("trim"):
        trim = float(method.split("_")[1])
        from scipy.stats import trim_mean
        z_final = trim_mean(z_cand, trim, axis=0)
    elif method == "huber":
        z_final = huber_location(z_cand, axis=0)
    elif method == "geomean":
        return geomean_rul(z_cand)  # already in RUL space
    else:
        raise ValueError(method)
    return np.exp(z_final * sigma + mu)


if __name__ == "__main__":
    print(f"cf. V2 solo test={v2_test_rmse:.2f}")
    print(f"cf. current best (global K=91, median, 8-seed ensemble): val=83.77 test=63.87\n")

    models = [train_one(s) for s in range(N_MODEL_SEEDS)]
    print(f"Trained {N_MODEL_SEEDS} delta-model seeds.\n")

    dz_val_all = [predict_delta_all(m, h_val) for m in models]
    dz_test_all = [predict_delta_all(m, h_test) for m in models]

    methods = ["mean", "median", "trim_0.1", "trim_0.2", "trim_0.3", "huber", "geomean"]
    results = {m: {} for m in methods}

    for method in methods:
        print(f"--- {method} ---")
        for K in K_GRID:
            ref_idx = build_reference_set(z_train, K)
            val_ens = np.mean([agg(dz_val_all[i], ref_idx, method) for i in range(N_MODEL_SEEDS)], axis=0)
            test_ens = np.mean([agg(dz_test_all[i], ref_idx, method) for i in range(N_MODEL_SEEDS)], axis=0)
            val_r = rmse(val_ens, val_label); test_r = rmse(test_ens, test_label)
            results[method][K] = dict(val_rmse=val_r, test_rmse=test_r)
            print(f"  K={K:3d}  val={val_r:7.2f}  test={test_r:7.2f}")

    print("\n=== Stability summary (std of test RMSE across K=15..91 -- lower = more trustworthy) ===")
    print(f"{'method':10s} {'test_mean':>10s} {'test_std':>9s} {'val_mean':>9s}")
    for method in methods:
        test_vals = [results[method][K]["test_rmse"] for K in K_GRID]
        val_vals = [results[method][K]["val_rmse"] for K in K_GRID]
        print(f"{method:10s} {np.mean(test_vals):10.2f} {np.std(test_vals):9.2f} {np.mean(val_vals):9.2f}")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved to {OUT_PKL}")
