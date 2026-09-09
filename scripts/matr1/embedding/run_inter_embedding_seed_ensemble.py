"""Seed-ensemble of the delta model itself (NOT a new V2 retrain -- reuses
the same cached V2 embeddings from v2_embeddings_seed0.pkl). Trains the
original best-known architecture (raw embeddings, plain 2-layer MLP,
Linear(48,64)->ReLU->Linear(64,1), K=8 mean at checkpoint-selection time)
multiple times with different random seeds (weight init + pair sampling),
then averages the resulting RUL predictions across the delta-model seeds --
a cheap variance-reduction step, following this project's established
seed-ensembling convention, that does not require retraining V2's CNN at
all.

Also tries combining this seed-ensemble with the K=91+median aggregation
found earlier (run_inter_embedding_batlinet_agg.py) to see whether the two
independent variance-reduction ideas (more delta-model seeds; more
reference cells + robust aggregation) compound.
"""
import pickle
import numpy as np
import torch
import torch.nn as nn

CACHE_DIR = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML\ipynb\exp_v3\matr1_feature_cache"
EMB_PKL = f"{CACHE_DIR}\\v2_embeddings_seed0.pkl"
OUT_PKL = f"{CACHE_DIR}\\inter_embedding_seed_ensemble.pkl"

MARGIN_M = 0.1
REF_SEED = 0
ANCHORS_PER_EPOCH = 32
TARGETS_PER_ANCHOR = 4
EPOCHS = 1000
PATIENCE = 200
EVAL_EVERY = 25
HIDDEN = 64
N_MODEL_SEEDS = 8

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


REF_IDX_8 = build_reference_set(z_train, 8)
REF_IDX_91 = build_reference_set(z_train, 91)
h_train_t = torch.tensor(h_train, dtype=torch.float32, device=device)
z_train_t = torch.tensor(z_train, dtype=torch.float32, device=device)


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
    return model, best_val_rmse


def predict_with_refs(model, h_target, ref_idx, method="mean"):
    h_refs = h_train[ref_idx]; z_refs = z_train[ref_idx]
    preds = []
    with torch.no_grad():
        for h_r, z_r in zip(h_refs, z_refs):
            dh = torch.tensor(h_target - h_r[None, :], dtype=torch.float32, device=device)
            dz_pred = model(dh).cpu().numpy()
            preds.append(z_r + dz_pred)
    stacked = np.stack(preds, axis=0)
    z_final = np.median(stacked, axis=0) if method == "median" else np.mean(stacked, axis=0)
    return np.exp(z_final * sigma + mu)


if __name__ == "__main__":
    print(f"cf. V2 solo test={v2_test_rmse:.2f}")
    print(f"cf. original single-seed (K=8,mean): val=84.90 test=68.41")
    print(f"cf. BatLiNet-agg best test (K=91,median,single delta model): test=66.18\n")

    models = []
    for seed in range(N_MODEL_SEEDS):
        model, val_rmse_k8 = train_one(seed)
        models.append(model)
        print(f"[delta-model seed={seed}] val_RMSE(K=8,mean)={val_rmse_k8:.2f}")

    print("\n=== Seed-ensemble of delta models (average final RUL predictions across seeds) ===")
    for K, ref_idx, ref_name in [(8, REF_IDX_8, "K=8"), (91, REF_IDX_91, "K=91")]:
        for method in ["mean", "median"]:
            val_preds = np.stack([predict_with_refs(m, h_val, ref_idx, method) for m in models], axis=0)
            test_preds = np.stack([predict_with_refs(m, h_test, ref_idx, method) for m in models], axis=0)
            val_ens = val_preds.mean(axis=0)
            test_ens = test_preds.mean(axis=0)
            val_r = rmse(val_ens, val_label)
            test_r = rmse(test_ens, test_label)
            print(f"{ref_name:6s} {method:8s} (ensemble of {N_MODEL_SEEDS} delta-model seeds): "
                  f"val={val_r:7.2f} test={test_r:7.2f}")

    # also report individual per-seed numbers for the record (K=8, mean)
    print("\n=== Individual delta-model seeds (K=8, mean) for reference ===")
    per_seed_test = []
    for i, m in enumerate(models):
        test_pred = predict_with_refs(m, h_test, REF_IDX_8, "mean")
        test_r = rmse(test_pred, test_label)
        per_seed_test.append(test_r)
        print(f"seed={i}: test={test_r:.2f}")
    print(f"per-seed mean+-std: {np.mean(per_seed_test):.2f} +- {np.std(per_seed_test):.2f}")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(dict(per_seed_test=per_seed_test, v2_test_rmse=v2_test_rmse), f)
    print(f"\nSaved to {OUT_PKL}")
