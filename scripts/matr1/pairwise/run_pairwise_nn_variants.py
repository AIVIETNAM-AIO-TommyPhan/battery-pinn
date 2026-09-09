"""Compare 2 pairwise-NN variants to pick which architecture is worth
building further on:
  (a) NN on the k=7 feature-selected subset (already shown to help the
      linear model: val 224.96 -> 139.35)
  (b) NN on raw 34 features but regularized (smaller hidden dim + weight
      decay) to test whether v1's instability was a capacity/overfitting
      issue rather than an architecture-choice issue.
Same protocol/reference-set conventions as run_pairwise_nn_v1.py throughout.
"""
import pickle
import numpy as np
import torch
import torch.nn as nn

CACHE_DIR = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML\ipynb\exp_v3\matr1_feature_cache"
FEATURE_CACHE_PKL = f"{CACHE_DIR}\\feature_cache.pkl"
OUT_PKL = f"{CACHE_DIR}\\pairwise_nn_variants.pkl"

MARGIN_M = 0.1
REF_SEED = 0
ANCHORS_PER_EPOCH = 32
TARGETS_PER_ANCHOR = 4
EPOCHS = 2000
PATIENCE = 300
EVAL_EVERY = 25

device = "cuda" if torch.cuda.is_available() else "cpu"


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


class PairwiseNN(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, dx):
        return self.net(dx).squeeze(-1)


with open(FEATURE_CACHE_PKL, "rb") as f:
    fc = pickle.load(f)

names = fc["feature_names"]
name_idx = {n: i for i, n in enumerate(names)}
train_scalar_full = np.array(fc["train_base_scalar_raw"]).astype(np.float32)
val_scalar_full = np.array(fc["val_scalar_raw"]).astype(np.float32)
test_scalar_full = np.array(fc["test_scalar_raw"]).astype(np.float32)
train_label = np.array(fc["train_base_label"]).ravel()
val_label = np.array(fc["val_label"]).ravel()
test_label = np.array(fc["test_label"]).ravel()

log_train = np.log(train_label)
mu, sigma = log_train.mean(), log_train.std()
z_train = (log_train - mu) / sigma
n_train = len(z_train)


def fixed_references(z, seed=REF_SEED):
    n = len(z)
    order = np.argsort(z)
    short_pool = order[: n // 3]; mid_pool = order[n // 3: 2 * n // 3]; long_pool = order[2 * n // 3:]
    rng = np.random.RandomState(seed)
    refs = list(rng.choice(short_pool, size=3, replace=False)) + \
           list(rng.choice(mid_pool, size=3, replace=False)) + \
           list(rng.choice(long_pool, size=2, replace=False))
    return np.array(refs)


REF_IDX = fixed_references(z_train)


def run_variant(name, feat_idx, hidden, weight_decay):
    X_train = train_scalar_full[:, feat_idx]
    X_val = val_scalar_full[:, feat_idx]
    X_test = test_scalar_full[:, feat_idx]
    feat_mean = X_train.mean(axis=0, keepdims=True)
    feat_std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train_n = (X_train - feat_mean) / feat_std
    X_val_n = (X_val - feat_mean) / feat_std
    X_test_n = (X_test - feat_mean) / feat_std

    X_refs = X_train_n[REF_IDX]
    z_refs = z_train[REF_IDX]

    rng = np.random.RandomState(0)

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

    torch.manual_seed(0)
    model = PairwiseNN(dim=len(feat_idx), hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)

    X_train_t = torch.tensor(X_train_n, dtype=torch.float32, device=device)
    z_train_t = torch.tensor(z_train, dtype=torch.float32, device=device)

    def infer(X_target_n):
        preds = []
        with torch.no_grad():
            for x_r, z_r in zip(X_refs, z_refs):
                dx = torch.tensor(X_target_n - x_r[None, :], dtype=torch.float32, device=device)
                dz_pred = model(dx).cpu().numpy()
                preds.append(z_r + dz_pred)
        z_final = np.mean(np.stack(preds, axis=0), axis=0)
        return np.exp(z_final * sigma + mu)

    best_val_rmse, best_epoch, best_state = float("inf"), None, None
    for epoch in range(EPOCHS):
        model.train()
        pi, pj = sample_pairs()
        dx = X_train_t[pi] - X_train_t[pj]
        dz_true = z_train_t[pi] - z_train_t[pj]
        pred = model(dx)
        loss = ((pred - dz_true) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

        if (epoch + 1) % EVAL_EVERY == 0 or (epoch + 1) == EPOCHS:
            model.eval()
            val_pred = infer(X_val_n)
            val_rmse = rmse(val_pred, val_label)
            improved = val_rmse < best_val_rmse
            if improved:
                best_val_rmse, best_epoch = val_rmse, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) - best_epoch >= PATIENCE:
                break

    model.load_state_dict(best_state); model.eval()
    val_pred = infer(X_val_n)
    test_pred = infer(X_test_n)
    val_rmse = rmse(val_pred, val_label)
    test_rmse = rmse(test_pred, test_label)
    print(f"[{name}] dim={len(feat_idx)} hidden={hidden} wd={weight_decay} "
          f"best_epoch={best_epoch} val_RMSE={val_rmse:.2f} test_RMSE={test_rmse:.2f}")
    return dict(val_rmse=val_rmse, test_rmse=test_rmse, val_pred=val_pred, test_pred=test_pred, best_epoch=best_epoch)


if __name__ == "__main__":
    k7_features = ['qdlin_diff_min', 'qdlin_diff_l2', 'voltage_hysteresis', 'qdlin_diff_mean',
                   'qdlin_diff_l1', 'temperature_rise_charge', 'qdlin_diff_std']
    k7_idx = [name_idx[n] for n in k7_features]

    results = {}
    results["a_k7_nn"] = run_variant("a_k7_nn", k7_idx, hidden=32, weight_decay=0.0)
    results["b_raw34_regularized"] = run_variant("b_raw34_regularized", list(range(34)), hidden=16, weight_decay=1e-2)
    results["c_k7_nn_regularized"] = run_variant("c_k7_nn_regularized", k7_idx, hidden=16, weight_decay=1e-2)

    print("\n=== Summary ===")
    for name, r in results.items():
        print(f"{name:24s} val={r['val_rmse']:8.2f} test={r['test_rmse']:8.2f}")
    print(f"cf. linear ridge k=7:      val=139.35 test=171.01")
    print(f"cf. pairwise-nn v1 raw34:  val=225.54 test=243.03")
    print(f"cf. V2 solo:               val=?      test=75.32")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved to {OUT_PKL}")
