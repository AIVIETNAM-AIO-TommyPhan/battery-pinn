"""Pairwise NN v1: a standalone, genuinely nonlinear NN for the inter-cell
Delta-RUL task, trained end-to-end on RAW scalar feature differences (all 34
candidate features) -- NOT the linear ridge model, and NOT dependent on V2's
embedding (that is a separate track, run_inter_embedding_stage1.py).

This is meant to be THE artifact to iterate on going forward for the
"pairwise NN" direction -- future improvements (architecture, feature
subset, loss) should modify this model, not the linear baseline
(run_pairwise_linear_baseline.py / run_pairwise_feature_selection.py, both
now closed as an inferior, linear-only track).

Same protocol conventions as the Inter-Embedding track for comparability:
margin m=0.1, ~32 anchors x <=4 targets/epoch, fixed 8-cell reference set
(3 short/3 mid/2 long RUL tercile, seed=0), checkpoint selected by
multi-reference val RMSE (not train loss).
"""
import pickle
import numpy as np
import torch
import torch.nn as nn

CACHE_DIR = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML\ipynb\exp_v3\matr1_feature_cache"
FEATURE_CACHE_PKL = f"{CACHE_DIR}\\feature_cache.pkl"
OUT_PKL = f"{CACHE_DIR}\\pairwise_nn_v1.pkl"

MARGIN_M = 0.1
REF_SEED = 0
ANCHORS_PER_EPOCH = 32
TARGETS_PER_ANCHOR = 4
EPOCHS = 2000
PATIENCE = 300
EVAL_EVERY = 25

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}")


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


class PairwiseNN(nn.Module):
    """MLP on raw feature differences -> Delta_z. Deliberately nonlinear
    (2 hidden layers) so it can capture interactions a linear ridge model
    cannot -- this is the axis being changed vs. the ridge baseline."""

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
train_scalar = np.array(fc["train_base_scalar_raw"]).astype(np.float32)
val_scalar = np.array(fc["val_scalar_raw"]).astype(np.float32)
test_scalar = np.array(fc["test_scalar_raw"]).astype(np.float32)
train_label = np.array(fc["train_base_label"]).ravel()
val_label = np.array(fc["val_label"]).ravel()
test_label = np.array(fc["test_label"]).ravel()

# standardize raw features (fit on train only) -- the ridge model implicitly
# handled scale via its own coefficients; an NN needs this done explicitly
feat_mean = train_scalar.mean(axis=0, keepdims=True)
feat_std = train_scalar.std(axis=0, keepdims=True) + 1e-8
train_scalar_n = (train_scalar - feat_mean) / feat_std
val_scalar_n = (val_scalar - feat_mean) / feat_std
test_scalar_n = (test_scalar - feat_mean) / feat_std

log_train = np.log(train_label)
mu, sigma = log_train.mean(), log_train.std()
z_train = (log_train - mu) / sigma
n_train = len(z_train)


def fixed_references(z, seed=REF_SEED):
    n = len(z)
    order = np.argsort(z)
    short_pool = order[: n // 3]
    mid_pool = order[n // 3: 2 * n // 3]
    long_pool = order[2 * n // 3:]
    rng = np.random.RandomState(seed)
    refs = list(rng.choice(short_pool, size=3, replace=False)) + \
           list(rng.choice(mid_pool, size=3, replace=False)) + \
           list(rng.choice(long_pool, size=2, replace=False))
    return np.array(refs)


REF_IDX = fixed_references(z_train)
print(f"Fixed reference cells: {REF_IDX.tolist()}, true RUL: {[round(x,1) for x in train_label[REF_IDX]]}")

X_refs = train_scalar_n[REF_IDX]
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
model = PairwiseNN(dim=34, hidden=64).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

X_train_t = torch.tensor(train_scalar_n, dtype=torch.float32, device=device)
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
        val_pred = infer(val_scalar_n)
        val_rmse = rmse(val_pred, val_label)
        improved = val_rmse < best_val_rmse
        if improved:
            best_val_rmse, best_epoch = val_rmse, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"  [pairwise-nn] [{epoch+1:4d}/{EPOCHS}] loss={loss.item():.4f} "
              f"RMSE_val={val_rmse:8.2f}{'  <- best val' if improved else ''}", flush=True)
        if (epoch + 1) - best_epoch >= PATIENCE:
            print(f"  [pairwise-nn] early stop at epoch {epoch+1}", flush=True)
            break

model.load_state_dict(best_state); model.eval()
val_pred = infer(val_scalar_n)
test_pred = infer(test_scalar_n)
val_rmse = rmse(val_pred, val_label)
test_rmse = rmse(test_pred, test_label)
print(f"\n=== Pairwise NN v1 (raw 34-feature diff, nonlinear MLP): "
      f"val_RMSE={val_rmse:.2f} test_RMSE={test_rmse:.2f} (best_epoch={best_epoch}) ===")

# reference comparisons already established this session
print(f"cf. linear ridge (all34): val=224.96 test=247.98")
print(f"cf. linear ridge (feature-selected k=7): val=139.35 test=171.01")
print(f"cf. V2 solo (seed0): val=? test=75.32")

with open(OUT_PKL, "wb") as f:
    pickle.dump(dict(val_pred=val_pred, test_pred=test_pred, val_rmse=val_rmse, test_rmse=test_rmse,
                      val_true=val_label, test_true=test_label, ref_idx=REF_IDX.tolist(),
                      best_epoch=best_epoch), f)
print(f"Saved to {OUT_PKL}")
