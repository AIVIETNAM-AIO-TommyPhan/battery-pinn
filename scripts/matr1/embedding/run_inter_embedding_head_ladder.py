"""NN-head architecture ladder for the Inter-Embedding Model, on cached V2
embeddings (extract_v2_embeddings_seed0.py). Pair representation FIXED at
"diff" (h_i - h_j, dim=48) throughout -- this experiment isolates the
contribution of NN head capacity alone, not representation choice (that was
run_inter_embedding_delta_variants.py, a separate, now-closed experiment).

Ladder:
  H0: Linear(48 -> 1)                                    -- lower baseline
  H1: Linear(48->64) -> GELU -> Linear(64->1)             -- nonlinear baseline
  H2: in_proj(48->64) -> 1 residual block(64) -> Linear(64->1)  -- main candidate

All three wrapped in the same antisymmetric construction:
  pred(hi,hj) = 0.5*(g(hi-hj) - g(hj-hi))
which guarantees pred(j,i) = -pred(i,j) exactly, regardless of head
nonlinearity (for H0/linear this is automatically satisfied without the
wrapper since a linear map of -x is -g(x); the wrapper is kept uniformly
across H0/H1/H2 for a clean, single-code-path comparison and is a strict
no-op for H0).

Everything else held fixed across H0/H1/H2: embeddings (train-only
standardized), pair sampling (margin=0.1, ~32 anchors x <=4 targets/epoch,
seed=0), fixed 8-cell reference set (3 short/3 mid/2 long tercile, seed=0),
optimizer (AdamW, lr=1e-3, weight_decay=1e-4), epoch cap 1000, patience=200
on val RMSE, gradient clipping at 1.0. Selection: pick the winner by VAL
RMSE only; reveal test RMSE for the winner once.
"""
import pickle
import numpy as np
import torch
import torch.nn as nn

CACHE_DIR = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML\ipynb\exp_v3\matr1_feature_cache"
EMB_PKL = f"{CACHE_DIR}\\v2_embeddings_seed0.pkl"
OUT_PKL = f"{CACHE_DIR}\\inter_embedding_head_ladder.pkl"

MARGIN_M = 0.1
REF_SEED = 0
ANCHORS_PER_EPOCH = 32
TARGETS_PER_ANCHOR = 4
EPOCHS = 1000
PATIENCE = 200
EVAL_EVERY = 25
HIDDEN = 64
DROPOUT = 0.05
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
K_NN = 8
DIM = 48  # "diff" representation, fixed

device = "cuda" if torch.cuda.is_available() else "cpu"


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=DROPOUT):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        h = self.act(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        return self.act(self.norm(h + residual))


def make_head(name):
    if name == "H0_linear":
        return nn.Linear(DIM, 1)
    if name == "H1_mlp":
        return nn.Sequential(nn.Linear(DIM, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))
    if name == "H2_residual":
        return nn.Sequential(
            nn.Linear(DIM, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU(), nn.Dropout(DROPOUT),
            ResidualBlock(HIDDEN),
            nn.Linear(HIDDEN, 1),
        )
    raise ValueError(name)


class AntisymmetricPairModel(nn.Module):
    """pred(hi,hj) = 0.5*(g(hi-hj) - g(hj-hi)); antisymmetric by construction."""

    def __init__(self, head_name):
        super().__init__()
        self.g = make_head(head_name)

    def forward(self, hi, hj):
        return 0.5 * (self.g(hi - hj).squeeze(-1) - self.g(hj - hi).squeeze(-1))


class PotentialModel(nn.Module):
    """H3: pred(hi,hj) = phi(hi) - phi(hj), a single-argument learned scalar
    potential over embedding space. Antisymmetric AND cycle-consistent
    (transitive: pred(i,j)+pred(j,k) == pred(i,k) EXACTLY, by construction,
    with no auxiliary loss needed) -- a strictly stronger structural
    guarantee than AntisymmetricPairModel, which only enforces antisymmetry.
    Also cheaper at inference: phi(h_target) computed once, reused against
    every reference."""

    def __init__(self, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(DIM, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            ResidualBlock(hidden),
            nn.Linear(hidden, 1),
        )

    def forward(self, hi, hj):
        return (self.phi(hi) - self.phi(hj)).squeeze(-1)


with open(EMB_PKL, "rb") as f:
    E = pickle.load(f)
h_train_raw = E["h_train"]; h_val_raw = E["h_val"]; h_test_raw = E["h_test"]
train_label = E["train_label"]; val_label = E["val_label"]; test_label = E["test_label"]
v2_test_rmse = E["v2_test_rmse"]

h_mu = h_train_raw.mean(axis=0, keepdims=True)
h_std = h_train_raw.std(axis=0, keepdims=True) + 1e-6
h_train = (h_train_raw - h_mu) / h_std
h_val = (h_val_raw - h_mu) / h_std
h_test = (h_test_raw - h_mu) / h_std

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
print(f"Fixed reference cells: {REF_IDX.tolist()}")

h_train_t = torch.tensor(h_train, dtype=torch.float32, device=device)
z_train_t = torch.tensor(z_train, dtype=torch.float32, device=device)
h_val_t = torch.tensor(h_val, dtype=torch.float32, device=device)
h_test_t = torch.tensor(h_test, dtype=torch.float32, device=device)
h_refs_t = h_train_t[REF_IDX]
z_refs = z_train[REF_IDX]


def infer(model, h_target_t):
    preds = []
    with torch.no_grad():
        for k in range(len(REF_IDX)):
            h_r = h_refs_t[k:k + 1].expand(h_target_t.shape[0], -1)
            dz_pred = model(h_target_t, h_r).cpu().numpy()
            preds.append(z_refs[k] + dz_pred)
    z_final = np.mean(np.stack(preds, axis=0), axis=0)
    return np.exp(z_final * sigma + mu)


def make_model(name):
    if name == "H3_potential":
        return PotentialModel()
    return AntisymmetricPairModel(name)


def run_head(name, seed=0):
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
    model = make_model(name).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)

    best_val_rmse, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(EPOCHS):
        model.train()
        pi, pj = sample_pairs()
        pi_t = torch.tensor(pi, dtype=torch.long, device=device)
        pj_t = torch.tensor(pj, dtype=torch.long, device=device)
        pred = model(h_train_t[pi_t], h_train_t[pj_t])
        dz_true = z_train_t[pi_t] - z_train_t[pj_t]
        loss = ((pred - dz_true) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        if (epoch + 1) % EVAL_EVERY == 0 or (epoch + 1) == EPOCHS:
            model.eval()
            val_pred = infer(model, h_val_t)
            val_rmse = rmse(val_pred, val_label)
            improved = val_rmse < best_val_rmse
            if improved:
                best_val_rmse, best_epoch = val_rmse, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) - best_epoch >= PATIENCE:
                break

    model.load_state_dict(best_state); model.eval()
    val_pred = infer(model, h_val_t)
    val_rmse = rmse(val_pred, val_label)
    print(f"[{name:12s}] n_params={n_params:6d} best_epoch={best_epoch:4d}  val_RMSE={val_rmse:8.2f}  "
          f"(test withheld until winner is picked)")
    return dict(val_rmse=val_rmse, best_epoch=best_epoch, model_state=best_state, n_params=n_params)


if __name__ == "__main__":
    print(f"cf. V2 solo test_RMSE={v2_test_rmse:.2f}\n")

    def knn_predict(h_target):
        preds = []
        for h_t in h_target:
            d = np.linalg.norm(h_train - h_t[None, :], axis=1)
            nn_idx = np.argsort(d)[:K_NN]
            preds.append(train_label[nn_idx].mean())
        return np.array(preds)

    knn_val_rmse = rmse(knn_predict(h_val), val_label)
    print(f"Control: k-NN (K={K_NN}, standardized embedding L2) val_RMSE={knn_val_rmse:.2f}\n")

    print("=== Head ladder screen (fixed 'diff' representation; val RMSE only, test withheld) ===")
    results = {}
    for name in ["H0_linear", "H1_mlp", "H2_residual", "H3_potential"]:
        results[name] = run_head(name)

    winner = min(results.keys(), key=lambda k: results[k]["val_rmse"])
    print(f"\nWinner by VAL: {winner}  val_RMSE={results[winner]['val_rmse']:.2f}")
    print(f"Winner beats k-NN control on val: {results[winner]['val_rmse'] <= knn_val_rmse}")

    model = make_model(winner).to(device)
    model.load_state_dict(results[winner]["model_state"]); model.eval()
    test_pred = infer(model, h_test_t)
    test_rmse = rmse(test_pred, test_label)
    knn_test_rmse = rmse(knn_predict(h_test), test_label)
    print(f"\n=== WINNER ({winner}) test result (revealed once) ===")
    print(f"val_RMSE={results[winner]['val_rmse']:.2f}  test_RMSE={test_rmse:.2f}")
    print(f"cf. V2 solo test={v2_test_rmse:.2f}  cf. k-NN control test={knn_test_rmse:.2f}")
    print(f"winner beats V2 on test: {test_rmse < v2_test_rmse}")

    print("\n=== Full ladder val RMSE (for the record) ===")
    for name, r in results.items():
        marker = " <- WINNER" if name == winner else ""
        print(f"{name:12s} params={r['n_params']:6d} val={r['val_rmse']:8.2f}{marker}")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(dict(results={k: {kk: vv for kk, vv in v.items() if kk != "model_state"}
                                   for k, v in results.items()},
                          winner=winner, winner_test_rmse=test_rmse, knn_val_rmse=knn_val_rmse,
                          knn_test_rmse=knn_test_rmse, v2_test_rmse=v2_test_rmse), f)
    print(f"\nSaved to {OUT_PKL}")
