"""H4: attention-weighted multi-reference aggregation, replacing the uniform
mean over the 8 fixed reference cells with a learned confidence weighting --

    Delta_z_pred(i,r_k) = delta_head(h_i, h_r_k)
    score(i,r_k)         = confidence_head(h_i, h_r_k)
    alpha_i              = softmax_k(score(i, r_1..K))
    z_pred(i)            = sum_k alpha_i,k * (z_r_k + Delta_z_pred(i,r_k))

Unlike H0-H3 (run_inter_embedding_head_ladder.py), which train on a
pairwise Delta_z regression loss and only apply multi-reference averaging
at inference time, H4 trains END-TO-END directly on the actual evaluation
objective (cell-level z RMSE via the fixed 8-reference aggregation) -- a
cleaner training signal, at the cost of one more set of learned parameters
(the confidence head) and a training loop over whole cells instead of
random pairs.

Reuses the DELTA_HEAD_NAME winner from the head ladder (fill in once that
result lands) for the delta head architecture; the confidence head is a
separate small MLP over concat[h_i, h_r] (NOT antisymmetric -- "how much to
trust/weight this reference" is not a signed quantity, unlike Delta_z).

Gate: H4 must beat H0-H3's best (uniform-mean) result on val to justify the
added complexity -- otherwise the extra confidence-head parameters are
overfitting 41 validation cells, not adding real signal.
"""
import pickle
import numpy as np
import torch
import torch.nn as nn

CACHE_DIR = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML\ipynb\exp_v3\matr1_feature_cache"
EMB_PKL = f"{CACHE_DIR}\\v2_embeddings_seed0.pkl"
LADDER_PKL = f"{CACHE_DIR}\\inter_embedding_head_ladder.pkl"
OUT_PKL = f"{CACHE_DIR}\\inter_embedding_attention_agg.pkl"

REF_SEED = 0
EPOCHS = 1000
PATIENCE = 200
EVAL_EVERY = 25
HIDDEN = 64
CONF_HIDDEN = 32
DROPOUT = 0.05
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
K_NN = 8
DIM = 48

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


def make_delta_head(name):
    """Same head family as run_inter_embedding_head_ladder.py, kept in sync
    so the winner's exact architecture can be reused here."""
    if name == "H0_linear":
        g = nn.Linear(DIM, 1)
    elif name == "H1_mlp":
        g = nn.Sequential(nn.Linear(DIM, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))
    elif name == "H2_residual":
        g = nn.Sequential(nn.Linear(DIM, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU(), nn.Dropout(DROPOUT),
                           ResidualBlock(HIDDEN), nn.Linear(HIDDEN, 1))
    else:
        raise ValueError(f"unsupported for attention-agg delta head: {name}")

    class Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.g = g

        def forward(self, hi, hj):
            return 0.5 * (self.g(hi - hj).squeeze(-1) - self.g(hj - hi).squeeze(-1))

    return Wrapped()


class PotentialDeltaHead(nn.Module):
    """H3-style delta head: phi(hi) - phi(hj), for use inside the attention model."""

    def __init__(self, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(DIM, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
                                  ResidualBlock(hidden), nn.Linear(hidden, 1))

    def forward(self, hi, hj):
        return (self.phi(hi) - self.phi(hj)).squeeze(-1)


class ConfidenceHead(nn.Module):
    """NOT antisymmetric -- 'how much to trust/weight this reference' is a
    magnitude, not a signed quantity."""

    def __init__(self, dim=DIM * 2, hidden=CONF_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, hi, hj):
        return self.net(torch.cat([hi, hj], dim=-1)).squeeze(-1)


class AttentionAggModel(nn.Module):
    def __init__(self, delta_head_name):
        super().__init__()
        self.delta_head = PotentialDeltaHead() if delta_head_name == "H3_potential" else make_delta_head(delta_head_name)
        self.confidence_head = ConfidenceHead()

    def forward(self, h_target, h_refs, z_refs):
        """h_target: [B,48]; h_refs: [K,48]; z_refs: [K] (numpy-free, all torch).
        Returns z_pred [B] (in z-space)."""
        K = h_refs.shape[0]
        B = h_target.shape[0]
        h_r_exp = h_refs.unsqueeze(0).expand(B, K, DIM).reshape(B * K, DIM)
        h_t_exp = h_target.unsqueeze(1).expand(B, K, DIM).reshape(B * K, DIM)
        dz = self.delta_head(h_t_exp, h_r_exp).reshape(B, K)
        score = self.confidence_head(h_t_exp, h_r_exp).reshape(B, K)
        alpha = torch.softmax(score, dim=1)
        z_r = z_refs.unsqueeze(0).expand(B, K)
        z_pred = (alpha * (z_r + dz)).sum(dim=1)
        return z_pred, alpha


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
NON_REF_TRAIN_IDX = np.array([i for i in range(n_train) if i not in set(REF_IDX)])
print(f"Fixed reference cells: {REF_IDX.tolist()}  (excluded from training anchors: {len(REF_IDX)} of {n_train})")

h_train_t = torch.tensor(h_train, dtype=torch.float32, device=device)
z_train_t = torch.tensor(z_train, dtype=torch.float32, device=device)
h_val_t = torch.tensor(h_val, dtype=torch.float32, device=device)
h_test_t = torch.tensor(h_test, dtype=torch.float32, device=device)
h_refs_t = h_train_t[REF_IDX]
z_refs_t = z_train_t[REF_IDX]
val_true = val_label
test_true = test_label
train_true_z_anchors = z_train_t[NON_REF_TRAIN_IDX]
h_train_anchors = h_train_t[NON_REF_TRAIN_IDX]


def infer_rul(model, h_target_t):
    model.eval()
    with torch.no_grad():
        z_pred, alpha = model(h_target_t, h_refs_t, z_refs_t)
    rul_pred = np.exp(z_pred.cpu().numpy() * sigma + mu)
    return rul_pred, alpha.cpu().numpy()


def run_attention_agg(delta_head_name, seed=0):
    torch.manual_seed(seed)
    model = AttentionAggModel(delta_head_name).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)

    best_val_rmse, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(EPOCHS):
        model.train()
        z_pred, _ = model(h_train_anchors, h_refs_t, z_refs_t)
        loss = ((z_pred - train_true_z_anchors) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        if (epoch + 1) % EVAL_EVERY == 0 or (epoch + 1) == EPOCHS:
            val_pred, _ = infer_rul(model, h_val_t)
            val_rmse = rmse(val_pred, val_true)
            improved = val_rmse < best_val_rmse
            if improved:
                best_val_rmse, best_epoch = val_rmse, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) - best_epoch >= PATIENCE:
                break

    model.load_state_dict(best_state); model.eval()
    print(f"[attn-agg / {delta_head_name}] n_params={n_params} best_epoch={best_epoch} "
          f"val_RMSE={best_val_rmse:.2f}")
    return model, best_val_rmse, best_epoch


if __name__ == "__main__":
    import os
    with open(LADDER_PKL, "rb") as f:
        ladder = pickle.load(f)
    winner = ladder["winner"]
    uniform_mean_val_rmse = ladder["results"][winner]["val_rmse"]
    uniform_mean_test_rmse = ladder["winner_test_rmse"]
    print(f"Head-ladder winner (uniform-mean aggregation): {winner}  "
          f"val={uniform_mean_val_rmse:.2f} test={uniform_mean_test_rmse:.2f}\n")

    def knn_predict(h_target):
        preds = []
        for h_t in h_target:
            d = np.linalg.norm(h_train - h_t[None, :], axis=1)
            nn_idx = np.argsort(d)[:K_NN]
            preds.append(train_label[nn_idx].mean())
        return np.array(preds)

    knn_val_rmse = rmse(knn_predict(h_val), val_label)

    model, val_rmse, best_epoch = run_attention_agg(winner)
    val_pred, val_alpha = infer_rul(model, h_val_t)
    test_pred, test_alpha = infer_rul(model, h_test_t)
    test_rmse = rmse(test_pred, test_label)

    print(f"\n=== H4 attention-agg ({winner} delta head) ===")
    print(f"val_RMSE={val_rmse:.2f}  test_RMSE={test_rmse:.2f}")
    print(f"cf. same delta head, uniform-mean (H-ladder): val={uniform_mean_val_rmse:.2f} test={uniform_mean_test_rmse:.2f}")
    print(f"cf. V2 solo test={v2_test_rmse:.2f}  cf. k-NN control val={knn_val_rmse:.2f}")
    print(f"H4 beats uniform-mean on val: {val_rmse < uniform_mean_val_rmse}")
    print(f"attention weight spread (val), mean per-reference alpha: {val_alpha.mean(axis=0).round(3).tolist()}")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(dict(delta_head=winner, val_rmse=val_rmse, test_rmse=test_rmse,
                          uniform_mean_val_rmse=uniform_mean_val_rmse, uniform_mean_test_rmse=uniform_mean_test_rmse,
                          val_alpha=val_alpha, test_alpha=test_alpha, v2_test_rmse=v2_test_rmse), f)
    print(f"\nSaved to {OUT_PKL}")
