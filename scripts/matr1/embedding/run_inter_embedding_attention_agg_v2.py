"""H4 v2: attention-weighted aggregation, this time built on the ORIGINAL
best-known delta architecture (raw/unstandardized embeddings, plain 2-layer
MLP -- matching run_inter_embedding_stage1.py exactly), not the standardized
+ antisymmetric H0 winner from run_inter_embedding_head_ladder.py that the
first H4 attempt (run_inter_embedding_attention_agg.py) was built on.

Rationale: the first H4 test was gated on an already-underperforming base
(H0: val=87.93/test=76.96, vs. the original's 84.90/68.41), so a "H4 worse
than H0" result didn't tell us whether attention-aggregation itself is bad,
or just inherited H0's disadvantage. This is the fairer test.

    Delta_z_pred(i,r_k) = delta_head(h_i, h_r_k)   # same MLP family as the original
    score(i,r_k)         = confidence_head(h_i, h_r_k)
    alpha_i               = softmax_k(score(i, r_1..K))
    z_pred(i)             = sum_k alpha_i,k * (z_r_k + Delta_z_pred(i,r_k))

Trained end-to-end on the cell-level z loss (not a pairwise Delta_z loss),
using the fixed K=8 reference set (3 short/3 mid/2 long tercile, seed=0) for
direct comparability with the original.
"""
import pickle
import numpy as np
import torch
import torch.nn as nn

CACHE_DIR = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML\ipynb\exp_v3\matr1_feature_cache"
EMB_PKL = f"{CACHE_DIR}\\v2_embeddings_seed0.pkl"
OUT_PKL = f"{CACHE_DIR}\\inter_embedding_attention_agg_v2.pkl"

REF_SEED = 0
EPOCHS = 1000
PATIENCE = 200
EVAL_EVERY = 25
HIDDEN = 64
CONF_HIDDEN = 32
DIM = 48

device = "cuda" if torch.cuda.is_available() else "cpu"


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


class DeltaHead(nn.Module):
    """Exact match to run_inter_embedding_stage1.py's InterEmbeddingModel:
    plain 2-layer MLP on raw Delta_h = hi - hj, no antisymmetric wrapper,
    no standardization upstream."""

    def __init__(self, dim=DIM, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, hi, hj):
        return self.net(hi - hj).squeeze(-1)


class ConfidenceHead(nn.Module):
    def __init__(self, dim=DIM * 2, hidden=CONF_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, hi, hj):
        return self.net(torch.cat([hi, hj], dim=-1)).squeeze(-1)


class AttentionAggModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.delta_head = DeltaHead()
        self.confidence_head = ConfidenceHead()

    def forward(self, h_target, h_refs, z_refs):
        K = h_refs.shape[0]; B = h_target.shape[0]
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
h_train = E["h_train"]; h_val = E["h_val"]; h_test = E["h_test"]  # RAW, unstandardized
train_label = E["train_label"]; val_label = E["val_label"]; test_label = E["test_label"]
v2_test_rmse = E["v2_test_rmse"]

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
NON_REF_IDX = np.array([i for i in range(n_train) if i not in set(REF_IDX)])
print(f"Fixed reference cells: {REF_IDX.tolist()}")

h_train_t = torch.tensor(h_train, dtype=torch.float32, device=device)
z_train_t = torch.tensor(z_train, dtype=torch.float32, device=device)
h_val_t = torch.tensor(h_val, dtype=torch.float32, device=device)
h_test_t = torch.tensor(h_test, dtype=torch.float32, device=device)
h_refs_t = h_train_t[REF_IDX]; z_refs_t = z_train_t[REF_IDX]
h_train_anchors = h_train_t[NON_REF_IDX]; z_train_anchors = z_train_t[NON_REF_IDX]


def infer_rul(model, h_target_t):
    model.eval()
    with torch.no_grad():
        z_pred, alpha = model(h_target_t, h_refs_t, z_refs_t)
    return np.exp(z_pred.cpu().numpy() * sigma + mu), alpha.cpu().numpy()


if __name__ == "__main__":
    print(f"cf. V2 solo test={v2_test_rmse:.2f}")
    print(f"cf. original Stage-1 (uniform mean, same delta arch): val=84.90 test=68.41\n")

    torch.manual_seed(0)
    model = AttentionAggModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_rmse, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(EPOCHS):
        model.train()
        z_pred, _ = model(h_train_anchors, h_refs_t, z_refs_t)
        loss = ((z_pred - z_train_anchors) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % EVAL_EVERY == 0 or (epoch + 1) == EPOCHS:
            val_pred, _ = infer_rul(model, h_val_t)
            val_rmse = rmse(val_pred, val_label)
            improved = val_rmse < best_val_rmse
            if improved:
                best_val_rmse, best_epoch = val_rmse, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"[{epoch+1:4d}/{EPOCHS}] loss={loss.item():.4f} val_RMSE={val_rmse:7.2f}"
                  f"{'  <- best val' if improved else ''}", flush=True)
            if (epoch + 1) - best_epoch >= PATIENCE:
                print(f"early stop at epoch {epoch+1}")
                break

    model.load_state_dict(best_state); model.eval()
    val_pred, val_alpha = infer_rul(model, h_val_t)
    test_pred, test_alpha = infer_rul(model, h_test_t)
    val_rmse = rmse(val_pred, val_label)
    test_rmse = rmse(test_pred, test_label)

    print(f"\n=== H4 v2 (attention-agg, original delta arch) ===")
    print(f"val_RMSE={val_rmse:.2f}  test_RMSE={test_rmse:.2f}  (best_epoch={best_epoch})")
    print(f"cf. same delta arch, uniform mean (original Stage-1): val=84.90 test=68.41")
    print(f"cf. V2 solo test={v2_test_rmse:.2f}")
    print(f"H4v2 beats original uniform-mean on val: {val_rmse < 84.90}")
    print(f"mean attention weight per reference (val): {val_alpha.mean(axis=0).round(3).tolist()}")
    print(f"mean attention weight per reference (test): {test_alpha.mean(axis=0).round(3).tolist()}")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(dict(val_rmse=val_rmse, test_rmse=test_rmse, val_alpha=val_alpha, test_alpha=test_alpha,
                          v2_test_rmse=v2_test_rmse, best_epoch=best_epoch), f)
    print(f"\nSaved to {OUT_PKL}")
