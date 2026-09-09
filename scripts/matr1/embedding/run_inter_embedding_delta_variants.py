"""Pair-representation ablation for the Inter-Embedding Model, on cached V2
embeddings (extract_v2_embeddings_seed0.py) -- no CNN retrain per variant.

REVISED from the first draft of this script: the original version
standardized embeddings (train-only z-score) and wrapped every head in an
antisymmetric construction (0.5*(g(x_ij)-g(x_ji))). Both of those choices
were later found to make results WORSE, independent of representation
choice (run_inter_embedding_head_ladder.py: standardized+antisymmetric H0
only reached val=87.93/test=76.96, vs. the raw/plain-MLP original's
val=84.90/test=68.41). Running the old version of this script would have
confounded "does representation matter" with "standardization+antisymmetry
hurts" -- already answered separately. This version isolates representation
choice alone: RAW/unstandardized embeddings, plain (non-antisymmetric)
2-layer MLP, matching the actual winning recipe
(run_inter_embedding_final.py) in every other respect.

Variants (dim shown for a 48-dim embedding):
  diff           : h_i - h_j                      (48)  -- the established baseline
  concat         : [h_i, h_j]                      (96)
  concat_diff    : [h_i, h_j, h_i-h_j]              (144)
  absdiff_concat : [h_i, h_j, |h_i-h_j|]            (144)
  product        : [h_i, h_j, h_i*h_j]              (144)
  sqdiff_concat  : [h_i, h_j, (h_i-h_j)^2]          (144)

Each trained as a single model (K=8 mean checkpoint selection, matching the
original single-model protocol) for direct comparability with the
established diff baseline (val=84.90, test=68.41). Selection: pick the
representation with the lowest VAL RMSE; test revealed only for the winner.
"""
import pickle
import numpy as np
import torch
import torch.nn as nn

CACHE_DIR = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML\ipynb\exp_v3\matr1_feature_cache"
EMB_PKL = f"{CACHE_DIR}\\v2_embeddings_seed0.pkl"
OUT_PKL = f"{CACHE_DIR}\\inter_embedding_delta_variants.pkl"

MARGIN_M = 0.1
REF_SEED = 0
ANCHORS_PER_EPOCH = 32
TARGETS_PER_ANCHOR = 4
EPOCHS = 1000
PATIENCE = 200
EVAL_EVERY = 25
HIDDEN = 64

device = "cuda" if torch.cuda.is_available() else "cpu"

VARIANTS = ["diff", "concat", "concat_diff", "absdiff_concat", "product", "sqdiff_concat"]
DIMS = {"diff": 48, "concat": 96, "concat_diff": 144, "absdiff_concat": 144, "product": 144, "sqdiff_concat": 144}


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def combine(name, hi, hj):
    if name == "diff":
        return hi - hj
    if name == "concat":
        return np.concatenate([hi, hj], axis=-1)
    if name == "concat_diff":
        return np.concatenate([hi, hj, hi - hj], axis=-1)
    if name == "absdiff_concat":
        return np.concatenate([hi, hj, np.abs(hi - hj)], axis=-1)
    if name == "product":
        return np.concatenate([hi, hj, hi * hj], axis=-1)
    if name == "sqdiff_concat":
        return np.concatenate([hi, hj, (hi - hj) ** 2], axis=-1)
    raise ValueError(name)


class PlainMLP(nn.Module):
    """Same family as the established winning delta head: plain 2-layer
    MLP, no antisymmetric wrapper, no standardization upstream."""

    def __init__(self, dim, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


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
print(f"Fixed reference cells: {REF_IDX.tolist()}")


def run_variant(name, seed=0):
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

    h_refs = h_train[REF_IDX]; z_refs = z_train[REF_IDX]

    def infer(h_target):
        preds = []
        with torch.no_grad():
            for h_r, z_r in zip(h_refs, z_refs):
                x = combine(name, h_target, np.broadcast_to(h_r, h_target.shape))
                x_t = torch.tensor(x, dtype=torch.float32, device=device)
                dz_pred = model(x_t).cpu().numpy()
                preds.append(z_r + dz_pred)
        z_final = np.mean(np.stack(preds, axis=0), axis=0)
        return np.exp(z_final * sigma + mu)

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
            val_pred = infer(h_val)
            val_rmse = rmse(val_pred, val_label)
            improved = val_rmse < best_val_rmse
            if improved:
                best_val_rmse, best_epoch = val_rmse, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) - best_epoch >= PATIENCE:
                break

    model.load_state_dict(best_state); model.eval()
    val_pred = infer(h_val)
    test_pred = infer(h_test)
    val_rmse = rmse(val_pred, val_label)
    test_rmse = rmse(test_pred, test_label)
    print(f"[{name:16s}] dim={dim:3d} best_epoch={best_epoch:4d}  val_RMSE={val_rmse:8.2f}  test_RMSE={test_rmse:8.2f}")
    return dict(val_rmse=val_rmse, test_rmse=test_rmse, val_pred=val_pred, test_pred=test_pred, best_epoch=best_epoch)


if __name__ == "__main__":
    print(f"cf. V2 solo test_RMSE={v2_test_rmse:.2f}")
    print(f"cf. established 'diff' baseline (single model, K=8 mean): val=84.90 test=68.41\n")

    results = {}
    for name in VARIANTS:
        results[name] = run_variant(name)

    winner = min(results.keys(), key=lambda k: results[k]["val_rmse"])
    print(f"\nWinner by VAL: {winner}  val_RMSE={results[winner]['val_rmse']:.2f}  test_RMSE={results[winner]['test_rmse']:.2f}")

    print("\n=== Full comparison (val AND test both shown -- single-model screen, not the final ensembled pipeline) ===")
    for name, r in results.items():
        marker = " <- WINNER" if name == winner else ""
        print(f"{name:16s} val={r['val_rmse']:8.2f} test={r['test_rmse']:8.2f}{marker}")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(dict(results={k: {kk: vv for kk, vv in v.items() if kk not in ("val_pred", "test_pred")}
                                   for k, v in results.items()},
                          winner=winner, v2_test_rmse=v2_test_rmse), f)
    print(f"\nSaved to {OUT_PKL}")
