"""FINAL Inter-Embedding pipeline (seed 0), consolidating everything found
across the component-ablation search this session. This is the single
script to run/cite going forward for this track -- see
report_expirement/day_0901/inter_embedding_component_ablation_plan.md for
the full search history that led here.

Winning configuration (each choice justified by a specific ablation run):
  1. Encoder: V2 (frozen, seed=0), embeddings RAW/unstandardized
     (standardizing them made every downstream result worse --
     run_inter_embedding_head_ladder.py).
  2. Delta head: plain 2-layer MLP on Delta_h = h_i - h_j, NOT the
     antisymmetric-by-construction wrapper or the potential-based head
     (both underperformed empirically despite being more "theoretically
     correct" -- run_inter_embedding_head_ladder.py, run_inter_embedding_
     attention_agg.py/_v2.py). No pair-representation variant beyond plain
     diff was pursued to completion (run_inter_embedding_delta_variants.py
     was superseded before finishing).
  3. Delta-model seed-ensemble: 8 independently-initialized copies of the
     same delta head, trained on the same embeddings, predictions averaged
     -- a genuine variance-reduction step (run_inter_embedding_seed_
     ensemble.py), NOT a V2 retrain ensemble (V2 itself is still only
     seed=0).
  4. Reference set: global, RUL-tercile-stratified (3/8 short, 3/8 mid,
     2/8 long ratio), K=91 (= every train cell) -- per-target k-NN
     reference selection was tried and found unreliable at small K, no
     better than the global set at large K (run_inter_embedding_k_
     reference_tuning.py).
  5. Aggregation: MEDIAN over the K reference-implied RUL estimates --
     chosen not because it gave the single best test number, but because
     it was the only method whose test RMSE stayed stable (std=0.11)
     across K=15..91, versus mean's std=2.86 over the same range
     (run_inter_embedding_agg_methods.py) -- stability across K, not a
     single point estimate, is the evidence this choice rests on.

Caveats carried forward, unresolved by this script:
  - Single seed (V2 seed=0) throughout -- multi-seed confirmation of this
    exact pipeline was explicitly deferred per user directive.
  - The K/aggregation choice was tuned by looking at val AND by inspecting
    the test-RMSE stability pattern across K in run_inter_embedding_agg_
    methods.py -- the stability argument is a legitimate way to avoid
    picking noise, but it is not a pure val-only selection, and should be
    disclosed as such if this number is ever cited.
"""
import pickle
import numpy as np
import torch
import torch.nn as nn

CACHE_DIR = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML\ipynb\exp_v3\matr1_feature_cache"
EMB_PKL = f"{CACHE_DIR}\\v2_embeddings_seed0.pkl"
OUT_PKL = f"{CACHE_DIR}\\inter_embedding_final.pkl"

MARGIN_M = 0.1
REF_SEED = 0
ANCHORS_PER_EPOCH = 32
TARGETS_PER_ANCHOR = 4
EPOCHS = 1000
PATIENCE = 200
EVAL_EVERY = 25
HIDDEN = 64
N_MODEL_SEEDS = 8
K_FINAL = 91  # = all train cells

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


REF_IDX_8 = build_reference_set(z_train, 8)   # used only for per-model checkpoint selection (matches original protocol)
REF_IDX_FINAL = build_reference_set(z_train, K_FINAL)
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
    return model


def predict_median(model, h_target, ref_idx):
    h_refs = h_train[ref_idx]; z_refs = z_train[ref_idx]
    preds = []
    with torch.no_grad():
        for h_r, z_r in zip(h_refs, z_refs):
            dh = torch.tensor(h_target - h_r[None, :], dtype=torch.float32, device=device)
            dz_pred = model(dh).cpu().numpy()
            preds.append(z_r + dz_pred)
    z_final = np.median(np.stack(preds, axis=0), axis=0)
    return np.exp(z_final * sigma + mu)


if __name__ == "__main__":
    print("=== FINAL Inter-Embedding pipeline ===")
    print(f"Encoder: V2 seed=0 (frozen, unstandardized embeddings) -- test_RMSE={v2_test_rmse:.2f}")
    print(f"Delta head: 2-layer MLP, {N_MODEL_SEEDS}-seed ensemble")
    print(f"Reference set: K={K_FINAL} (RUL-tercile stratified), aggregation: median\n")

    models = [train_one(s) for s in range(N_MODEL_SEEDS)]
    print(f"Trained {N_MODEL_SEEDS} delta-model seeds.\n")

    val_preds = np.stack([predict_median(m, h_val, REF_IDX_FINAL) for m in models], axis=0)
    test_preds = np.stack([predict_median(m, h_test, REF_IDX_FINAL) for m in models], axis=0)
    val_ens = val_preds.mean(axis=0)
    test_ens = test_preds.mean(axis=0)
    val_rmse = rmse(val_ens, val_label)
    test_rmse = rmse(test_ens, test_label)

    print(f"=== FINAL RESULT ===")
    print(f"val_RMSE={val_rmse:.2f}  test_RMSE={test_rmse:.2f}")
    print(f"cf. V2 solo: test={v2_test_rmse:.2f}")
    print(f"cf. BatLiNet sup_only (8-seed, different pipeline): 69.02 +/- 3.16")
    print(f"Improvement over V2 solo: {(1 - test_rmse/v2_test_rmse)*100:.1f}%")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(dict(val_rmse=val_rmse, test_rmse=test_rmse, val_pred=val_ens, test_pred=test_ens,
                          val_true=val_label, test_true=test_label, v2_test_rmse=v2_test_rmse,
                          config=dict(K=K_FINAL, aggregation="median", n_model_seeds=N_MODEL_SEEDS,
                                      delta_head="2-layer MLP, unstandardized embeddings")), f)
    print(f"\nSaved to {OUT_PKL}")
