"""Inter-Cell Embedding delta-model: learns pairwise RUL differences from
frozen CNN embeddings, then infers by median-aggregating over a reference
set. Extracted from scripts/crush/ensemble/run_crush_v1v2inter_tier3_ensemble.py.

under_penalty defaults to 1.0 (symmetric MSE on the pairwise delta) --
CRUSH's script passes 1.4 to match its asymmetric RUL loss; other datasets'
Inter-Embedding scripts may use plain MSE (under_penalty=1.0) -- check the
calling script before assuming otherwise.
"""
import numpy as np
import torch
import torch.nn as nn


class InterEmbeddingModel(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, delta_h):
        return self.net(delta_h).squeeze(-1)


def build_reference_set(z, K, seed=0):
    """Stratified short/mid/long-life reference draw (3/8, 3/8, 2/8 split)."""
    n = len(z)
    if K >= n:
        return np.arange(n)
    order = np.argsort(z)
    short_pool = order[: n // 3]
    mid_pool = order[n // 3: 2 * n // 3]
    long_pool = order[2 * n // 3:]
    rng = np.random.RandomState(seed)
    k_short = max(1, round(K * 3 / 8))
    k_long = max(1, round(K * 2 / 8))
    k_mid = max(1, K - k_short - k_long)
    k_short = min(k_short, len(short_pool))
    k_mid = min(k_mid, len(mid_pool))
    k_long = min(k_long, len(long_pool))
    refs = (list(rng.choice(short_pool, size=k_short, replace=False))
            + list(rng.choice(mid_pool, size=k_mid, replace=False))
            + list(rng.choice(long_pool, size=k_long, replace=False)))
    return np.array(refs)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run_inter_embedding(h_train, h_val, h_test, train_label_np, val_label_np, device,
                         under_penalty=1.0, margin_m=0.1, ref_seed=0,
                         anchors_per_epoch=32, targets_per_anchor=4,
                         epochs=1000, patience=200, eval_every=25,
                         hidden=64, n_delta_seeds=8):
    log_train = np.log(train_label_np)
    mu, sigma = log_train.mean(), log_train.std()
    z_train = (log_train - mu) / sigma
    n_train = len(z_train)
    dim = h_train.shape[1]
    K_final = n_train

    ref_idx_8 = build_reference_set(z_train, min(8, n_train), seed=ref_seed)
    ref_idx_final = build_reference_set(z_train, K_final, seed=ref_seed)
    h_train_t = torch.tensor(h_train, dtype=torch.float32, device=device)
    z_train_t = torch.tensor(z_train, dtype=torch.float32, device=device)

    def infer_k8(model, h_target):
        h_refs = h_train[ref_idx_8]
        z_refs = z_train[ref_idx_8]
        preds = []
        with torch.no_grad():
            for h_r, z_r in zip(h_refs, z_refs):
                dh = torch.tensor(h_target - h_r[None, :], dtype=torch.float32, device=device)
                dz_pred = model(dh).cpu().numpy()
                preds.append(z_r + dz_pred)
        return np.exp(np.mean(np.stack(preds, axis=0), axis=0) * sigma + mu)

    def train_one(seed):
        rng = np.random.RandomState(seed)

        def sample_pairs():
            anchors = rng.choice(n_train, size=min(anchors_per_epoch, n_train), replace=False)
            pi, pj = [], []
            for a in anchors:
                cands = [j for j in range(n_train) if j != a and abs(z_train[a] - z_train[j]) > margin_m]
                if not cands:
                    continue
                chosen = rng.choice(cands, size=min(targets_per_anchor, len(cands)), replace=False)
                for j in chosen:
                    pi.append(a)
                    pj.append(j)
            return np.array(pi), np.array(pj)

        torch.manual_seed(seed)
        model = InterEmbeddingModel(dim=dim, hidden=hidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        best_val_rmse, best_epoch, best_state = float("inf"), 0, None
        for epoch in range(epochs):
            model.train()
            pi, pj = sample_pairs()
            dh = h_train_t[pi] - h_train_t[pj]
            dz_true = z_train_t[pi] - z_train_t[pj]
            pred = model(dh)
            diff = pred - dz_true
            weight = torch.where(diff < 0, under_penalty, 1.0)
            loss = (weight * diff ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (epoch + 1) % eval_every == 0 or (epoch + 1) == epochs:
                model.eval()
                val_rmse = rmse(infer_k8(model, h_val), val_label_np)
                if val_rmse < best_val_rmse:
                    best_val_rmse, best_epoch = val_rmse, epoch + 1
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if (epoch + 1) - best_epoch >= patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        return model

    def predict_median(model, h_target, ref_idx):
        h_refs = h_train[ref_idx]
        z_refs = z_train[ref_idx]
        preds = []
        with torch.no_grad():
            for h_r, z_r in zip(h_refs, z_refs):
                dh = torch.tensor(h_target - h_r[None, :], dtype=torch.float32, device=device)
                dz_pred = model(dh).cpu().numpy()
                preds.append(z_r + dz_pred)
        z_final = np.median(np.stack(preds, axis=0), axis=0)
        return np.exp(z_final * sigma + mu)

    models = [train_one(s) for s in range(n_delta_seeds)]
    val_preds = np.stack([predict_median(m, h_val, ref_idx_final) for m in models], axis=0)
    test_preds = np.stack([predict_median(m, h_test, ref_idx_final) for m in models], axis=0)
    return val_preds.mean(axis=0), test_preds.mean(axis=0)
