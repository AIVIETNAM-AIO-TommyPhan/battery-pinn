"""K-Fold CV for V1 on SNL standalone (16 valid train cells -- single-split
val is too fragile at this size, per the n=1/n=2 cluster diagnostics: n=1
gave weak +0.419 corr but RMSE=405, n=2 gave strong +0.957 corr but R2=-0.049
because train_base shrank to just 8 cells). 4-fold CV: each fold holds out
4 cells as a genuine val, trains on the other 12, early-stops on that fold's
own val. Final test = official 15-cell SNL test, evaluated by averaging the
4 fold-models' predictions (ensemble), never individually selected by
looking at test -- same protocol as run_hust_v1_kfold.py.
"""
import os
import sys
import pickle

import numpy as np
import torch
import torch.nn as nn

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"
sys.path.insert(0, REPO)

from batteryml.data.databundle import DataBundle
from batteryml.builders import DATA_TRANSFORMATIONS
import BatLiNet  # noqa: F401

CACHE_DIR = os.path.join(REPO, "ipynb", "exp_v3", "snl")
OUT_PKL = os.path.join(CACHE_DIR, "snl_v1_kfold_seed0.pkl")

DIFF_BASE = 9
EPOCHS = 1000
EVAL_EVERY = 25
PATIENCE = 150
TOP3_FEATURES = ["qdlin_diff_std", "voltage_slope_50_90", "voltage_soc_90"]
SEED = 0
K_FOLDS = 4

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}  K={K_FOLDS}", flush=True)


@torch.no_grad()
def smoothing(feature):
    med = feature.median(-1)[0].unsqueeze(-1).expand(*feature.shape)
    med_diff = (feature - med).abs()
    med_diff_std = med_diff.std(-1, keepdim=True).expand(*feature.shape)
    mask = med_diff > 3 * med_diff_std
    feature = feature.clone(); feature[mask] = 0.
    return feature


@torch.no_grad()
def _remove_glitches(x, width=25, threshold=3):
    left = torch.roll(x, shifts=1, dims=-1); right = torch.roll(x, shifts=-1, dims=-1)
    diff_left = (left - x).abs(); diff_right = (right - x).abs()
    nsl = diff_left > diff_left.std(-1, keepdim=True) * threshold
    nsr = diff_right > diff_right.std(-1, keepdim=True) * threshold
    for _ in range(width):
        nsl = nsl | torch.roll(nsl, shifts=1, dims=-1)
        nsr = nsr | torch.roll(nsr, shifts=-1, dims=-1)
    to_smooth = nsl & nsr
    x = x.clone(); x[to_smooth] = 0.
    return x


@torch.no_grad()
def filter_cycles(feature, enable=True):
    if not enable:
        return feature
    feature = feature.clone()
    max_val = feature.abs().amax(-1)
    mvd = (max_val - max_val.median(-1, keepdim=True)[0]).abs()
    mask = mvd > mvd.std(-1, keepdim=True) * 5
    mean_val = feature.mean(-1)
    mnd = (mean_val - mean_val.median(-1, keepdim=True)[0]).abs()
    mask |= mnd > mnd.std(-1, keepdim=True) * 5
    feature[mask] = 0.
    return feature


@torch.no_grad()
def clean_feature(diffed_feature, num_edge=50, filter_cycles_enable=True):
    feature = diffed_feature.clone()
    feature[..., :num_edge] = smoothing(feature[..., :num_edge])
    feature[..., -num_edge:] = smoothing(feature[..., -num_edge:])
    feature = _remove_glitches(feature)
    feature = filter_cycles(feature, enable=filter_cycles_enable)
    return feature


class LearnableCyclePositionalEncoding(nn.Module):
    def __init__(self, num_cycles=50, d_model=32):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, num_cycles, d_model))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    def forward(self, x):
        return x + self.pos


class CycleAttentionPooling(nn.Module):
    def __init__(self, d_model=32, hidden_dim=16):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        logits = self.score(x)
        weights = torch.softmax(logits, dim=1)
        pooled = (weights * x).sum(dim=1)
        return pooled, weights.squeeze(-1)


class SmallCNNScalarBranchAxisAware(nn.Module):
    def __init__(self, n_scalar, use_pe: bool):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.pe = LearnableCyclePositionalEncoding(num_cycles=50, d_model=32) if use_pe else None
        self.pool = CycleAttentionPooling(d_model=32, hidden_dim=16)
        self.scalar = nn.Sequential(nn.Linear(n_scalar, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
        self.head = nn.Linear(32 + 16, 1)

    def forward(self, x, scalar):
        conv3_out = self.backbone(x)
        cycle_tokens = conv3_out.mean(dim=-1).transpose(1, 2)
        if self.pe is not None:
            cycle_tokens = self.pe(cycle_tokens)
        signal_embedding, attn = self.pool(cycle_tokens)
        scalar_embedding = self.scalar(scalar)
        out = self.head(torch.cat([signal_embedding, scalar_embedding], dim=1)).squeeze(1)
        return out


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def train_one_fold(fold_idx, train_ids, val_ids, test_ids, lookup, feature_names):
    idxs3 = [feature_names.index(w) for w in TOP3_FEATURES]

    def stack(ids):
        feats = torch.stack([lookup[i][0] for i in ids])
        labels = torch.stack([lookup[i][1] for i in ids])
        scalars_raw = torch.stack([lookup[i][2] for i in ids])
        return feats, labels, scalars_raw

    train_feature, train_label, train_scalar_raw = stack(train_ids)
    val_feature, val_label, val_scalar_raw = stack(val_ids)
    test_feature, test_label, test_scalar_raw = stack(test_ids)

    _mean = torch.nan_to_num(train_scalar_raw.nanmean(dim=0), nan=0.0)
    train_f = torch.where(torch.isnan(train_scalar_raw), _mean.expand_as(train_scalar_raw), train_scalar_raw)
    val_f = torch.where(torch.isnan(val_scalar_raw), _mean.expand_as(val_scalar_raw), val_scalar_raw)
    test_f = torch.where(torch.isnan(test_scalar_raw), _mean.expand_as(test_scalar_raw), test_scalar_raw)
    _std = train_f.std(dim=0).clamp_min(1e-6)
    train_scalar = (train_f - _mean) / _std
    val_scalar = (val_f - _mean) / _std
    test_scalar = (test_f - _mean) / _std

    tr_s, va_s, te_s = train_scalar[:, idxs3], val_scalar[:, idxs3], test_scalar[:, idxs3]

    import random
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

    db = DataBundle(train_feature, train_label, test_feature, test_label,
                     feature_transformation=None,
                     label_transformation=DATA_TRANSFORMATIONS.build(
                         {"name": "SequentialDataTransformation", "transformations": [
                             {"name": "LogScaleDataTransformation"}, {"name": "ZScoreDataTransformation"}]}))
    model = SmallCNNScalarBranchAxisAware(n_scalar=tr_s.shape[1], use_pe=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_feature_cpu = db.train_data.feature
    train_label_cpu = db.train_data.label
    n_train = train_feature_cpu.shape[0]

    def predict_on(feature, scalar):
        feat = feature.to(device)
        diffed_e = clean_feature(feat - feat[:, :, [DIFF_BASE]])
        with torch.no_grad():
            pred = model(diffed_e, scalar.to(device)).cpu()
        return db.label_transformation.inverse_transform(pred).numpy().astype(float).ravel()

    val_true = val_label.numpy().ravel()
    test_true = test_label.numpy().ravel()
    best_val_rmse, best_val_epoch, best_state = float("inf"), None, None

    for epoch in range(EPOCHS):
        model.train(); optimizer.zero_grad()
        feat_c = train_feature_cpu.to(device)
        label_c = train_label_cpu.to(device)
        scalar_c = tr_s.to(device)
        diffed_c = clean_feature(feat_c - feat_c[:, :, [DIFF_BASE]])
        pred_c = model(diffed_c, scalar_c)
        loss = ((pred_c - label_c) ** 2).mean()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % EVAL_EVERY == 0 or (epoch + 1) == EPOCHS:
            model.eval()
            val_pred = predict_on(val_feature, va_s)
            val_rmse = rmse(val_pred, val_true)
            improved = val_rmse < best_val_rmse
            if improved:
                best_val_rmse, best_val_epoch = val_rmse, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            test_rmse_demo = rmse(predict_on(test_feature, te_s), test_true)
            print(f"  [fold{fold_idx}] [{epoch+1:4d}/{EPOCHS}] val_RMSE={val_rmse:7.2f}  test_RMSE={test_rmse_demo:7.2f}"
                  f"{'  <- best val' if improved else ''}", flush=True)
            if best_val_epoch is not None and (epoch + 1) - best_val_epoch >= PATIENCE:
                print(f"  [fold{fold_idx}] early stop at epoch {epoch+1}", flush=True)
                break

    model.load_state_dict(best_state); model.eval()
    test_pred = predict_on(test_feature, te_s)
    val_pred_final = predict_on(val_feature, va_s)
    print(f"[fold{fold_idx}] FINAL best_val_epoch={best_val_epoch}  val_RMSE={best_val_rmse:.2f}  "
          f"test_RMSE={rmse(test_pred, test_true):.2f}\n", flush=True)
    return dict(best_val_epoch=best_val_epoch, val_rmse=best_val_rmse, val_pred=val_pred_final, val_true=val_true,
                test_pred=test_pred, test_true=test_true)


if __name__ == "__main__":
    with open(os.path.join(CACHE_DIR, "feature_cache_snl_base.pkl"), "rb") as f:
        B = pickle.load(f)
    all_label = B["train_pool_label"].numpy().ravel()
    valid_mask = ~np.isnan(all_label)
    train_ids_all = [i for i, m in zip(B["train_pool_ids"], valid_mask) if m]

    lookup = {}
    for i, cid in enumerate(B["train_pool_ids"]):
        if valid_mask[i]:
            lookup[cid] = (B["train_pool_feature"][i], B["train_pool_label"][i], B["train_pool_scalar_raw"][i])

    test_label = B["test_label"].numpy().ravel()
    valid_test_mask = ~np.isnan(test_label)
    test_ids = [i for i, m in zip(B["test_ids"], valid_test_mask) if m]
    for i, cid in enumerate(B["test_ids"]):
        if valid_test_mask[i]:
            lookup[cid] = (B["test_feature"][i], B["test_label"][i], B["test_scalar_raw"][i])

    feature_names = B["feature_names"]
    print(f"train_pool={len(train_ids_all)}  test={len(test_ids)}", flush=True)

    rng = np.random.RandomState(SEED)
    shuffled = list(train_ids_all)
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, K_FOLDS)

    all_results = {}
    for k in range(K_FOLDS):
        val_fold = list(folds[k])
        train_fold = [c for c in train_ids_all if c not in val_fold]
        print(f"\n=== Fold {k}: train={len(train_fold)}  val={len(val_fold)} ===", flush=True)
        res = train_one_fold(k, train_fold, val_fold, test_ids, lookup, feature_names)
        all_results[k] = res
        with open(OUT_PKL, "wb") as f:
            pickle.dump(all_results, f)

    print("\n=== K-FOLD SUMMARY ===")
    test_preds = np.stack([all_results[k]["test_pred"] for k in range(K_FOLDS)], axis=0)
    test_true = all_results[0]["test_true"]
    ens_pred = test_preds.mean(axis=0)
    for k in range(K_FOLDS):
        print(f"fold{k}: val_RMSE={all_results[k]['val_rmse']:.2f}  test_RMSE={rmse(all_results[k]['test_pred'], test_true):.2f}  "
              f"best_epoch={all_results[k]['best_val_epoch']}")
    fold_test_rmses = np.array([rmse(all_results[k]['test_pred'], test_true) for k in range(K_FOLDS)])
    print(f"\nmean individual fold test_RMSE = {fold_test_rmses.mean():.2f} +/- {fold_test_rmses.std():.2f}")
    print(f"4-fold ENSEMBLE test_RMSE = {rmse(ens_pred, test_true):.2f}")
