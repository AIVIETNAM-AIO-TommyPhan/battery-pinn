"""Retrain V2 (seed=0, patience=350, deterministic -- reproduces the already
confirmed 75.32 test RMSE) ONE more time, this time caching the extracted
48-dim embeddings (train/val/test) to disk so many Inter-Embedding
architecture variants can be tried quickly afterward without repeating the
CNN retrain each time."""
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

CACHE_DIR = os.path.join(REPO, "ipynb", "exp_v3", "matr1_feature_cache")
FEATURE_CACHE_PKL = os.path.join(CACHE_DIR, "feature_cache.pkl")
OUT_PKL = os.path.join(CACHE_DIR, "v2_embeddings_seed0.pkl")

DIFF_BASE = 9
EPOCHS_V2 = 1000
EVAL_EVERY = 50
PATIENCE_V2 = 350
TOP3_FEATURES = ["qdlin_diff_std", "voltage_slope_50_90", "voltage_soc_90"]

device = "cuda" if torch.cuda.is_available() else "cpu"


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


class CycleAttentionPooling(nn.Module):
    def __init__(self, d_model=32, hidden_dim=16):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        logits = self.score(x)
        weights = torch.softmax(logits, dim=1)
        pooled = (weights * x).sum(dim=1)
        return pooled, weights.squeeze(-1)


class V2WithEmbedding(nn.Module):
    def __init__(self, n_scalar):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.pool = CycleAttentionPooling(d_model=32, hidden_dim=16)
        self.scalar = nn.Sequential(nn.Linear(n_scalar, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
        self.head = nn.Linear(32 + 16, 1)

    def forward(self, x, scalar, return_embedding=False):
        conv3_out = self.backbone(x)
        cycle_tokens = conv3_out.mean(dim=-1).transpose(1, 2)
        signal_embedding, attn = self.pool(cycle_tokens)
        scalar_embedding = self.scalar(scalar)
        h = torch.cat([signal_embedding, scalar_embedding], dim=1)
        out = self.head(h).squeeze(1)
        if return_embedding:
            return out, h
        return out


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


if __name__ == "__main__":
    import random
    seed = 0
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

    with open(FEATURE_CACHE_PKL, "rb") as f:
        _C = pickle.load(f)
    train_base_feature = _C["train_base_feature"]; train_base_label = _C["train_base_label"]
    val_feature = _C["val_feature"]; val_label = _C["val_label"]
    test_feature = _C["test_feature"]; test_label = _C["test_label"]
    train_base_scalar = _C["train_base_scalar"]; val_scalar = _C["val_scalar"]; test_scalar = _C["test_scalar"]
    ALL_FEATURE_NAMES = _C["feature_names"]
    _idxs = [ALL_FEATURE_NAMES.index(w) for w in TOP3_FEATURES]
    tr_s = train_base_scalar[:, _idxs]; va_s = val_scalar[:, _idxs]; te_s = test_scalar[:, _idxs]

    db = DataBundle(train_base_feature, train_base_label, test_feature, test_label,
                     feature_transformation=None,
                     label_transformation=DATA_TRANSFORMATIONS.build(
                         {"name": "SequentialDataTransformation", "transformations": [
                             {"name": "LogScaleDataTransformation"}, {"name": "ZScoreDataTransformation"}]}))
    model = V2WithEmbedding(n_scalar=tr_s.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    feat_dev = db.train_data.feature.to(device)
    label_dev = db.train_data.label.to(device)
    scalar_dev = tr_s.to(device)
    diffed = clean_feature(feat_dev - feat_dev[:, :, [DIFF_BASE]])

    def predict_on(feature, scalar):
        feat = feature.to(device)
        diffed_e = clean_feature(feat - feat[:, :, [DIFF_BASE]])
        with torch.no_grad():
            pred = model(diffed_e, scalar.to(device)).cpu()
        return db.label_transformation.inverse_transform(pred).numpy().astype(float).ravel()

    val_true = val_label.numpy().ravel()
    test_true_np = test_label.numpy().ravel()
    best_val_rmse, best_val_epoch, best_state = float("inf"), None, None

    for epoch in range(EPOCHS_V2):
        model.train(); optimizer.zero_grad()
        pred = model(diffed, scalar_dev)
        loss = ((pred - label_dev) ** 2).mean()
        loss.backward(); optimizer.step()
        if (epoch + 1) % EVAL_EVERY == 0 or (epoch + 1) == EPOCHS_V2:
            model.eval()
            val_pred = predict_on(val_feature, va_s)
            test_pred_demo = predict_on(test_feature, te_s)
            val_rmse = rmse(val_pred, val_true)
            test_rmse_demo = rmse(test_pred_demo, test_true_np)
            improved = val_rmse < best_val_rmse
            if improved:
                best_val_rmse, best_val_epoch = val_rmse, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"[{epoch+1:4d}/{EPOCHS_V2}] val_RMSE={val_rmse:7.2f} test_RMSE={test_rmse_demo:7.2f}"
                  f"{'  <- best val' if improved else ''}", flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
            if (epoch + 1) - best_val_epoch >= PATIENCE_V2:
                print(f"early stop at epoch {epoch+1}", flush=True)
                break

    model.load_state_dict(best_state); model.eval()

    def extract(feature, scalar):
        feat = feature.to(device)
        diffed_e = clean_feature(feat - feat[:, :, [DIFF_BASE]])
        with torch.no_grad():
            _, h = model(diffed_e, scalar.to(device), return_embedding=True)
        return h.cpu().numpy()

    h_train = extract(train_base_feature, tr_s)
    h_val = extract(val_feature, va_s)
    h_test = extract(test_feature, te_s)
    v2_test_rmse = rmse(predict_on(test_feature, te_s), test_true_np)
    print(f"\nFinal V2 test_RMSE={v2_test_rmse:.2f}, embeddings: train={h_train.shape} val={h_val.shape} test={h_test.shape}")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(dict(h_train=h_train, h_val=h_val, h_test=h_test,
                          train_label=train_base_label.numpy().ravel(),
                          val_label=val_label.numpy().ravel(), test_label=test_label.numpy().ravel(),
                          v2_val_pred=val_pred, v2_test_pred=test_pred_demo, v2_test_rmse=v2_test_rmse), f)
    print(f"Saved to {OUT_PKL}")
