"""HUST -- SmallCNN pattern screen (V1/V2), generic single-model trainer.

Copy of ../matr1_feature_cache/run_matr2_smallcnn_screen.py, repointed at this
folder's CACHE_DIR so it can run standalone against any feature_cache_hust*.pkl
cache built here. Logic is dataset-agnostic -- CACHE_NAME selects which
feature_cache_*.pkl to load, from this folder only. Kept in sync with the
MATR2 original in the sibling folder; if the MATR2 version's architecture or
training loop changes, port the change here too.

CLI: python run_hust_smallcnn_screen.py <cache_name> [V1|V2] [seed]
  e.g. python run_hust_smallcnn_screen.py feature_cache_hust_n2_matr20 V1 0

Architecture: SmallCNNScalarBranchAxisAware, attention pooling over cycles,
top-3 pre-declared scalar features (qdlin_diff_std, voltage_slope_50_90,
voltage_soc_90). use_pe=True -> "V1", use_pe=False -> "V2".
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

CACHE_DIR = os.path.join(REPO, "ipynb", "exp_v3", "hust")
CACHE_NAME = sys.argv[1] if len(sys.argv) > 1 else "feature_cache_hust_n2_matr20"
USE_PE = (len(sys.argv) > 2 and sys.argv[2] == "V1")  # V1=use_pe=True, default/V2=use_pe=False
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
FEATURE_CACHE_PKL = os.path.join(CACHE_DIR, f"{CACHE_NAME}.pkl")
_tag = "V1" if USE_PE else "V2"
OUT_PKL = os.path.join(CACHE_DIR, f"{CACHE_NAME}_{_tag}_screen_seed{SEED}.pkl")

DIFF_BASE = 9
EPOCHS = 1000
EVAL_EVERY = 50
PATIENCE = 200
TOP3_FEATURES = ["qdlin_diff_std", "voltage_slope_50_90", "voltage_soc_90"]
CHUNK = 48  # 4GB-GPU-safe forward/backward chunk size (full-batch OOMs above ~90-100 cells)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}  diff_base={DIFF_BASE}  epochs={EPOCHS}  eval_every={EVAL_EVERY}", flush=True)


@torch.no_grad()
def smoothing(feature):
    med = feature.median(-1)[0].unsqueeze(-1).expand(*feature.shape)
    med_diff = (feature - med).abs()
    med_diff_std = med_diff.std(-1, keepdim=True).expand(*feature.shape)
    mask = med_diff > 3 * med_diff_std
    feature = feature.clone()
    feature[mask] = 0.
    return feature


@torch.no_grad()
def _remove_glitches(x, width=25, threshold=3):
    left = torch.roll(x, shifts=1, dims=-1)
    right = torch.roll(x, shifts=-1, dims=-1)
    diff_left = (left - x).abs()
    diff_right = (right - x).abs()
    non_smooth_left = diff_left > diff_left.std(-1, keepdim=True) * threshold
    non_smooth_right = diff_right > diff_right.std(-1, keepdim=True) * threshold
    for _ in range(width):
        non_smooth_left = non_smooth_left | torch.roll(non_smooth_left, shifts=1, dims=-1)
        non_smooth_right = non_smooth_right | torch.roll(non_smooth_right, shifts=-1, dims=-1)
    to_smooth = non_smooth_left & non_smooth_right
    x = x.clone()
    x[to_smooth] = 0.
    return x


@torch.no_grad()
def filter_cycles(feature, enable=True):
    if not enable:
        return feature
    feature = feature.clone()
    max_val = feature.abs().amax(-1)
    max_val_diff = (max_val - max_val.median(-1, keepdim=True)[0]).abs()
    mask = max_val_diff > max_val_diff.std(-1, keepdim=True) * 5
    mean_val = feature.mean(-1)
    mean_val_diff = (mean_val - mean_val.median(-1, keepdim=True)[0]).abs()
    mask |= mean_val_diff > mean_val_diff.std(-1, keepdim=True) * 5
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
        assert x.ndim == 3 and x.shape[1:] == self.pos.shape[1:]
        return x + self.pos


class CycleAttentionPooling(nn.Module):
    def __init__(self, d_model=32, hidden_dim=16):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        logits = self.score(x)
        weights = torch.softmax(logits, dim=1)
        pooled = (weights * x).sum(dim=1)
        return pooled, weights.squeeze(-1)


class SmallCNNScalarBranchAxisAware(nn.Module):
    def __init__(self, n_scalar, use_pe: bool, aggregation: str):
        super().__init__()
        assert aggregation == "attention"
        self.aggregation = aggregation
        self.backbone = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.pe = LearnableCyclePositionalEncoding(num_cycles=50, d_model=32) if use_pe else None
        self.pool = CycleAttentionPooling(d_model=32, hidden_dim=16)
        self.scalar = nn.Sequential(
            nn.Linear(n_scalar, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
        )
        self.head = nn.Linear(32 + 16, 1)

    def forward(self, x, scalar):
        conv3_out = self.backbone(x)
        assert conv3_out.shape[1] == 32 and conv3_out.shape[2] == 50
        cycle_tokens = conv3_out.mean(dim=-1).transpose(1, 2)
        if self.pe is not None:
            cycle_tokens = self.pe(cycle_tokens)
        signal_embedding, attn = self.pool(cycle_tokens)
        scalar_embedding = self.scalar(scalar)
        out = self.head(torch.cat([signal_embedding, scalar_embedding], dim=1))
        return out.squeeze(1)


def compute_full_metrics(pred, true):
    mae = float(np.mean(np.abs(pred - true)))
    rmse_ = float(np.sqrt(np.mean((pred - true) ** 2)))
    mape = float(np.mean(np.abs((pred - true) / true))) * 100
    ss_res = np.sum((pred - true) ** 2); ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return dict(rmse=rmse_, mae=mae, mape=mape, r2=r2)


if __name__ == "__main__":
    with open(FEATURE_CACHE_PKL, "rb") as f:
        _C = pickle.load(f)
    train_base_feature = _C["train_base_feature"]; train_base_label = _C["train_base_label"]
    val_feature = _C["val_feature"]; val_label = _C["val_label"]
    test_feature = _C["test_feature"]; test_label = _C["test_label"]
    train_base_scalar = _C["train_base_scalar"]; val_scalar = _C["val_scalar"]; test_scalar = _C["test_scalar"]
    ALL_FEATURE_NAMES = _C["feature_names"]

    _idxs = [ALL_FEATURE_NAMES.index(w) for w in TOP3_FEATURES]
    tr_s = train_base_scalar[:, _idxs]
    va_s = val_scalar[:, _idxs]
    te_s = test_scalar[:, _idxs]

    import random
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

    db = DataBundle(train_base_feature, train_base_label, test_feature, test_label,
                     feature_transformation=None,
                     label_transformation=DATA_TRANSFORMATIONS.build(
                         {"name": "SequentialDataTransformation", "transformations": [
                             {"name": "LogScaleDataTransformation"}, {"name": "ZScoreDataTransformation"}]}))
    model = SmallCNNScalarBranchAxisAware(n_scalar=tr_s.shape[1], use_pe=USE_PE, aggregation="attention").to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"n_parameters={n_params}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # keep the big tensors on CPU; only a CHUNK-sized slice goes to GPU at a time --
    # full-batch forward+backward OOMs the 4GB card above ~90-100 cells.
    train_feature_cpu = db.train_data.feature
    train_label_cpu = db.train_data.label
    n_train = train_feature_cpu.shape[0]

    def predict_on(feature, scalar):
        preds = []
        for s in range(0, feature.shape[0], CHUNK):
            feat = feature[s:s+CHUNK].to(device)
            diffed_e = clean_feature(feat - feat[:, :, [DIFF_BASE]])
            with torch.no_grad():
                pred = model(diffed_e, scalar[s:s+CHUNK].to(device)).cpu()
            preds.append(pred)
            del feat, diffed_e
            if device == "cuda":
                torch.cuda.empty_cache()
        pred = torch.cat(preds)
        return db.label_transformation.inverse_transform(pred).numpy().astype(float).ravel()

    val_true = val_label.numpy().ravel()
    test_true_np = test_label.numpy().ravel()
    best_val_rmse, best_val_epoch = float("inf"), None
    best_val_pred, best_test_pred = None, None
    history = []

    for epoch in range(EPOCHS):
        model.train(); optimizer.zero_grad()
        loss_sum = 0.0
        for s in range(0, n_train, CHUNK):
            feat_c = train_feature_cpu[s:s+CHUNK].to(device)
            label_c = train_label_cpu[s:s+CHUNK].to(device)
            scalar_c = tr_s[s:s+CHUNK].to(device)
            diffed_c = clean_feature(feat_c - feat_c[:, :, [DIFF_BASE]])
            pred_c = model(diffed_c, scalar_c)
            chunk_loss = ((pred_c - label_c) ** 2).mean() * (feat_c.shape[0] / n_train)
            chunk_loss.backward()
            loss_sum += chunk_loss.item()
            del feat_c, label_c, scalar_c, diffed_c, pred_c, chunk_loss
        optimizer.step()
        if device == "cuda":
            torch.cuda.empty_cache()
        if (epoch + 1) % EVAL_EVERY == 0 or (epoch + 1) == EPOCHS:
            model.eval()
            val_pred = predict_on(val_feature, va_s)
            test_pred = predict_on(test_feature, te_s)
            train_pred = predict_on(train_base_feature, tr_s)
            train_rmse = float(np.sqrt(np.mean((train_base_label.numpy().ravel() - train_pred) ** 2)))
            val_rmse = float(np.sqrt(np.mean((val_true - val_pred) ** 2)))
            test_rmse = float(np.sqrt(np.mean((test_true_np - test_pred) ** 2)))
            improved = val_rmse < best_val_rmse
            if improved:
                best_val_rmse, best_val_epoch = val_rmse, epoch + 1
                best_val_pred, best_test_pred = val_pred.copy(), test_pred.copy()
            history.append(dict(epoch=epoch + 1, train_rmse=train_rmse, val_rmse=val_rmse, test_rmse=test_rmse))
            if len(history) >= 3:
                v = np.array([h["val_rmse"] for h in history])
                t = np.array([h["test_rmse"] for h in history])
                running_corr = float(np.corrcoef(v, t)[0, 1])
            else:
                running_corr = float("nan")
            history[-1]["val_test_corr_running"] = running_corr
            print(f"[{epoch+1:4d}/{EPOCHS}] loss={loss_sum:.4f} "
                  f"train_RMSE={train_rmse:7.2f}  val_RMSE={val_rmse:7.2f}  test_RMSE={test_rmse:7.2f}"
                  f"  val~test_corr(running)={running_corr:+.3f}"
                  f"{'  <- best val' if improved else ''}", flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
            if best_val_epoch is not None and (epoch + 1) - best_val_epoch >= PATIENCE:
                print(f"early stop at epoch {epoch+1} (no val improvement for "
                      f"{(epoch+1)-best_val_epoch} epochs)", flush=True)
                break

    v_all = np.array([h["val_rmse"] for h in history])
    t_all = np.array([h["test_rmse"] for h in history])
    final_corr = float(np.corrcoef(v_all, t_all)[0, 1])
    with open(OUT_PKL, "wb") as f:
        pickle.dump(dict(history=history, best_val_epoch=best_val_epoch, best_val_rmse=best_val_rmse,
                          n_parameters=n_params, val_test_corr=final_corr,
                          val_ids=_C["val_ids"], test_ids=_C["test_ids"],
                          val_true=val_true, test_true=test_true_np,
                          best_val_pred=best_val_pred, best_test_pred=best_test_pred), f)
    print(f"\nDone. best_val_epoch={best_val_epoch}  best_val_rmse={best_val_rmse:.2f}  "
          f"val~test_corr(full run)={final_corr:+.3f}")
    print(f"Saved -> {OUT_PKL}")
