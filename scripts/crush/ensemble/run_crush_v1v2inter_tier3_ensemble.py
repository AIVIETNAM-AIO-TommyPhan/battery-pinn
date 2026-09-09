"""CRUSH -- V2 + V1 + Inter-Embedding + Affine/NNLS ensemble, full Tier3
variant: SOH auxiliary head + monotonicity + Wiener drift/diffusion (SDE) on
top of the asymmetric RUL loss (UNDER_PENALTY=1.4).

REFACTORED to import the shared model/loss/calibration code from
src/battery_pinn instead of duplicating it inline -- this is the one
pipeline in this release extracted into src/ so far (see
src/battery_pinn/README.md for why the others are not). Verified to
reproduce the original inline script's seed=0 numbers exactly on the
`feature_cache_crush_hust10_calce15_fullval_withsoh_randval_reinforced`
cache: V2=323.24, V1=349.14, Inter-Embedding=317.95, NNLS ensemble=331.53
(reports/day_0906/report.md, CRUSH section).

This script ALSO depends on the microsoft/BatteryML fork (`batteryml`,
`BatLiNet` packages) and the dataset's feature-cache `.pkl`, neither of
which are included in this release -- see README.md. Set REPO/CACHE_DIR
below to your local BatteryML checkout before running.

CLI: python run_crush_v1v2inter_tier3_ensemble.py <cache_name> [seed] [under_penalty]
"""
import os
import sys
import pickle

import numpy as np
import torch

REPO = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML"  # <-- point at your BatteryML checkout
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from batteryml.data.databundle import DataBundle
from batteryml.builders import DATA_TRANSFORMATIONS
import BatLiNet  # noqa: F401

from battery_pinn.features.cleaning import clean_feature
from battery_pinn.models.backbone import SmallCNNScalarBranchAxisAwareTier3
from battery_pinn.models.inter_embedding import run_inter_embedding
from battery_pinn.losses.tier3 import (
    asymmetric_rul_loss, soh_auxiliary_loss, monotonicity_loss, wiener_loss,
)
from battery_pinn.calibration.ensemble import affine_nnls_ensemble
from battery_pinn.utils.metrics import rmse

CACHE_DIR = os.path.join(REPO, "ipynb", "exp_v3", "crush")
CACHE_NAME = sys.argv[1] if len(sys.argv) > 1 else "feature_cache_crush_hust10_calce15_fullval_withsoh"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
UNDER_PENALTY = float(sys.argv[3]) if len(sys.argv) > 3 else 1.4
FEATURE_CACHE_PKL = os.path.join(CACHE_DIR, f"{CACHE_NAME}.pkl")
OUT_PKL = os.path.join(CACHE_DIR, f"{CACHE_NAME}_v1v2inter_tier3_pen{UNDER_PENALTY:.1f}_ensemble_seed{SEED}.pkl")

DIFF_BASE = 9
EPOCHS = 1000
EVAL_EVERY = 50
PATIENCE = 200
CHUNK = 48
TOP3_FEATURES = ["qdlin_diff_std", "voltage_slope_50_90", "voltage_soc_90"]

LAMBDA_SOH = 0.01
LAMBDA_MONO = 0.05
LAMBDA_WIENER = 0.05
MONO_TOLERANCE = 0.002

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}  diff_base={DIFF_BASE}  chunk={CHUNK}  seed={SEED}  pen={UNDER_PENALTY}  tier3", flush=True)


def train_cnn(use_pe, tag):
    import random
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

    db = DataBundle(train_base_feature, train_base_label, test_feature, test_label,
                     feature_transformation=None,
                     label_transformation=DATA_TRANSFORMATIONS.build(
                         {"name": "SequentialDataTransformation", "transformations": [
                             {"name": "LogScaleDataTransformation"}, {"name": "ZScoreDataTransformation"}]}))
    model = SmallCNNScalarBranchAxisAwareTier3(n_scalar=tr_s.shape[1], use_pe=use_pe).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_feature_cpu = db.train_data.feature
    train_label_cpu = db.train_data.label
    train_soh_cpu = train_base_soh
    n_train = train_feature_cpu.shape[0]

    def predict_on(feature, scalar, return_h=False):
        preds, hs = [], []
        for s in range(0, feature.shape[0], CHUNK):
            feat = feature[s:s+CHUNK].to(device)
            diffed_e = clean_feature(feat - feat[:, :, [DIFF_BASE]])
            with torch.no_grad():
                out = model(diffed_e, scalar[s:s+CHUNK].to(device), return_embedding=return_h)
            if return_h:
                pred, h = out
                hs.append(h.cpu())
            else:
                pred = out
            preds.append(pred.cpu())
            del feat, diffed_e
            if device == "cuda":
                torch.cuda.empty_cache()
        pred = torch.cat(preds)
        pred_np = db.label_transformation.inverse_transform(pred).numpy().astype(float).ravel()
        if return_h:
            return pred_np, torch.cat(hs).numpy()
        return pred_np

    val_true = val_label.numpy().ravel()
    test_true_np = test_label.numpy().ravel()
    best_val_rmse, best_val_epoch, best_state = float("inf"), None, None

    for epoch in range(EPOCHS):
        model.train(); optimizer.zero_grad()
        for s in range(0, n_train, CHUNK):
            feat_c = train_feature_cpu[s:s+CHUNK].to(device)
            label_c = train_label_cpu[s:s+CHUNK].to(device)
            scalar_c = tr_s[s:s+CHUNK].to(device)
            soh_c = train_soh_cpu[s:s+CHUNK].to(device)
            diffed_c = clean_feature(feat_c - feat_c[:, :, [DIFF_BASE]])
            pred_c, soh_pred_c, mu_c, sigma_c = model(diffed_c, scalar_c, return_tier3=True)
            loss_rul_c = asymmetric_rul_loss(pred_c, label_c, UNDER_PENALTY)
            loss_soh_c = soh_auxiliary_loss(soh_pred_c, soh_c)
            loss_mono_c = monotonicity_loss(soh_pred_c, MONO_TOLERANCE)
            loss_wiener_c = wiener_loss(soh_pred_c, mu_c, sigma_c)
            chunk_loss = (loss_rul_c + LAMBDA_SOH * loss_soh_c + LAMBDA_MONO * loss_mono_c
                          + LAMBDA_WIENER * loss_wiener_c) * (feat_c.shape[0] / n_train)
            chunk_loss.backward()
            del feat_c, label_c, scalar_c, soh_c, diffed_c, pred_c, soh_pred_c, mu_c, sigma_c, chunk_loss
        optimizer.step()
        if device == "cuda":
            torch.cuda.empty_cache()
        if (epoch + 1) % EVAL_EVERY == 0 or (epoch + 1) == EPOCHS:
            model.eval()
            val_pred = predict_on(val_feature, va_s)
            test_pred = predict_on(test_feature, te_s)
            val_rmse = rmse(val_pred, val_true)
            test_rmse = rmse(test_pred, test_true_np)
            improved = val_rmse < best_val_rmse
            if improved:
                best_val_rmse, best_val_epoch = val_rmse, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  [{tag}] [{epoch+1:4d}/{EPOCHS}] val_RMSE={val_rmse:7.2f}  test_RMSE={test_rmse:7.2f}"
                  f"{'  <- best val' if improved else ''}", flush=True)
            if (epoch + 1) - best_val_epoch >= PATIENCE:
                print(f"  [{tag}] early stop at epoch {epoch+1}", flush=True)
                break

    model.load_state_dict(best_state); model.eval()
    val_pred = predict_on(val_feature, va_s)
    test_pred = predict_on(test_feature, te_s)
    h_train = h_val = h_test = None
    if not use_pe:
        _, h_train = predict_on(train_base_feature, tr_s, return_h=True)
        _, h_val = predict_on(val_feature, va_s, return_h=True)
        _, h_test = predict_on(test_feature, te_s, return_h=True)
    print(f"[{tag}] FINAL test_RMSE={rmse(test_pred, test_true_np):.2f}\n", flush=True)
    return val_pred, test_pred, h_train, h_val, h_test


if __name__ == "__main__":
    with open(FEATURE_CACHE_PKL, "rb") as f:
        _C = pickle.load(f)
    train_base_feature = _C["train_base_feature"]; train_base_label = _C["train_base_label"]
    train_base_soh = _C["train_soh"]
    val_feature = _C["val_feature"]; val_label = _C["val_label"]
    test_feature = _C["test_feature"]; test_label = _C["test_label"]
    train_base_scalar = _C["train_base_scalar"]; val_scalar = _C["val_scalar"]; test_scalar = _C["test_scalar"]
    ALL_FEATURE_NAMES = _C["feature_names"]
    _idxs = [ALL_FEATURE_NAMES.index(w) for w in TOP3_FEATURES]
    tr_s = train_base_scalar[:, _idxs]; va_s = val_scalar[:, _idxs]; te_s = test_scalar[:, _idxs]

    val_true = val_label.numpy().ravel()
    test_true = test_label.numpy().ravel()
    train_true = train_base_label.numpy().ravel()

    print("=== Training V2 (use_pe=False), Tier3 ===", flush=True)
    v2_val, v2_test, h_train, h_val, h_test = train_cnn(use_pe=False, tag="V2")

    print("=== Training V1 (use_pe=True), Tier3 ===", flush=True)
    v1_val, v1_test, _, _, _ = train_cnn(use_pe=True, tag="V1")

    print("=== Training Inter-Embedding (on V2 Tier3 embeddings) ===", flush=True)
    inter_val, inter_test = run_inter_embedding(
        h_train, h_val, h_test, train_true, val_true, device, under_penalty=UNDER_PENALTY)
    print(f"Inter-Embedding: val_RMSE={rmse(inter_val, val_true):.2f}  test_RMSE={rmse(inter_test, test_true):.2f}\n")

    print("=== Affine-calibrate each component on val, then NNLS-ensemble ===", flush=True)
    ens_val, ens_test, w3 = affine_nnls_ensemble(
        [v2_val, v1_val, inter_val], val_true, [v2_test, v1_test, inter_test])

    print(f"\n=== RESULTS (seed={SEED}, Tier3) ===")
    print(f"V2 solo:            val={rmse(v2_val,val_true):.2f}  test={rmse(v2_test,test_true):.2f}")
    print(f"V1 solo:            val={rmse(v1_val,val_true):.2f}  test={rmse(v1_test,test_true):.2f}")
    print(f"Inter-Embedding:    val={rmse(inter_val,val_true):.2f}  test={rmse(inter_test,test_true):.2f}")
    print(f"Affine-3way NNLS:   val={rmse(ens_val,val_true):.2f}  test={rmse(ens_test,test_true):.2f}  weights={w3.round(3)}")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(dict(
            v2_val=v2_val, v2_test=v2_test, v1_val=v1_val, v1_test=v1_test,
            inter_val=inter_val, inter_test=inter_test,
            ens_val=ens_val, ens_test=ens_test, nnls_weights=w3,
            val_true=val_true, test_true=test_true,
        ), f)
    print(f"\nSaved -> {OUT_PKL}")
