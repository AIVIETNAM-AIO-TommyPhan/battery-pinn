"""End-to-end reference: transform -> build model (V1 + V2) -> loss -> Inter-Embedding.

This is a GENERIC, dataset-agnostic walkthrough of the full pipeline using
only src/battery_pinn's extracted building blocks -- it is not tied to any
one dataset's cache paths (unlike scripts/crush/ensemble/run_crush_v1v2inter_tier3_ensemble.py,
which is the concrete, cache-loading, CLI-driven version this was distilled
from and re-verified against, see that script's docstring).

Still depends on the external BatteryML fork for `DataBundle` and the
label-transformation registry (`batteryml.builders.DATA_TRANSFORMATIONS`) --
neither is vendored into this package, see README.md's "Dependencies"
section. Everything else below is pure src/battery_pinn + torch/numpy/scipy.

Expected input shapes (matching every dataset's feature_cache_*.pkl schema
established in reports/day_0906/report.md):
  train_base_feature : [N_train, 6, 100, 1000]  (6-channel ΔQ(V), 100 cycles, 1000 voltage bins)
  train_base_scalar   : [N_train, 3]             (TOP3_FEATURES, already z-scored)
  train_base_label    : [N_train]                (raw RUL, cycles)
  train_base_soh       : [N_train, 100]           (SOH trajectory, cycles 0-99; Tier-1/3 only)
  val_*, test_* : same shapes, N_val / N_test rows
"""
import sys
import os

sys.path.insert(0, r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML")  # <-- your BatteryML checkout
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from batteryml.data.databundle import DataBundle
from batteryml.builders import DATA_TRANSFORMATIONS

from battery_pinn.features.cleaning import clean_feature
from battery_pinn.models.backbone import SmallCNNScalarBranchAxisAwareTier3
from battery_pinn.models.inter_embedding import run_inter_embedding
from battery_pinn.losses.tier3 import tier1_total_loss, tier3_total_loss
from battery_pinn.calibration.ensemble import affine_nnls_ensemble, simple_mean_ensemble
from battery_pinn.utils.metrics import full_metrics, rmse

DIFF_BASE = 9      # cycle index subtracted as the ΔQ(V) reference cycle
CHUNK = 48         # 4GB-GPU-safe forward/backward chunk size
EPOCHS = 1000
EVAL_EVERY = 50
PATIENCE = 200
SEED = 0
UNDER_PENALTY = 1.0   # 1.0 = symmetric MSE; CRUSH is the one dataset that uses 1.4
TIER = "tier3"        # "tier1" or "tier3"
LAMBDA_SOH = 0.01
LAMBDA_MONO = 0.05
LAMBDA_WIENER = 0.05
MONO_TOLERANCE = 0.002

device = "cuda" if torch.cuda.is_available() else "cpu"


# --- 1. TRANSFORM ------------------------------------------------------
# (a) label transform: log-scale then z-score, fit on train_base only,
#     applied identically to val/test at inverse-transform time.
# (b) feature transform: none at the DataBundle level (feature_transformation=None)
#     -- the ΔQ(V) cleaning happens per-chunk inside the training loop instead
#     (see step 3), since it needs the raw feature - reference-cycle diff first.
def build_data_bundle(train_base_feature, train_base_label, test_feature, test_label):
    return DataBundle(
        train_base_feature, train_base_label, test_feature, test_label,
        feature_transformation=None,
        label_transformation=DATA_TRANSFORMATIONS.build({
            "name": "SequentialDataTransformation",
            "transformations": [
                {"name": "LogScaleDataTransformation"},
                {"name": "ZScoreDataTransformation"},
            ],
        }),
    )


# --- 2. BUILD MODEL (V1 + V2) ------------------------------------------
# V1 = use_pe=True (learnable cycle positional encoding); V2 = use_pe=False.
# Same architecture otherwise -- both are instances of the one Tier-3-capable
# class; Tier-1 training just never calls forward(..., return_tier3=True).
def build_v1_v2(n_scalar, n_cycles_soh=100):
    v1 = SmallCNNScalarBranchAxisAwareTier3(n_scalar=n_scalar, use_pe=True, n_cycles_soh=n_cycles_soh).to(device)
    v2 = SmallCNNScalarBranchAxisAwareTier3(n_scalar=n_scalar, use_pe=False, n_cycles_soh=n_cycles_soh).to(device)
    return v1, v2


# --- 3. TRAIN ONE MODEL (feature cleaning -> forward -> loss) ----------
def train_one(model, db, train_base_scalar, train_base_soh, val_feature, val_scalar, val_label,
              test_feature, test_scalar, test_label, tag, tier=TIER):
    import random
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_feature_cpu = db.train_data.feature
    train_label_cpu = db.train_data.label
    n_train = train_feature_cpu.shape[0]
    val_true = val_label.numpy().ravel()
    test_true = test_label.numpy().ravel()

    def predict_on(feature, scalar, return_h=False):
        preds, hs = [], []
        for s in range(0, feature.shape[0], CHUNK):
            feat = feature[s:s + CHUNK].to(device)
            diffed = clean_feature(feat - feat[:, :, [DIFF_BASE]])   # <-- the actual "transform" step
            with torch.no_grad():
                out = model(diffed, scalar[s:s + CHUNK].to(device), return_embedding=return_h)
            pred, h = (out if return_h else (out, None))
            preds.append(pred.cpu())
            if return_h:
                hs.append(h.cpu())
            del feat, diffed
        pred = torch.cat(preds)
        pred_np = db.label_transformation.inverse_transform(pred).numpy().astype(float).ravel()
        return (pred_np, torch.cat(hs).numpy()) if return_h else pred_np

    best_val_rmse, best_val_epoch, best_state = float("inf"), None, None
    for epoch in range(EPOCHS):
        model.train(); optimizer.zero_grad()
        for s in range(0, n_train, CHUNK):
            feat_c = train_feature_cpu[s:s + CHUNK].to(device)
            label_c = train_label_cpu[s:s + CHUNK].to(device)
            scalar_c = train_base_scalar[s:s + CHUNK].to(device)
            diffed_c = clean_feature(feat_c - feat_c[:, :, [DIFF_BASE]])   # transform, again, on the train chunk

            if tier == "tier3":
                soh_c = train_base_soh[s:s + CHUNK].to(device)
                pred_c, soh_pred_c, mu_c, sigma_c = model(diffed_c, scalar_c, return_tier3=True)
                chunk_loss, _ = tier3_total_loss(
                    pred_c, label_c, soh_pred_c, soh_c, mu_c, sigma_c,
                    lambda_soh=LAMBDA_SOH, lambda_mono=LAMBDA_MONO, lambda_wiener=LAMBDA_WIENER,
                    mono_tolerance=MONO_TOLERANCE, under_penalty=UNDER_PENALTY)
            else:  # tier1
                soh_c = train_base_soh[s:s + CHUNK].to(device)
                pred_c, soh_pred_c, _, _ = model(diffed_c, scalar_c, return_tier3=True)
                chunk_loss, _ = tier1_total_loss(
                    pred_c, label_c, soh_pred_c, soh_c,
                    lambda_soh=LAMBDA_SOH, under_penalty=UNDER_PENALTY)

            (chunk_loss * (feat_c.shape[0] / n_train)).backward()
            del feat_c, label_c, scalar_c, diffed_c, pred_c, chunk_loss
        optimizer.step()

        if (epoch + 1) % EVAL_EVERY == 0 or (epoch + 1) == EPOCHS:
            model.eval()
            val_pred = predict_on(val_feature, val_scalar)
            val_rmse = rmse(val_pred, val_true)
            if val_rmse < best_val_rmse:
                best_val_rmse, best_val_epoch = val_rmse, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) - best_val_epoch >= PATIENCE:
                break

    model.load_state_dict(best_state); model.eval()
    val_pred = predict_on(val_feature, val_scalar)
    test_pred = predict_on(test_feature, test_scalar)
    _, h_train = predict_on(train_feature_cpu, train_base_scalar, return_h=True)
    _, h_val = predict_on(val_feature, val_scalar, return_h=True)
    _, h_test = predict_on(test_feature, test_scalar, return_h=True)
    print(f"[{tag}] best_val_epoch={best_val_epoch}  test_RMSE={rmse(test_pred, test_true):.1f}")
    return val_pred, test_pred, h_train, h_val, h_test


# --- 4. FULL PIPELINE: V1 + V2 + INTER-EMBEDDING + ENSEMBLE -------------
def run_pipeline(train_base_feature, train_base_label, train_base_scalar, train_base_soh,
                  val_feature, val_label, val_scalar,
                  test_feature, test_label, test_scalar):
    db = build_data_bundle(train_base_feature, train_base_label, test_feature, test_label)
    v1, v2 = build_v1_v2(n_scalar=train_base_scalar.shape[1])

    val_true = val_label.numpy().ravel()
    test_true = test_label.numpy().ravel()

    v2_val, v2_test, h_train, h_val, h_test = train_one(
        v2, db, train_base_scalar, train_base_soh, val_feature, val_scalar, val_label,
        test_feature, test_scalar, test_label, tag="V2")
    v1_val, v1_test, *_ = train_one(
        v1, db, train_base_scalar, train_base_soh, val_feature, val_scalar, val_label,
        test_feature, test_scalar, test_label, tag="V1")

    # Inter-Embedding: trained on V2's frozen embeddings (project convention --
    # use the non-positionally-encoded backbone's embedding space).
    inter_val, inter_test = run_inter_embedding(
        h_train, h_val, h_test, train_base_label.numpy().ravel(), val_true, device,
        under_penalty=UNDER_PENALTY)

    # Ensemble: try both, report both (NNLS is not a safe default -- see
    # calibration/ensemble.py's module docstring).
    nnls_val, nnls_test, weights = affine_nnls_ensemble(
        [v2_val, v1_val, inter_val], val_true, [v2_test, v1_test, inter_test])
    mean_val, mean_test = simple_mean_ensemble([v2_val, v1_val, inter_val], [v2_test, v1_test, inter_test])

    print("V2 solo:  ", full_metrics(v2_test, test_true))
    print("V1 solo:  ", full_metrics(v1_test, test_true))
    print("Inter:    ", full_metrics(inter_test, test_true))
    print("NNLS ens: ", full_metrics(nnls_test, test_true), "weights=", weights.round(3))
    print("Mean ens: ", full_metrics(mean_test, test_true))
    return dict(v2=(v2_val, v2_test), v1=(v1_val, v1_test), inter=(inter_val, inter_test),
                nnls=(nnls_val, nnls_test, weights), mean=(mean_val, mean_test))


if __name__ == "__main__":
    raise SystemExit(
        "This is a reference implementation, not a runnable CLI -- it has no "
        "feature_cache_*.pkl to load. Call run_pipeline(...) from a script "
        "that loads a real cache (see any scripts/<dataset>/ensemble/run_*.py "
        "for the concrete, dataset-specific version this was distilled from)."
    )
