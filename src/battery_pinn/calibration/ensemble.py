"""Val-calibration + multi-component ensembling.

Every ensemble used across this project's reports (MATR1/HUST/CRUSH/CRUH)
follows the same two-stage shape:
  1. (optional) per-component affine calibration: fit y = a*x + b on
     (component's val prediction, val ground truth), apply to both val and
     test for that component.
  2. combine the (calibrated or raw) components into one prediction, either
     by NNLS-fit weights (minimizing val error) or a plain unweighted mean.

`ENSEMBLE_METHODS` below is the full catalog used in this project. Every
entry takes the SAME two required inputs -- `val_components`, a list of 1-D
np.ndarray val predictions (one per component), and `test_components`, the
matching list of test predictions -- plus `val_true` (1-D np.ndarray) when
the method needs it to calibrate/fit weights. Extracted verbatim from
scripts/crush/ensemble/run_crush_v1v2inter_tier3_ensemble.py; see that
script for the concrete 3-component (V2, V1, Inter-Embedding) call site.

Known caveat (reports/day_0906/report.md, CRUSH section + Error Analysis):
NNLS's extra flexibility (one free weight per component, fit on val) can
overfit a small val set. On CRUSH's original val=15 cache, which of
{affine_nnls, simple_mean} ranked better even reversed between 5 and 8
seeds; on the redesigned val=20 cache, simple_mean beat affine_nnls on both
mean RMSE and stability across all 8 seeds. **Always run both and report
both** rather than assuming NNLS is the stronger default -- it usually
isn't, once val is small enough to matter.
"""
import numpy as np
from scipy.optimize import nnls


def affine_fit(val_pred, val_true, test_pred):
    """Fit y = a*x + b on (val_pred, val_true), apply to both val and test.
    Input: val_pred, val_true, test_pred -- all 1-D np.ndarray, same dtype.
    Output: (calibrated_val_pred, calibrated_test_pred)."""
    A = np.vstack([val_pred, np.ones_like(val_pred)]).T
    a, b = np.linalg.lstsq(A, val_true, rcond=None)[0]
    return a * val_pred + b, a * test_pred + b


def _calibrate_all(val_components, val_true, test_components):
    calibrated = [affine_fit(v, val_true, t) for v, t in zip(val_components, test_components)]
    return [c[0] for c in calibrated], [c[1] for c in calibrated]


def affine_nnls_ensemble(val_components, val_true, test_components):
    """Affine-calibrate each component on val, then NNLS-fit non-negative
    weights (minimizing val error) to combine them.
    Input: val_components/test_components = list[np.ndarray] (raw,
    uncalibrated predictions, one array per component); val_true =
    np.ndarray ground truth.
    Output: (ens_val, ens_test, weights) -- weights has one entry per
    component, in the same order as the input lists."""
    val_cal, test_cal = _calibrate_all(val_components, val_true, test_components)
    V = np.stack(val_cal, axis=1)
    T = np.stack(test_cal, axis=1)
    weights, _ = nnls(V, val_true)
    return V @ weights, T @ weights, weights


def affine_mean_ensemble(val_components, val_true, test_components):
    """Affine-calibrate each component on val, then take the unweighted mean.
    Input/output shapes identical to affine_nnls_ensemble, except no weights
    are returned (implicitly uniform, 1/n_components each)."""
    val_cal, test_cal = _calibrate_all(val_components, val_true, test_components)
    return np.mean(val_cal, axis=0), np.mean(test_cal, axis=0)


def simple_mean_ensemble(val_components, test_components):
    """Unweighted mean of the RAW (uncalibrated) components -- no val_true
    needed. Input: val_components/test_components = list[np.ndarray].
    Output: (ens_val, ens_test)."""
    return np.mean(val_components, axis=0), np.mean(test_components, axis=0)


ENSEMBLE_METHODS = {
    "affine_nnls": dict(
        fn=affine_nnls_ensemble,
        needs_val_true=True,
        description="Per-component affine calibration + NNLS-weighted combination "
                    "(this project's original default; often overfits small val sets).",
        inputs="val_components: list[np.ndarray], val_true: np.ndarray, test_components: list[np.ndarray]",
    ),
    "affine_mean": dict(
        fn=affine_mean_ensemble,
        needs_val_true=True,
        description="Per-component affine calibration + unweighted mean.",
        inputs="val_components: list[np.ndarray], val_true: np.ndarray, test_components: list[np.ndarray]",
    ),
    "simple_mean": dict(
        fn=simple_mean_ensemble,
        needs_val_true=False,
        description="Unweighted mean of raw (uncalibrated) predictions -- no val-based "
                    "fitting at all; the most robust choice found for small val sets "
                    "(CRUSH, 8-seed: 339.82±13.88 vs. affine_nnls's 340.97±21.76).",
        inputs="val_components: list[np.ndarray], test_components: list[np.ndarray]",
    ),
}


def combine(method, val_components, test_components, val_true=None):
    """Dispatch to one of ENSEMBLE_METHODS by name. Raises KeyError with the
    valid method names if `method` is unrecognized."""
    if method not in ENSEMBLE_METHODS:
        raise KeyError(f"unknown ensemble method {method!r}; choose from {list(ENSEMBLE_METHODS)}")
    spec = ENSEMBLE_METHODS[method]
    if spec["needs_val_true"]:
        if val_true is None:
            raise ValueError(f"method {method!r} requires val_true")
        return spec["fn"](val_components, val_true, test_components)
    return spec["fn"](val_components, test_components)
