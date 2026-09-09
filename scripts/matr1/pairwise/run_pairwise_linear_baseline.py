"""Simplest possible pairwise inter-cell model: ridge regression on RAW scalar
feature differences (no embedding, no GPU) -- a deliberately cheap baseline to
establish a floor before trusting the more complex InterEmbeddingModel
(run_inter_embedding_stage1.py). Uses the identical fixed reference-cell
convention (3 short + 3 mid + 2 long RUL tercile, seed=0) so results are
directly comparable.

Two variants: top-3 features (matches the project's selected feature set) and
all 34 candidate features (matches Stage 0's diagnostic upper bound).
"""
import pickle
import numpy as np
from sklearn.linear_model import Ridge

CACHE_DIR = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML\ipynb\exp_v3\matr1_feature_cache"
FEATURE_CACHE_PKL = f"{CACHE_DIR}\\feature_cache.pkl"
OUT_PKL = f"{CACHE_DIR}\\pairwise_linear_baseline.pkl"

MARGIN_M = 0.1
REF_SEED = 0
TOP3_FEATURES = ["qdlin_diff_std", "voltage_slope_50_90", "voltage_soc_90"]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


with open(FEATURE_CACHE_PKL, "rb") as f:
    fc = pickle.load(f)

names = fc["feature_names"]
idx = {n: i for i, n in enumerate(names)}
top3_idx = [idx[n] for n in TOP3_FEATURES]

train_scalar = np.array(fc["train_base_scalar_raw"])
val_scalar = np.array(fc["val_scalar_raw"])
test_scalar = np.array(fc["test_scalar_raw"])
train_label = np.array(fc["train_base_label"]).ravel()
val_label = np.array(fc["val_label"]).ravel()
test_label = np.array(fc["test_label"]).ravel()

log_train = np.log(train_label)
mu, sigma = log_train.mean(), log_train.std()
z_train = (log_train - mu) / sigma


def make_pairs(z):
    n = len(z)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j and abs(z[i] - z[j]) > MARGIN_M]
    return pairs


def fixed_references(z, seed=REF_SEED):
    n = len(z)
    order = np.argsort(z)
    short_pool = order[: n // 3]
    mid_pool = order[n // 3: 2 * n // 3]
    long_pool = order[2 * n // 3:]
    rng = np.random.RandomState(seed)
    refs = list(rng.choice(short_pool, size=3, replace=False)) + \
           list(rng.choice(mid_pool, size=3, replace=False)) + \
           list(rng.choice(long_pool, size=2, replace=False))
    return np.array(refs)


def run_variant(feat_idx, name):
    X_train = train_scalar[:, feat_idx]
    pairs = make_pairs(z_train)
    dX = np.array([X_train[i] - X_train[j] for i, j in pairs])
    dz = np.array([z_train[i] - z_train[j] for i, j in pairs])
    model = Ridge(alpha=1.0)
    model.fit(dX, dz)

    ref_idx = fixed_references(z_train)
    print(f"[{name}] fixed reference cells (train idx): {ref_idx.tolist()}, "
          f"true RUL: {[round(x, 1) for x in train_label[ref_idx]]}")

    X_refs = X_train[ref_idx]
    z_refs = z_train[ref_idx]
    rul_refs = train_label[ref_idx]

    def predict(X_target):
        preds = []
        for x_r, z_r in zip(X_refs, z_refs):
            dx = X_target - x_r[None, :]
            dz_pred = model.predict(dx)
            z_pred_r = z_r + dz_pred
            rul_pred_r = np.exp(z_pred_r * sigma + mu)
            preds.append(rul_pred_r)
        return np.mean(np.stack(preds, axis=0), axis=0)

    val_pred = predict(val_scalar[:, feat_idx])
    test_pred = predict(test_scalar[:, feat_idx])
    val_rmse = rmse(val_pred, val_label)
    test_rmse = rmse(test_pred, test_label)

    # Control 1: RUL-stratified average, same fixed refs, no learning
    baseline_pred_val = np.full(len(val_label), rul_refs.mean())
    baseline_pred_test = np.full(len(test_label), rul_refs.mean())
    baseline_val_rmse = rmse(baseline_pred_val, val_label)
    baseline_test_rmse = rmse(baseline_pred_test, test_label)

    print(f"[{name}] n_pairs={len(pairs)}  ridge in-sample R2 on Delta z = "
          f"{model.score(dX, dz):.4f}")
    print(f"[{name}] pairwise-linear: val_RMSE={val_rmse:.2f} test_RMSE={test_rmse:.2f}")
    print(f"[{name}] RUL-stratified baseline (no learning): val_RMSE={baseline_val_rmse:.2f} test_RMSE={baseline_test_rmse:.2f}")
    print(f"[{name}] beats baseline on val: {val_rmse <= baseline_val_rmse}")
    print()
    return dict(val_pred=val_pred, test_pred=test_pred, val_rmse=val_rmse, test_rmse=test_rmse,
                baseline_val_rmse=baseline_val_rmse, baseline_test_rmse=baseline_test_rmse,
                ref_idx=ref_idx.tolist(), n_pairs=len(pairs))


if __name__ == "__main__":
    results = {}
    results["top3"] = run_variant(top3_idx, "top3")
    results["all34"] = run_variant(list(range(34)), "all34")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved to {OUT_PKL}")
