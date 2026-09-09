"""Feature selection for the pairwise inter-cell task (Delta_feature -> Delta_z),
ranked by VAL RMSE via the same multi-reference inference procedure -- not by
test RMSE, learning directly from the leakage already disclosed in paper Sec
2.2 (top-3 absolute-RUL feature selection was ranked by test there).

Then: try top-k feature combinations, and ensemble the best pairwise config
with the existing Tier-3 (Wiener PINN) model via NNLS fit on val only.
"""
import pickle
import numpy as np
from sklearn.linear_model import Ridge
from scipy.optimize import nnls

CACHE_DIR = r"C:\Users\TommyPhan_Exp\Desktop\course\Kaggle\batteryml\BatteryML\ipynb\exp_v3\matr1_feature_cache"
FEATURE_CACHE_PKL = f"{CACHE_DIR}\\feature_cache.pkl"
PINN_TIERS_PKL = f"{CACHE_DIR}\\v2_pinn_tiers_seed0.pkl"
OUT_PKL = f"{CACHE_DIR}\\pairwise_feature_selection.pkl"

MARGIN_M = 0.1
REF_SEED = 0


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


with open(FEATURE_CACHE_PKL, "rb") as f:
    fc = pickle.load(f)

names = fc["feature_names"]
train_scalar = np.array(fc["train_base_scalar_raw"])
val_scalar = np.array(fc["val_scalar_raw"])
test_scalar = np.array(fc["test_scalar_raw"])
train_label = np.array(fc["train_base_label"]).ravel()
val_label = np.array(fc["val_label"]).ravel()
test_label = np.array(fc["test_label"]).ravel()

log_train = np.log(train_label)
mu, sigma = log_train.mean(), log_train.std()
z_train = (log_train - mu) / sigma
n_train = len(z_train)


def make_pairs(z):
    return [(i, j) for i in range(len(z)) for j in range(len(z)) if i != j and abs(z[i] - z[j]) > MARGIN_M]


PAIRS = make_pairs(z_train)


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


REF_IDX = fixed_references(z_train)
REF_RUL = train_label[REF_IDX]


def eval_feature_subset(feat_idx):
    X_train = train_scalar[:, feat_idx]
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    dX = np.array([X_train[i] - X_train[j] for i, j in PAIRS])
    dz = np.array([z_train[i] - z_train[j] for i, j in PAIRS])
    model = Ridge(alpha=1.0)
    model.fit(dX, dz)

    X_refs = X_train[REF_IDX]
    z_refs = z_train[REF_IDX]

    def predict(X_target):
        preds = []
        for x_r, z_r in zip(X_refs, z_refs):
            dx = X_target - x_r[None, :]
            dz_pred = model.predict(dx)
            z_pred_r = z_r + dz_pred
            preds.append(np.exp(z_pred_r * sigma + mu))
        return np.mean(np.stack(preds, axis=0), axis=0)

    val_scalar_sub = val_scalar[:, feat_idx]
    test_scalar_sub = test_scalar[:, feat_idx]
    if val_scalar_sub.ndim == 1:
        val_scalar_sub = val_scalar_sub.reshape(-1, 1)
        test_scalar_sub = test_scalar_sub.reshape(-1, 1)
    val_pred = predict(val_scalar_sub)
    test_pred = predict(test_scalar_sub)
    return rmse(val_pred, val_label), rmse(test_pred, test_label), val_pred, test_pred


if __name__ == "__main__":
    print(f"Fixed reference cells: {REF_IDX.tolist()}, true RUL: {[round(x,1) for x in REF_RUL]}\n")

    # ---- Step 1: single-feature ablation, ranked by VAL RMSE ----
    print("=== Single-feature pairwise ablation (ranked by VAL RMSE, not test) ===")
    single_results = {}
    for i, name in enumerate(names):
        val_r, test_r, _, _ = eval_feature_subset([i])
        single_results[name] = (val_r, test_r)

    ranked = sorted(single_results.items(), key=lambda kv: kv[1][0])
    print(f"{'rank':4s} {'feature':28s} {'val_RMSE':>9s} {'test_RMSE':>10s}")
    for rank, (name, (val_r, test_r)) in enumerate(ranked, 1):
        print(f"{rank:<4d} {name:28s} {val_r:9.2f} {test_r:10.2f}")

    top_features_by_val = [name for name, _ in ranked]

    # ---- Step 2: top-k combinations (k chosen by VAL, k in a pre-declared grid) ----
    print("\n=== Top-k feature combinations (features chosen by Step 1's VAL ranking) ===")
    idx_by_name = {n: i for i, n in enumerate(names)}
    combo_results = {}
    for k in [1, 2, 3, 5, 7, 10, 15, 20, 34]:
        feat_names_k = top_features_by_val[:k]
        feat_idx_k = [idx_by_name[n] for n in feat_names_k]
        val_r, test_r, val_pred, test_pred = eval_feature_subset(feat_idx_k)
        combo_results[k] = dict(val_rmse=val_r, test_rmse=test_r, features=feat_names_k,
                                 val_pred=val_pred, test_pred=test_pred)
        print(f"k={k:2d}  val_RMSE={val_r:8.2f}  test_RMSE={test_r:8.2f}  features={feat_names_k if k<=5 else feat_names_k[:5]+['...']}")

    best_k = min(combo_results.keys(), key=lambda k: combo_results[k]["val_rmse"])
    print(f"\nBest k by VAL: k={best_k}  val_RMSE={combo_results[best_k]['val_rmse']:.2f}  "
          f"test_RMSE={combo_results[best_k]['test_rmse']:.2f}")

    # ---- Step 3: ensemble best pairwise config with Tier-3 (PINN) via NNLS, val-only ----
    print("\n=== Ensemble: best pairwise-linear config + Tier-3 (Wiener PINN), NNLS on val ===")
    with open(PINN_TIERS_PKL, "rb") as f:
        pinn = pickle.load(f)
    t3 = pinn["3"]
    pw_val = combo_results[best_k]["val_pred"]
    pw_test = combo_results[best_k]["test_pred"]

    V = np.stack([pw_val, t3["val_pred"]], axis=1)
    w, _ = nnls(V, val_label)
    ens_val = w[0] * pw_val + w[1] * t3["val_pred"]
    ens_test = w[0] * pw_test + w[1] * t3["test_pred"]
    ens_val_rmse = rmse(ens_val, val_label)
    ens_test_rmse = rmse(ens_test, test_label)
    corr = float(np.corrcoef(pw_test - test_label, t3["test_pred"] - t3["test_true"])[0, 1])

    print(f"pairwise-linear (k={best_k}) alone: val={combo_results[best_k]['val_rmse']:.2f} test={combo_results[best_k]['test_rmse']:.2f}")
    print(f"Tier-3 alone: val={rmse(t3['val_pred'], val_label):.2f} test={rmse(t3['test_pred'], t3['test_true']):.2f}")
    print(f"NNLS weights (pairwise, tier3): {w.tolist()}")
    print(f"Ensemble: val_RMSE={ens_val_rmse:.2f} test_RMSE={ens_test_rmse:.2f}  corr(e_pairwise,e_tier3)={corr:.3f}")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(dict(single_results=single_results, ranked=ranked, combo_results=combo_results,
                          best_k=best_k, ensemble=dict(weights=w.tolist(), val_rmse=ens_val_rmse,
                                                        test_rmse=ens_test_rmse, corr=corr)), f)
    print(f"\nSaved to {OUT_PKL}")
