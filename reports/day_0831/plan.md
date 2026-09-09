# Plan — 2026-08-31

Carried over from `day_0827`/`day_0901` architecture-ladder work: the V1/V2
`SmallCNNScalarBranchAxisAware` (top-3 scalar features, attention pooling)
established on MATR1 is reused as-is on two new benchmarks this session —
MATR2 (secondary/unseen-protocol test) and HUST — with the user's explicit
focus for this file being **HUST's val-set-construction problem** and the
resulting headline.

## Scoreboard going in

| Dataset | Status | Ours | README target |
|---|---|---|---|
| MATR1 | ✅ beaten (prior session) | 72.15 ± 4.88 RMSE (8-seed Affine-3way ensemble) | 90 (PCR) |
| MATR2 | 🟡 explored, not beaten | best legitimate: 291.62-282.94 RMSE range (single-split diagnostics, not multi-seed) | 149 (Discharge model) |
| HUST | ✅ beaten (this session) | **291.24 ± 13.09 RMSE (5-seed, V1 solo)** | 322 |

## Task table (easiest → hardest)

| # | Task | What / why | Est. | Status |
|---|---|---|---|---|
| 1 | Reproduce official README baselines locally (MATR2 Discharge model, HUST context) | Sanity gate before any custom split work — confirms 148.58≈149 (MATR2) is real | 15 min | ✅ Done — MATR2 Discharge=148.58, matches README |
| 2 | MATR2: find a val set for the custom SmallCNN pipeline | Official MATR2 has no val; tried RUL-matching, feature-distance, K-Fold, HUST-borrowing — see `report.md` | 3-4h | 🟡 Done — no config beat README=149; best diagnostic ~283-292 RMSE, not multi-seed-confirmed |
| 3 | HUST: same val-set problem, native pool this time | HUST train(55)/test(22) are well RUL-matched (unlike MATR2) — expected easier, turned out harder for a different reason (small-n variance) | 3h | ✅ Done — cluster(feature+RUL)-based val=12 is the only construction that gives genuine positive val~test correlation |
| 4 | HUST: augment train with MATR cells, tune selection method and count | RUL-range filter hurt; feature+RUL z-distance selection helped; swept count 15/20/30/45 | 1.5h | ✅ Done — count=20 is the sweet spot (282.94-291.62 range across selection methods) |
| 5 | HUST: 5-seed confirmation of the winning config | Multi-seed required before any claim, per project convention | 1h | ✅ Done — 291.24 ± 13.09 RMSE, beats README=322 by ~9.6% |
| 6 | HUST: 3-model (V2+V1+Inter-Embedding) ensemble, per seed | Test whether ensembling improves on V1 solo | 1.5h | ❌ Done — ensemble (NNLS or any weighting) never beats V1 solo; isotonic reconfirmed harmful |

## Standing directions

- Val must be constructed from a principled, test-blind rule (RUL/feature
  clustering on the pool's own data) — never selected by checking which
  candidate correlates best with test.
- Multi-seed (5-seed minimum) required before any RMSE is claimed as a result.
- Isotonic calibration is banned as a candidate — reconfirmed harmful again
  this session (5/5 seeds × 2 models = 10/10 cases worse on test).
- CHUNK=48 mini-batch fix required for any train_base >~90 cells on this 4GB
  GPU (full-batch OOMs); PATIENCE=200 is the standard early-stop unless a
  task explicitly disables it for observation-only diagnostics.

## Open items carried from `day_0901` report

1. MATR1 Stage 1+ architecture-ladder search (per `Project_Architecture_Ladder_Spec.md`)
   still open — baseline to beat is 101.18 ± 6.61 (not the earlier single-seed 95.42).

## Standing constraints

- EOL threshold: 80% default for MATR1/MATR2/HUST (SNL/CRUSH use 90% — not
  touched this session).
- Never select val, calibration method, or ensemble weights by looking
  directly at test performance.
- Windows/joblib: never nest `n_jobs=-1`.
- Report MAE, RMSE, MAPE, R² together — never RMSE alone.
