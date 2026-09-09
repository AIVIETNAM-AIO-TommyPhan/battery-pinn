# Session report — 2026-09-01

Status legend: ✅ done/verified (multi-seed, sanity-gated) · 🟡 partial or awaiting a
further run · ⬜ not started · ❌ attempted, does not beat baseline.

Plan for the day: [plan.md](plan.md), architecture spec: [Project_Architecture_Ladder_Spec.md](Project_Architecture_Ladder_Spec.md).

## Summary table

| Dataset | Official baseline (README) | Our best result | Beats baseline? |
|---|---|---|---|
| MATR1 (Stage 0 baseline) | 90 (PCR) | **101.18 ± 6.61 RMSE** (5-seed, `SmallCNNScalarBranch`) | ❌ open — this is the reference point Stage 1+ must beat, not yet a claim against README |

---

## MATR1 — Stage 0 sanity baseline, `SmallCNNScalarBranch`, 5-seed (task #1)

**What we did:** Reproduced the historical baseline fresh before starting the
architecture-ladder search. `SmallCNNScalarBranch` (CNN + top-3 scalar branch:
`qdlin_diff_std`, `voltage_slope_50_90`, `voltage_soc_90`), input
`clean_feature(feat - feat[:, :, [9], :])` (`DIFF_BASE=9`), `AdamW(lr=1e-3)` default
decay, `EPOCHS=1000`, `EVAL_EVERY=50`. Seed=0 trained fresh this session
(`ipynb/exp_v3/matr1_feature_cache/run_stage0_sanity.py`); seeds 1-4 reused from
`top3_multiseed_results.pkl` (same model/protocol/code path, deterministic seeding)
instead of retraining — saves ~1.5-2h of GPU time with no loss of rigor once seed=0
confirms the code path is unchanged. Dataset MATR1, EOL threshold 80% (default), split
= official MATR1_train/MATR1_test (via `NEW_TRAIN_BASE_IDS`/`NEW_VAL_IDS`/`OFFICIAL_TEST_IDS`
in `matr1_feature_cache/feature_cache.pkl`), seeds 0-4.

**Protocol correction found and applied before running:** `Project_Architecture_Ladder_Spec.md`
originally pinned `PATIENCE=1_000_000_000` ("matches the historical 95.42 run exactly").
This was wrong — the source notebook that produced 95.42
(`ipynb/exp_v2/matr/matr1/variant_c_tier0_safety_loss_v1.ipynb`, cell 11) and
`report_expirement/day_0825/report.md` both confirm the actual protocol was
**`patience=350`**. Running with `patience=1e9` would not have reproduced 95.42.
Corrected in `plan.md` and the spec before executing.

**Result: RMSE 101.18 ± 6.61, MAE 82.39 ± 6.92, MAPE not recomputed this pass, R² 0.931 ± 0.009**

| seed | RMSE | source |
|---|---|---|
| 0 | 95.42 (95.4175, fresh) | trained fresh this session |
| 1 | 107.04 | reused, `top3_multiseed_results.pkl` |
| 2 | 98.61 | reused, `top3_multiseed_results.pkl` |
| 3 | 110.81 | reused, `top3_multiseed_results.pkl` |
| 4 | 94.00 | reused, `top3_multiseed_results.pkl` |
| **mean ± std** | **101.18 ± 6.61** | |

Bias (test tercile split): short_bias +83.66 ± 18.79, long_bias +9.49 ± 24.92.
n_parameters = 15,153.

**Why it's credible:**
- Sanity gate: fresh seed=0 hit its best-val checkpoint at epoch 200 (val_RMSE=97.76,
  test_RMSE=95.4175), an exact match (diff=0.0025) to the historical 95.42, at the same
  epoch the historical run picked (epoch=200, per `top3_multiseed_results.pkl`).
  Training then correctly degraded (val_RMSE rising through epoch 550) and early-stopped
  at the patience=350 threshold — the trajectory shape matches what patience=350 predicts,
  not what patience=1e9 would have produced.
- Seeds 1-4 reuse is deterministic (`torch.manual_seed`, `cudnn.deterministic=True`,
  same code path) — validity of reuse rests entirely on seed=0's exact match, which held.

**Why the claim must stay modest / Caveats:**
- 101.18 ± 6.61 is *worse* than the single-seed 95.42 headline previously cited in
  `day_0825` — that number was always a best/lucky-seed read, never validated multi-seed
  until now. This is the correction: **101.18 is the honest Stage 0 baseline**, not 95.42.
  Stage 1+ candidates must be compared against 101.18, per `passes_primary_gate()`.
- MAPE not recomputed in this pass (only RMSE/MAE/R²/bias were wired into the aggregate
  script); trivial to add if needed for the manuscript.
- Still open vs. README=90 — Stage 0 is a reproduction/reference step, not a claim.

**Notebook/script:** `ipynb/exp_v3/matr1_feature_cache/run_stage0_sanity.py` (executed
directly, not as a notebook — single fast verification run, output pkl matches the
schema in `Project_Architecture_Ladder_Spec.md` §Stage 0). Output:
`ipynb/exp_v3/matr1_feature_cache/stage0_smallcnn_scalar_branch_5seed.pkl`.

---

## Open items

1. Stage 1 (axis-aware pooling head, V1-V4 screen) is now unblocked — baseline to beat
   is **101.18 ± 6.61** (not 95.42).
2. MAPE not recomputed for Stage 0 aggregate — add if a full paper-table pass is needed
   before Stage 4 summary.
3. `Project_Architecture_Ladder_Spec.md`'s `PATIENCE=1_000_000_000` pin was corrected to
   `350` throughout (protocol block, Stage 0 section, output schema) — worth a final
   grep before Stage 4 to confirm no stale `1e9` references remain if the spec is edited
   further.
