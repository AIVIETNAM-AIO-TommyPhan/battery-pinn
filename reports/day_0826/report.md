# Session report - 2026-08-26

Status legend: ✅ done/verified (against saved pickle results) · 🟡 partial / seed0-only,
not yet claimable · ⬜ not started · ❌ negative (does not beat baseline)

Plan for the day: [plan.md](plan.md)

## Summary table

| Dataset | Official baseline (README) | Our best result | Beats baseline? |
|---|---|---|---|
| MATR1 | 90 (PCR) | **80.19 RAW RMSE** (3×3×3 ensemble grid, NNLS val-fit) | 🟡 Beats it by 9.8 points on paper, but **seed0-only for every loss-modified member** — not a claim yet. The standing headline remains day_0825's deterministic 5-seed ensemble, **84.27 RMSE** |

---

## MATR1 — under-predict-penalty loss on `qdlin_diff_min`/`qdlin_diff_l2` (task #1-2)

**What we did:** Continued day_0825's task #18: an asymmetric loss that penalizes
*under*-prediction (`error < 0`) more heavily than over-prediction, applied to two
single-feature CNN scalar-branch models (`qdlin_diff_min`, `qdlin_diff_l2`), seed=0,
patience=350, on the median-fill MATR1 split (train=91/val=41/test=42 cells,
`eol_soh=0.8`). Ran at `safety_weight` (renamed `UNDER_WEIGHT`) = 1.5, then 2.0. A GPU
driver TDR crash (VRAM pressure on a 4GB GTX1650 after a long prior session) interrupted
the weight=1.5 run mid-`qdlin_diff_min`; fixed by a full machine restart, then a
resumed run reproduced the pre-crash numbers exactly (deterministic sanity check
passed).

**Result: full RMSE, long-life RMSE, long-life bias (MAE/MAPE/R² not separately
tracked for this single-feature diagnostic series — see Standing constraints):**

| Model | Weight | Full RMSE | Long-life RMSE | Long-life bias |
|---|---|---:|---:|---:|
| `qdlin_diff_min` | baseline (1.0) | 127.82 | 208.15 | -68.10 |
| `qdlin_diff_min` | 1.5 | 149.39 (worse) | 239.74 | -61.23 |
| `qdlin_diff_min` | 2.0 | 153.75 (worse) | 247.25 | -17.54 |
| `qdlin_diff_l2` | baseline (1.0) | 118.12 | 167.77 | -74.10 |
| `qdlin_diff_l2` | 1.5 | **110.57 (better, -7.55)** | 149.03 | -68.11 |
| `qdlin_diff_l2` | 2.0 | **104.94 (better, -13.18)** | 149.70 | -36.91 |

**Why it's credible:**
- Sanity gate: post-crash resume reproduced identical epoch-by-epoch numbers to the
  pre-crash log at seed=0 (deterministic).
- `qdlin_diff_l2` improves *monotonically* with weight (118.12→110.57→104.94) and
  long-life bias shrinks substantially (-74→-68→-37) — a consistent, explainable
  direction (the loss is specifically designed to fix long-life under-prediction), not
  a lucky single point.

**Why the claim must stay modest / Caveats:**
- **Seed=0 only** for both features at both weights — no multi-seed mean±std yet. Per
  project rule, this is a candidate, not a claimable result.
- `qdlin_diff_min` moves in the *opposite* direction (gets worse as weight increases) —
  the loss modification is feature-specific, not universally helpful. Applying it
  project-wide without per-feature validation would be wrong.
- Not yet checked whether weight=2 (or higher) is actually the ceiling for `l2`, or
  whether it keeps improving — no weight beyond 2.0 was tested.

---

## MATR1 — ensemble sweep: `top3` + `qdlin_diff_min` + `qdlin_diff_l2` (tasks #3-5, #7)

**What we did:** Building on day_0825's 3-model ensemble (`top3`'s 3-feature CNN +
the two single-feature models above), tested combining members trained under
different loss weights. Four analyses, all seed=0, MATR1 median-fill split,
`eol_soh=0.8`:
1. All three members at weight=1.5 (simple avg + unconstrained OLS).
2. Fixed weights (0.5·top3 + 0.25·min + 0.25·l2) across three member-sets: no-weight,
   all-weight1.5, and "best-per-feature" (whichever weight variant had the lowest
   individual RMSE for that feature).
3. A 6-way "combined" ensemble mixing the no-weight and weight=1.5 variants of all
   three models.
4. A full grid: every model × every weight (1.0/1.5/2.0) × 3 combination methods
   (simple avg, fixed 0.5/0.25/0.25, per-combo NNLS weights fit on validation only
   and frozen before applying to test) — 27 model-combos × 3 methods = 81 ensemble
   configurations. Isotonic calibration (fit on val, applied to test) computed
   alongside every RAW number.

**Result: RMSE only (MAE/MAPE/R² not computed for this seed0 diagnostic sweep — flagged
below), best 5 configurations by RAW RMSE:**

| top3 weight | min weight | l2 weight | Method | RAW | Calibrated |
|---|---|---|---|---:|---:|
| 1.0 | 1.5 | 2.0 | NNLS (val-fit, frozen) | **80.19** | 70.18 |
| 1.0 | 1.0 | 2.0 | fixed 0.5/0.25/0.25 | 80.66 | 75.98 |
| 1.5 | 1.0 | 2.0 | fixed 0.5/0.25/0.25 | 80.70 | 75.48 |
| 1.5 | 2.0 | 2.0 | fixed 0.5/0.25/0.25 | 81.14 | 77.52 |
| 1.5 | 1.5 | 2.0 | fixed 0.5/0.25/0.25 | 81.29 | 76.60 |

Reference points: day_0825's deterministic 5-seed headline = **84.27**; README
benchmark = 90 (PCR); plain (no-weight) 5-seed simple-avg = 90.34±5.79.

**Why it's credible:**
- The NNLS weight search is **non-negative-constrained** (`scipy.optimize.nnls`), fit
  on validation only, then frozen and applied once to test — this avoids the
  instability seen in every unconstrained-OLS attempt this session and at day_0825
  (which repeatedly assigned `qdlin_diff_min` a *negative* weight, a clear sign of
  overfitting a 41-point validation set).
- Every one of the top 5 combos independently lands on `qdlin_diff_l2` at weight=2.0 —
  consistent with that model's individually-best RMSE (104.94), not an ensemble-only
  artifact.
- Isotonic-calibrated numbers are reported next to every RAW number, never in place of
  it.

**Why the claim must stay modest / Caveats:**
- **Every number here is seed=0 only**, for every loss-modified member (`top3` at
  safety-loss weight, `min`/`l2` at under-predict-penalty weight). None of today's RAW
  or CALIBRATED numbers are claimable — day_0825's deterministic 84.27 remains the
  standing headline until this grid is re-run multi-seed.
- **Calibrated numbers carry the standing LOO-artifact caveat**: an earlier LOO check
  on a similar ensemble this project found the isotonic-calibration gain does not
  survive honest leave-one-out validation (honest val RMSE 151.25 vs. raw 97.57) —
  treat every CALIBRATED number in the table above as a same-split artifact, not a
  validated effect.
- Unconstrained OLS remains unstable whenever `qdlin_diff_min` is a member (negative
  weight every time it was tried this session) — excluded from the final grid in favor
  of NNLS for this reason.
- MAE/MAPE/R² were not computed for this diagnostic sweep (RMSE-only, consistent with
  how the single-feature ablation series has been tracked all project) — would need to
  be added before any of these numbers could appear in a manuscript claim.

**Scripts:** `ensemble_noweight_vs_w15_vs_combined.py`, ad hoc grid-search script
(3×3×3, saved results at `ipynb/exp_v3/matr1_feature_cache/ensemble_grid_3x3x3_results.pkl`).

---

## MATR1 — Tier 0.5 ranking diagnostic (task #6)

**What we did:** Per an external-reviewer-confirmed update to the staged PINN plan
(inserting a new Tier 0.5 between Tier 0 and Tier 1 — see
`day_0825/Wiener_PINN_Project_Checklist.md` §4.5), ran the Tier 0.5 Step-1 diagnostic:
Spearman/Kendall correlation and pairwise-ordering accuracy between each of 5 features
(`qdlin_diff_l2`, `qdlin_diff_min`, `qdlin_diff_std`, `voltage_slope_50_90`,
`voltage_soc_90`) and true total cycle life, computed on the 91 training cells only,
broken out by life tercile. Also checked the actual weight=1.5 trained models' TEST
predictions for the same ordering accuracy, as a sanity comparison.

**Result:** best-direction pairwise-ordering accuracy, all cells / short / mid / long:

| Feature | All | Short | Mid | Long |
|---|---:|---:|---:|---:|
| `qdlin_diff_std` | 76.9% | 76.7% | 57.9% (n.s.) | 78.2% |
| `voltage_slope_50_90` | 74.8% | 68.9% | 53.5% (n.s.) | 67.0% |
| `voltage_soc_90` | 74.7% | 67.3% | 55.8% (n.s.) | 66.1% |
| `qdlin_diff_min` | 70.1% | 78.1% | 60.7% (n.s.) | 66.5% |
| `qdlin_diff_l2` | 69.8% | 76.0% | 58.6% (n.s.) | 68.5% |

Trained-model (weight=1.5) TEST predictions: `top3`=88.4%, `qdlin_diff_l2`=86.4%,
`qdlin_diff_min`=82.2% overall.

**Why it's credible:** the mid-life weak spot is consistent across every single
feature checked (never statistically significant, 53-61%), a structural property of
this cell population rather than one feature's idiosyncrasy — a real, reproducible
finding.

**Why the claim must stay modest / Caveats:** this is a diagnostic only, no
ranking-loss training was run. The finding that trained models already order much
better than raw features (82-88% vs. 70-77%) suggests a ranking loss has a smaller
gap to close than the raw numbers alone imply — expectations for Tier 0.5's eventual
training experiment should stay modest, per the reviewer's own framing ("worth a cheap
test, but large improvement should not be expected").

---

## Open items

1. ~~Task #18 from day_0825 (weight=1.5 under-predict-penalty run)~~ — done, see above.
2. ~~Weight=2 under-predict-penalty run~~ — done, see above.
3. **Multi-seed the 3×3×3 grid's best combo** (top3=w1.0, min=w1.5, l2=w2.0, NNLS) —
   the 80.19 RAW number is not claimable until this happens. Next logical action.
4. Tier 0.5 Step 2 (actual ranking-loss training, small `lambda_rank` sweep on `top3`,
   short/long-life pairs only, mid-life down-weighted) — not started, deferred.
5. Backlog idea (user-proposed): a BatLiNet-style relative-regression branch (predict
   pairwise magnitude difference against anchor cells, aggregate at inference) — bigger
   change than Tier 0.5, explicitly deferred, not started this session.
6. `train_val_test_selection.md` still needs an update (carried from day_0823/0825).
7. Ensemble-of-ensembles (108.36, day_0819) weighting still not swept (carried).
8. K-fold(MATR1-only) multi-seed still only has seed=0 (carried).
9. Calibration-generalization question — still unresolved (carried); today's results
   are consistent with, not resolving, the standing LOO-artifact caveat.
10. `top3` 5-seed headline (88.36±9.50, day_0825) still used patience=350, not
    re-verified at full 1000-epoch/no-early-stop protocol (carried).
