# Plan — 2026-08-26

Carried over from [day_0825/plan.md](../day_0825/plan.md). That session's headline was
the deterministic seed-averaged 3-model ensemble (`top3`+`qdlin_diff_min`+`qdlin_diff_l2`,
fixed weights 0.5/0.25/0.25): **84.27 RMSE**, beating README (90) by 5.7 points. Task #18
there (under-predict-penalty on `min`/`l2`, weight=1.5) was left 🟡 running — this session
continues and closes it out, then extends the ensemble sweep and reruns at weight=2.

External reviewer feedback (relayed by user) confirmed the staged PINN plan and added a
new **Tier 0.5** (cross-cell feature-ordering / pairwise ranking loss) between Tier 0 and
Tier 1 — recorded in `report_expirement/day_0825/Wiener_PINN_Project_Checklist.md` §4.5.
Tier 0.5 is *not* the same as Tier 2 (within-cell SOH monotonicity): Tier 0.5 needs no SOH
head and acts directly on the RUL head, so it's cheaper than Tier 1, but still gated on a
train-cells-only ranking-diagnostic (Spearman/Kendall/pairwise-ordering-accuracy) passing
before any ranking-loss training is attempted.

## Scoreboard going in

| Dataset | Status | Ours | README target |
|---|---|---|---|
| MATR1 | 🟡 candidate found, seed0-only — 84.27 (day_0825, multi-seed/deterministic) remains the honest headline | Today's best RAW: **80.19** (3×3×3 grid, top3=w1.0/min=w1.5/l2=w2.0, NNLS val-fit) — seed0-only across every loss-modified member, not yet claimable. Deterministic 5-seed 84.27 from day_0825 is still the number that stands until this is multi-seeded | 90 (PCR) |

## Task table (easiest → hardest)

| # | Task | What / why | Est. | Status |
|---|---|---|---|---|
| 1 | Finish task #18 from day_0825: under-predict penalty, `qdlin_diff_min`/`qdlin_diff_l2`, weight=1.5, seed=0, patience=350 | Relaunched after a GPU-driver TDR crash (fixed via restart) and one process stall (killed+resumed cleanly from pickle, verified deterministic). **`qdlin_diff_min`: 149.39 (baseline 127.82, worse, +21.57)**. **`qdlin_diff_l2`: 110.57 (baseline 118.12, better, -7.55)** — long-life RMSE 167.77→149.03, long-life bias -74.14→-68.11, driven mainly by 3 longest-life cells (+175 to +206 each) | ~1h GPU (post-crash reruns) | ✅ done |
| 2 | Same under-predict-penalty mechanism, weight=2, both features, seed=0, patience=350 | **`qdlin_diff_min`: 153.75 (baseline 127.82, worse, +25.93; long_bias -68.1→-17.54)**. **`qdlin_diff_l2`: 104.94 (baseline 118.12, better, -13.18 — best l2 variant found; long_bias -74.1→-36.91)**. `min` degrades monotonically with weight (127.82→149.39→153.75); `l2` improves monotonically (118.12→110.57→104.94) | ~30min GPU | ✅ done |
| 3 | 3-way ensemble (`top3`+`qdlin_diff_min`+`qdlin_diff_l2`), all at weight=1.5, seed0-only — simple avg + OLS, raw + calibrated | Simple avg: RAW=90.67, CAL=85.78. OLS unstable (min gets a negative weight, -0.32 — noise-fit on 41-pt val, not signal), RAW=103.88 (worse than simple avg). Reference: plain (no-weight) 5-seed simple-avg=90.34±5.79 | ~10min analysis | ✅ done |
| 4 | Fixed-weight 0.5/0.25/0.25 (top3/min/l2) sweep across three member-sets: no-weight, weight=1.5, and best-per-feature selection | **No-weight: RAW=86.99, CAL=89.36. Weight=1.5: RAW=83.22, CAL=77.38. Best-per-feature (top3=no-weight 95.42, min=no-weight 127.82, l2=weight1.5 110.57): RAW=82.32, CAL=83.53.** Also ran simple-avg (87.59/81.13) and OLS (100.02/92.02, unstable) on the best-per-feature trio | ~15min analysis | ✅ done |
| 5 | 6-way "combined" ensemble (no-weight + weight=1.5, all 6 base predictions) — simple avg + OLS + fixed-weight, raw + calibrated | Fixed-weight (0.25/0.125/0.125 split per regime, i.e. 0.5/0.25/0.25 total): RAW=84.24, CAL=74.75. Simple avg: RAW=89.54/CAL=88.78. OLS: RAW=99.51/CAL=84.38 | ~10min analysis | ✅ done |
| 6 | Insert Tier 0.5 (cross-cell ranking loss) into the staged PINN plan, per external-reviewer-confirmed update | Updated `day_0825/Wiener_PINN_Project_Checklist.md` §4.5 (new section between Tier 0 and Tier 1) + §12 final comparison table. Also ran the Step-1 ranking diagnostic itself (Spearman/Kendall/pairwise-accuracy for `qdlin_diff_l2`/`qdlin_diff_min`/`top3`'s 3 features vs. true life, train cells): moderate real signal (~70-77% best-direction accuracy), mid-life always the weak tercile (53-61%, never significant) across every feature. Trained-model predictions already order far better (82-88%) than raw features — ranking loss has a smaller gap to close than expected. Not started as a training experiment — gated behind Tier 0 closing | ~30min (doc + diagnostic) | ✅ doc + diagnostic done, ⬜ ranking-loss training not started |
| 7 | **Full 3×3×3 ensemble grid search**: every combo of {top3, min, l2} × {weight 1.0, 1.5, 2.0} each, × 3 combination methods (simple avg, fixed 0.5/0.25/0.25, per-combo NNLS weights fit on val-only/frozen-to-test) — RAW + calibrated | **New session-best: RAW=80.19** (top3=w1.0, min=w1.5, l2=w2.0, NNLS weights [0.704, 0.122, 0.175] — non-negative-constrained, no OLS instability), CAL=70.18. Second-best: RAW=80.66-80.70 via plain fixed-weight combos (top3=w1.0-1.5, min=w1.0, l2=w2.0). All 27×3 combos saved to `ensemble_grid_3x3x3_results.pkl` | ~15min analysis | ✅ done |

## Standing directions

- **All ensemble numbers this session (tasks #3-5) are seed0-only** for every
  loss-modified member (`top3` safety-loss, `min`/`l2` under-predict-penalty) —
  none of today's RAW/CAL numbers are claimable yet; per day_0825's task #13 decision
  (still pending), do not auto-promote any single weight to 5-seed evaluation without
  an explicit decision to do so.
- **OLS-weighting is unstable whenever `qdlin_diff_min` is a member** — it repeatedly
  fits a negative weight (-0.21 to -0.32 across every OLS run today), which is
  overfitting the 41-point val set, not a real signal. Simple average and manually-chosen
  fixed weights (0.5/0.25/0.25) both consistently beat OLS on RAW RMSE — treat any
  OLS-selected result as untrustworthy until cross-validated within val (same standing
  rule as day_0825's OLS+intercept retraction, task #5 there).
- **Calibrated numbers still carry the day_0825 LOO-artifact caveat** (honest val RMSE
  151.25 vs raw 97.57 for a similar ensemble) — report every CALIBRATED number in this
  session next to that caveat, never standalone. The RAW numbers (82.32 best-per-feature,
  83.22 all-weight1.5) are this session's only numbers worth taking at face value.
- **Tier 0.5 (cross-cell ranking loss) is a separate branch from Tier 2 (SOH
  monotonicity)** — Tier 0.5 acts on the existing direct-RUL head (no new SOH head,
  cheaper than Tier 1), Tier 2 acts on a predicted SOH trajectory (needs Tier 1's SOH
  head first). Updated staged order: Tier 0 → Tier 0.5 → Tier 1 → Tier 2 → optional
  smoothness → Tier 3. Do not start Tier 0.5's ranking-loss training before its own
  Step-1 diagnostic (Spearman/Kendall/pairwise-ordering-accuracy on `qdlin_diff_l2` vs.
  true total life, train cells only) shows a reliable ordering.

## Open items carried from day_0825 report

1. `train_val_test_selection.md` still needs an update, now also folding in this
   session's under-predict-penalty and ensemble findings.
2. Ensemble-of-ensembles (108.36, R²=0.921, day_0819) weighting still not swept.
3. K-fold(MATR1-only) multi-seed still only has seed=0 (130.76), not touched.
4. Calibration-generalization question — still unresolved; this session's numbers
   reinforce rather than resolve it (LOO artifact caveat applied consistently).
5. `top3` 5-seed headline (88.36±9.50) still used patience=350, not re-verified at
   full 1000-epoch/no-early-stop.
7. `variant_c_cycle_window_ensemble_v1.ipynb` cell 0 still missing `import BatLiNet`.
8. Section-4/SmallMLP idea from `02_plan_test_model_code.md` — still not started.
9. Only 14/34 ablation features have cached predictions — still a gap, low priority.

## Open items from today

10. Task #2 (weight=2 run) not yet finished — once done, decide whether weight=1.5 or
    weight=2 is the better `qdlin_diff_l2` variant before any multi-seed commitment.
11. Tier 0 (day_0825 task #13 + this session's under-predict-penalty extension) still
    has no closed decision — needed before Tier 0.5's diagnostic step is worth running.
12. Tier 0.5's Step-1 ranking diagnostic (Spearman/Kendall/pairwise-ordering-accuracy for
    `qdlin_diff_l2` vs. true total life, train cells only, broken out by life tercile) —
    not yet run.
13. **Backlog idea (user-proposed, not started): a BatLiNet-style relative-regression
    branch** — predict the *magnitude* of pairwise RUL difference against a set of
    reference/anchor cells (not just the sign, unlike Tier 0.5's ranking loss), then
    aggregate multiple pairwise predictions into one final estimate at inference time,
    mirroring BatLiNet's fixed-aggregation design (see `[[feedback_small_n_nn_training_lessons]]`
    memory). This is a bigger step than Tier 0.5 — a real architecture/inference-time
    change, not a cheap auxiliary loss — so it sits *after* Tier 0.5's screen, not
    instead of it. Explicitly deferred by the user ("sau" — later), not to be started
    this session.

## Standing constraints

(copied forward from CLAUDE.md / day_0825, still in force)

- Multi-seed mean±std before any claim; no single-seed claims; pre-declared
  hyperparameters; no train-side CV selection.
- All 4 metrics (MAE/RMSE/MAPE/R²) in every reported row where a formal claim is made;
  MATR2 not in scope this session (MATR1-only work).
- Comparisons vs. the README benchmark bar (MATR1=90, PCR) — never vs. locally
  reproduced baselines for the "beaten?" verdict.
- cuda only (GTX 1650 4GB), one GPU job at a time; full 1000-epoch cap, no early
  stopping for any run whose number will be claimed (patience=350 used again this
  session for the under-predict-penalty screens — diagnostic-only, flagged, not yet
  re-verified at the full protocol).
- Windows/joblib: never nest `n_jobs=-1`.
- GPU driver TDR / VRAM pressure can crash long sessions on this 4GB GTX1650 — if it
  recurs, a full restart is the known fix; stall heuristic: CPU frozen 3+ checks over
  5+ minutes despite GPU=100% = genuine stall (kill+relaunch from the incremental
  pickle), vs. CPU still climbing = healthy (including normal stdout-buffering delays).
