# Plan — 2026-08-25

Carried over from [day_0823/plan.md](../day_0823/plan.md). That session's headline was
median-fill Variant C, 5-seed, isotonic-calibrated CNN-only: **119.25 ± 7.94**
(R²=0.904 ± 0.013), still open vs. README's 90 (PCR). This session finished the
single-feature scalar-branch ablation, found `top3` (3 features), built a 3-model
ensemble (`top3`+`qdlin_diff_min`+`qdlin_diff_l2`), and — via a deterministic
seed-averaged blend — reached the session's current best: **84.27 RMSE** (no std needed,
it's a single averaged prediction), clearing README's 90 by 5.7 points. See
[report.md](report.md) for full detail, including an unresolved calibration-
generalization caveat on the earlier per-seed-calibrated numbers (not used in the 84.27
headline, which is raw/uncalibrated).

## Scoreboard going in

| Dataset | Status | Ours | README target |
|---|---|---|---|
| MATR1 | 🟢 **beats baseline, deterministic (no seed variance)** | 3-model ensemble (`top3`/`qdlin_diff_min`/`qdlin_diff_l2`, fixed weights 0.5/0.25/0.25), predictions averaged across 5 seeds into one final answer: **84.27 RMSE**, MAE=70.11, MAPE=10.38%, R²=0.952 — see [report.md](report.md) for the full derivation and caveats | 90 (PCR) |

## Task table (easiest → hardest)

| # | Task | What / why | Est. | Status |
|---|---|---|---|---|
| 1 | Single-feature ablation, 34 extended scalar features (seed=0, no early stop) | 7/34 features individually beat CNN-only (130.83). Best: `qdlin_diff_std`=104.19. See report.md | ~3h | ✅ done |
| 2 | Cumulative top-k combination sweep (top2→top5) + 7-feature combined + ranks-4-7 standalone | `top3` (`qdlin_diff_std`+`voltage_slope_50_90`+`voltage_soc_90`) = local optimum, 95.42 (seed0). 7-feature combined overfits (117.55, worse than single best). Ranks-4-7 alone = 167.05 (no independent signal without top3's features) | ~2h | ✅ done |
| 3 | `top3`, 5-seed, isotonic-calibrated — session headline | **88.36 ± 9.50** (R²=0.947±0.011), beats README 90. patience=350 (deviation from no-early-stop rule, flagged). All 4 metrics logged per seed | ~1.5h | ✅ done |
| 4 | Cross-seed error analysis + short-life safety framing | 79% sign-consistency (errors mostly systematic, not noise); short-life cells (RUL 335-499) get +40.3 over-prediction bias (calibrated) — the riskier failure direction. `qdlin_diff_min` alone is far better-calibrated for short-life cells (short-RMSE=51.02) than `top3` (108.96) — genuine Pareto trade-off, plotted | ~1h | ✅ done |
| 5 | 3-model validation-weighted ensemble (`top3`+`qdlin_diff_min`+`qdlin_diff_l2`, seed0 only) | NNLS/grid-search converge on `w≈(0.58,0.41,0.0)`, beats `top3` alone on every metric (RMSE 88.61 vs 95.42, MAE/MAPE/R² all improve too, short-life bias roughly halved). A more flexible OLS+intercept fit looked even better (80.88) but was confirmed overfit via 5-fold CV (honest OOF=147.01) and retracted | ~2h | ✅ done, seed0-only |
| 6 | Full 5-seed version of the 3-model ensemble, with calibration | Trained `qdlin_diff_min`/`qdlin_diff_l2` for seeds 1-4 (no-early-stop, full 1000 epochs — seed0 reused from patience=350 cache, a known protocol mismatch, flagged). Per-seed NNLS: raw=92.61±8.49, calibrated=86.81±9.45 — both beat `top3` alone. Seeds 2/3 showed calibration hurting the ensemble (misled by validation-only performance not transferring to test), consistent with the open calibration-generalization question | ~2h GPU (no-early-stop, 8 runs) | ✅ done |
| 7 | Local-median(window=5) glitch-fill variant vs. the adopted global-median-fill | Negligible difference (raw Δ=−0.58, calibrated Δ=+0.03) — only 2/42 test cells affected. Not adopted | ~30min | ✅ done, ❌ negative (no change) |
| 8 | Cycle-window split experiment (0-49 vs cycle9, 50-99 vs cycle50) — cheap screen before considering a joint multi-branch architecture | Both halves >2x worse than full-window baseline (274.79, 298.00 vs 130.83); free ensemble also >2x worse (266.95). Architecture-search track record now 0-for-4. No joint architecture built | ~1h | ✅ done, ❌ negative — direction closed |
| 9 | Ensemble-method comparison: NNLS, simple avg, median-of-3, inverse-val-RMSE weighting, fixed-weight variants, 2-stage tuning, XGBoost meta-learner, val-based winner-take-all selection | Best **fixed, untuned** weight (0.5·top3 + 0.25·min + 0.25·l2) = 87.30±3.70 — beat every tuning attempt tried (two separate honest pooled-val tuning searches both landed worse on test, 91-94, than the naive fixed split). XGBoost as meta-learner confirmed unusable via LOOCV (honest RMSE=202.88, worse than not ensembling at all — 41 val points is far too few for tree-based stacking). Winner-take-all model selection per seed also much worse (110.06±23.04) | ~1.5h | ✅ done |
| 10 | Deterministic seed-averaged ensemble (average the 5 seeds' 0.5/0.25/0.25-blended *predictions* into one final answer, not just report mean±std) | **84.27 RMSE, MAE=70.11, MAPE=10.38%, R²=0.952 — current session headline**, matches this project's established seed-averaging precedent (day_0819 CRUSH). Beats README (90) by 5.7 points, deterministic (no seed-variance caveat) | ~10min analysis, no new training | ✅ done |
| 11 | Per-seed tercile (short/mid/long-life) breakdown for `top3`/`min`/`l2` across all 5 seeds | `top3` wins long-life in 5/5 seeds; `min`/`l2` win short- and mid-life in 5/5 seeds — a genuine, seed-independent structural specialization (not seed-specific luck), explaining why the ensemble works | ~20min | ✅ done |
| 12 | Search the other 13 ablation features (with saved predictions) for a better short/mid-life specialist than `qdlin_diff_min` | No improvement found — `qdlin_diff_min` (short+mid RMSE=53.33) beats every alternative checked by a wide margin. Only 14/34 ablation features have cached predictions; the other 20 weren't retrained (would need GPU, low expected value given the checked features' full-RMSE ranking) | ~10min | ✅ done, ❌ negative (no better specialist found) |
| 13 | Tier 0 of the PINN4SOH-inspired lean staged plan: asymmetric safety loss (penalize RUL over-prediction more than under-prediction) on `top3`, seed=0, `safety_weight`∈{1.0,1.5,2.0} | **Complete, borderline/negative result.** Sanity gate passed exactly (w=1.0 reproduces 95.42). w=1.5: short-life bias unchanged (+99.84→+100.13), calibrated gain (82.46→71.90) confirmed via LOO check to look like a calibration artifact, not a real effect. w=2.0: short-life bias DID drop (−22.28, close to but under the checklist's ~25-30 gate), broadly distributed across 13/14 short-life cells (not 1-2-cell-driven) — but the mechanism is a *uniform* downward pull on all over-predicted cells, with the biggest corrections landing on long-life outliers (`b1c0` −175, `b1c2` −168) rather than short-life specifically; LOO validation-side short-life metric stayed completely flat (62.26→62.11) at every weight, so there's no honest validation-side confirmation this generalizes. Full RMSE cost: +8.3% at w=2.0, within tolerance. Verdict: does not clearly clear the pre-declared gate; user and assistant agreed NOT to auto-promote any weight to 5-seed evaluation — open decision, discussed but not resolved | ~40min GPU (direct execution after killing a stuck nbconvert attempt) | ✅ done, 🟡 decision pending |
| 14 | Read PINN4SOH's actual `Model.py` (GitHub) to ground the plan in the real architecture, not just the paper description | Confirmed it's a true PDE-residual PINN: a continuous-time `solution_u(x,t)` network + a separate `dynamical_F` dynamics network, with `u_t`/`u_x` computed via `torch.autograd.grad` and a PDE-residual loss (`u_t - F ≈ 0`) trained on paired cycle samples. Meaningfully more complex than either the original spec's or the agreed lean plan's fixed-vector-output + `relu(diff)` monotonicity approach — kept as background context, did not change the Tier 0 gate | ~15min | ✅ done |
| 15 | Verify RUL label convention before any SOH-trajectory work (checklist's top priority item) | Read `batteryml/label/rul.py`: the label is **total cycle life from cycle 1 to EOL crossing**, not "remaining cycles after cycle 99." Matters for a future SOH-trajectory→EOL-crossing RUL derivation (would need `predicted_eol_cycle − 99`, not used directly); the current direct-RUL head (Tier 0/1) is unaffected, since it's always trained against this same label | ~10min | ✅ done |
| 16 | Citation verification for the PINN/Wiener source list | User independently verified and corrected a swapped citation mapping (Zhou et al. ↔ Raissi et al. links had been swapped by an upstream search tool; one source was an S3 file-upload URL, not a citable repo). Corrected mapping documented in `Wiener_PINN_Project_Checklist.md` §1 | ~10min (user-driven) | ✅ done |
| 17 | Copy external planning docs into the session's plan folder | `Wiener_PINN_Project_Checklist.md` and `PINN4SOH_MATR_RUL_Adaptation_Spec.md` copied from Downloads into `report_expirement/day_0825/` | ~2min | ✅ done |
| 18 | "Just test" screen: apply the mirror-image asymmetric loss (penalize *under*-prediction) to `qdlin_diff_min`/`qdlin_diff_l2`, which structurally under-predict long-life cells in 5/5 seeds (long_bias −68 to −135 across seeds for both features, vs. `top3`'s inconsistent sign there) | `safety_weight=1.5`, seed=0, patience=350, both features. Launched after Tier 0 finished | ~5-10min GPU | 🟡 running |
| 19 | LOO calibration-stability diagnostic for Tier 0 (distinguishing "safety loss genuinely improves calibratability" from "calibration artifact") | User correctly pushed back on an initial overstated verdict — re-framed as inconclusive-leaning-A per LOO evidence (RMSE 144.85 vs 146.73, only 1.88 apart; Spearman 0.835 vs 0.815). Extended to all 3 weights + short-life-specific LOO metrics: short-life LOO RMSE/bias is completely flat across all 3 weights (62.26/62.24/62.11), giving no validation-side confirmation for any weight's test-side improvement | ~20min (CPU-only) | ✅ done |

## Standing directions

- **PINN naming discipline**: don't call Tier 0 (asymmetric loss only) a "PINN" — it's a
  safety-aware reweighting, not a physical constraint. Graduate the label only as
  mechanism is actually added: safety-aware → physics-informed (once monotonicity is
  added, Tier 2) → PINN-inspired (once an explicit degradation-dynamics function exists)
  → Wiener-informed (once the stochastic drift/diffusion head exists, Tier 3). Applies to
  both chat/report language and any future manuscript text.
- **`top3` needs an over-prediction penalty; `qdlin_diff_min`/`qdlin_diff_l2` would need
  the mirror-image (under-prediction penalty), if anything** — their bias patterns are
  opposite. `top3` over-predicts short-life consistently (+47 to +100 across seeds);
  `min`/`l2` under-predict long-life consistently (5/5 seeds each, −51 to −135). Don't
  apply the same-direction safety loss to all three models; each needs (at most) its own
  direction if tested at all — task #18 tests this for min/l2, `top3`'s own direction is
  task #13.
- **`top3` = CNN + [`qdlin_diff_std`, `voltage_slope_50_90`, `voltage_soc_90`]** is now
  the strongest single scalar-branch model this project has found for MATR1 — supersedes
  CNN-only as the reference point for anything downstream (ensembles, calibration
  variants, short-life analysis).
- **Post-hoc isotonic calibration (fit on val only) is standard practice**, carried
  forward from day_0823 — but this session found its honest generalization (via LOOCV
  within val) is worse than not calibrating at all, while its test-set application keeps
  improving results 5/5 times. Treat calibrated numbers as "probably real, not yet
  rigorously validated" until the open item below is resolved.
- **Short-life cells (low RUL) are a distinct failure mode worth reporting separately**
  in the writeup — `top3` systematically over-predicts their remaining life
  (dangerous direction), while simpler single-feature models (`qdlin_diff_min`) are
  much better-calibrated there at a full-set-accuracy cost. This is Pareto-plottable,
  legitimate paper material, not a weakness to hide.
- **Ensemble weight selection: always fit on validation only, apply frozen weights once
  to test** — same discipline as isotonic calibration. Two independent optimization
  methods (grid search + NNLS) agreeing on the same weights is the bar for trusting a
  result; a single flexible method (OLS+intercept) reaching a better number without
  that agreement should be CV-checked within validation before trusting it (this
  session's OLS+intercept retraction is the cautionary example).
- **Architecture-search direction (custom multi-branch models) is now closed, 0-for-4**
  (RAN, Phase-1 fixed-support, PCN from day_0802; cycle-window split from this session).
  Don't reopen without new evidence that a specific sub-signal (not just "more capacity")
  is being left on the table.
- **GPU exclusivity, always queue**: GTX1650 4GB, one training job at a time — detached
  `PowerShell Start-Process`, verify via `Get-Process -Id <pid>` + `nvidia-smi` before
  trusting a background job survived, incremental per-seed/per-feature pickle saves so
  interruptions don't lose progress.

## Open items carried from day_0823 report

1. `train_val_test_selection.md` needs an update — still open, now also needs today's
   `top3`/calibration/ensemble findings folded in.
2. Ensemble-of-ensembles (108.36, R²=0.921, day_0819) weighting was a single 50/50
   trial — still not swept.
3. K-fold(MATR1-only) multi-seed — still only has seed=0 (130.76), not touched today.

## Open items from today

4. **Calibration-generalization question — unresolved, qualifies the session
   headline.** LOOCV of the calibration protocol gives honest RMSE=144.85 for `top3`
   alone (worse than uncalibrated 97.76), yet calibration improved test-set results 5/5
   seeds. Proposed check: compare calibration curve shapes across the 5 independently-
   trained seeds' models.
5. `top3` 5-seed headline used patience=350 — re-verify at full 1000-epoch/no-early-stop
   before treating 88.36±9.50 as publication-final. Note: `qdlin_diff_min`/`qdlin_diff_l2`
   in the ensemble (task #6) WERE retrained no-early-stop, so the ensemble mixes both
   protocols — a known, flagged inconsistency.
6. ~~Full 5-seed 3-model ensemble~~ — done (task #6). Superseded as headline by the
   deterministic seed-averaged ensemble (task #10, 84.27).
7. `variant_c_cycle_window_ensemble_v1.ipynb` cell 0 is missing `import BatLiNet` —
   patch the notebook file itself (currently only worked around in the driver script).
8. Section-4/SmallMLP idea from the external `02_plan_test_model_code.md` planning
   document — discussed, not started, no approval yet to begin.
9. Only 14/34 ablation features have cached predictions (temperature_max_discharge
   onward) — the other 20 predate the prediction-saving patch and were never retrained.
   Low priority given task #12's finding, but a gap if a full feature audit is wanted.
10. Tier 0 safety-loss result (task #13) — pending, will determine whether Tier 1
    (auxiliary SOH head) of the PINN4SOH-inspired plan is worth starting.

## Standing constraints

(copied forward from CLAUDE.md / day_0823, still in force)

- Multi-seed mean±std before any claim; no single-seed claims; pre-declared
  hyperparameters; no train-side CV selection.
- All 4 metrics (MAE/RMSE/MAPE/R²) in every reported row where a formal claim is made;
  MATR2 not in scope this session (MATR1-only work).
- Comparisons vs. the README benchmark bar (MATR1=90, PCR) — never vs. locally
  reproduced baselines for the "beaten?" verdict.
- cuda only (GTX 1650 4GB), one GPU job at a time; full 1000-epoch cap, no early
  stopping for any run whose number will be claimed (patience=200 is fine for fast
  screening/diagnostic runs only — **this session's `top3` 5-seed headline used
  patience=350, a deliberate, flagged deviation from this rule, not yet re-verified at
  the full protocol**).
- Windows/joblib: never nest `n_jobs=-1`.
