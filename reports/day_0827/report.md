# Session report — 2026-08-27

Status legend: ✅ done/verified (against official in-process pipeline numbers) ·
🟡 partial or awaiting an external run · ⬜ not started · ❌ attempted, does not
beat baseline.

Plan for the day: [plan.md](plan.md)

Dataset: **MATR1** only. EOL threshold: 0.80. Split: official MATR1
train(91)/val(41)/test(42). Seed=0 for every row below unless the row itself
says "multi-seed".

## Summary table

| Topic | Official baseline (README) | Our best result | Beats baseline? |
|---|---|---|---|
| MATR1 cycle-life RMSE | 90 (PCR) | **RAW mean=93.64±6.10** (5-seed) / **CAL mean=74.66±7.79** (5-seed), Tier2+`qdlin_diff_l2` ensemble, w_l2=0.16 frozen | ❌ RAW does NOT beat baseline once multi-seed confirmed (seed=0's 76.23 was a lucky seed). ✅ CAL beats baseline with ~2 std margin, but carries the standing calibration caveat (41-pt val, LOO check found it doesn't always generalize) |

## All experiments (this session, MATR1, seed=0 unless noted)

| # | Task | Config | MAE | RMSE | MAPE | R² | short_bias | long_bias |
|---|---|---|---|---|---|---|---|---|
| 1 | Tier0 (safety loss) | sw=1.0 (baseline) | 80.30 | 95.42 | 13.36% | 0.9389 | +99.84 | −18.56 |
| 1 | Tier0 | sw=1.5 | 83.52 | 101.42 | 13.66% | 0.9310 | — | — |
| 1 | Tier0 | sw=2.0 | 83.86 | 103.37 | 12.73% | 0.9283 | — | — |
| 1 | Tier0.5 (ranking loss) | λ=0.0 (=baseline) | 80.30 | 95.42 | 13.36% | 0.9389 | +99.84 | −18.56 |
| 1 | Tier0.5 | λ=0.001 | 80.92 | 96.25 | 13.52% | 0.9378 | +101.63 | −13.27 |
| 1 | Tier0.5 | λ=0.005 | 78.47 | 98.75 | 12.28% | 0.9345 | +76.51 | −36.00 |
| 1 | Tier0.5 | **λ=0.01 (best)** | **79.75** | **94.51** | **13.34%** | **0.9401** | +100.75 | −12.57 |
| 1 | Tier0.5 | λ=0.05 | 81.46 | 97.31 | 13.69% | 0.9364 | +102.34 | −9.22 |
| 1 | Tier0.5 | λ=0.1 | 83.57 | 97.92 | 14.27% | 0.9356 | +104.43 | +0.12 |
| 2 | Tier0×Tier0.5 grid | sw=1.5, λ=0.005 | 83.04 | 101.02 | 13.62% | 0.9315 | +99.36 | −28.57 |
| 2 | Tier0×Tier0.5 grid | sw=1.5, λ=0.01 | 80.59 | 95.82 | 12.61% | 0.9384 | +75.14 | −52.74 |
| 2 | Tier0×Tier0.5 grid | sw=2.0, λ=0.005 (CAL=72.85) | 82.35 | 101.00 | 12.54% | 0.9315 | +75.39 | −62.70 |
| 2 | Tier0×Tier0.5 grid | sw=2.0, λ=0.01 | 83.43 | 102.98 | 12.58% | 0.9288 | +75.15 | −69.47 |
| 3 | Tier1 (SOH head) | λ_soh=0.01 | 81.83 | 102.21 | 12.75% | 0.9299 | +81.14 | −41.36 |
| 3 | Tier1 | λ_soh=0.05 | 86.00 | 103.99 | 13.75% | 0.9274 | +86.30 | −44.87 |
| 3 | Tier1 | λ_soh=0.1 | 88.54 | 106.58 | 14.16% | 0.9238 | +88.00 | −46.94 |
| 4 | Tier2 (+monotonicity) | λ_mono=0.001 | 79.53 | 99.40 | 12.08% | 0.9337 | +68.91 | −49.14 |
| 4 | Tier2 | λ_mono=0.01 | 81.18 | 101.22 | 12.58% | 0.9312 | +77.18 | −42.99 |
| 4 | Tier2 | **λ_mono=0.05 (best, CAL=70.93)** | **82.15** | **99.61** | **13.90%** | **0.9334** | **+99.64** | **−8.27** |
| 5 | Tier3 (+Wiener) | λ_w=0.001 | 81.55 | 100.34 | 12.30% | 0.9324 | +71.54 | −65.93 |
| 5 | Tier3 | λ_w=0.01 | 80.33 | 97.05 | 12.12% | 0.9368 | +66.63 | −65.52 |
| 5 | Tier3 | **λ_w=0.05 (best)** | **80.37** | **94.54** | **13.04%** | **0.9400** | +88.86 | −42.91 |
| 6 | Combined Tier0+0.5+1+2 | sw=1.0 | 82.11 | 98.32 | 13.81% | 0.9351 | +104.31 | −11.49 |
| 6 | Combined | sw=1.5 (CAL=71.91) | 82.52 | 100.34 | 13.55% | 0.9324 | +99.08 | −28.62 |
| 6 | Combined | sw=2.0 | 82.22 | 102.15 | 12.77% | 0.9300 | +82.59 | −52.81 |
| 7 | Ensemble | 17-member NNLS | — | 95.64 | — | — | — | — |
| 7 | Ensemble | narrow 4-avg (baseline+T0.5+T2+T3 best) | 79.68 | 94.25 | 13.27% | 0.9404 | — | — |
| 7 | Ensemble | calibrate-then-ensemble (top3+min+l2, 0.5/0.25/0.25, CAL) | 59.91 | 81.42 | 8.34% | 0.9555 | — | — |
| 9 | Headline members | Tier2 alone | 82.15 | 99.61 | 13.90% | 0.9334 | +99.64 | −8.27 |
| 9 | Headline members | `qdlin_diff_l2` alone (under-predict, w=2) | 78.96 | 104.94 | 11.53% | 0.9261 | — | — |
| 9 | **Headline (seed=0 only)** | **Combo RAW, w_l2=0.34 (val-chosen)** | **65.10** | **76.23** | **10.23%** | **0.9610** | **+57.00** | **−18.01** |
| 9 | Headline (seed=0 only) | Combo CAL, w_l2=0.16 (val-chosen) | 46.60 | 67.20 | 6.54% | 0.9697 | — | — |
| 10 | **Headline, 5-seed confirmation** | Tier2 alone | — | mean=101.69±5.59 | — | — | — | — |
| 10 | Headline, 5-seed | `qdlin_diff_l2` alone | — | mean=103.98±15.53 | — | — | — | — |
| 10 | **Headline, 5-seed** | **Combo RAW, w_l2=0.16 frozen** | — | **mean=93.64±6.10** | — | — | — | — |
| 10 | Headline, 5-seed | Combo CAL, w_l2=0.16 frozen | — | mean=74.66±7.79 | — | — | — | — |
| 11 | 3-way ensemble, 5-seed | +3rd model tier05_lam0.1 (w=0.00/0.32/0.68) | — | RAW mean=88.98±7.32 / CAL mean=77.42±8.28 | — | — | — | — |
| 11 | 3-way ensemble, 5-seed | +3rd model tier3_lam0.05 (w=0.54/0.30/0.16) | — | RAW mean=88.54±7.83 / CAL mean=75.11±8.53 | — | — | — | — |
| 12 | Tier1 SOH root-cause | bias vs λ_soh (0→0.01→0.05→0.1) | — | — | — | — | +99.8→+81.1→+86.3→+88.0 | −18.6→**−41.4**→−44.9→−46.9 |
| 13 | Huber-loss investigation, seed=1 | `l2_under` MSE (baseline for this seed) | — | 130.77 | — | — | — | — |
| 13 | Huber, seed=1 | `l2_under` Huber (δ=1.0, weight=2) | — | 123.20 | — | — | — | — |
| 13 | Huber ensemble, seed=1 | Tier2 + Huber-l2under, w=0.16 frozen | — | RAW=98.27 | — | — | — | — |
| 13 | Huber ensemble, seed=1 | Tier2 + Huber-l2under, w=0.08 val-tuned | — | RAW=99.54 / CAL=87.17 | — | — | — | — |
| 13 | Huber+PINN hybrid, seed=1 | Exp C: Tier2-arch + l2 feature + MSE | — | standalone=131.49 / ensemble RAW=99.80 | — | — | — | — |
| 13 | Huber+PINN hybrid, seed=1 | Exp A: Tier2-arch + l2 feature + Huber | — | standalone=129.19 / ensemble RAW=99.89 | — | — | — | — |
| 13 | Huber+PINN hybrid, seed=1 | Exp B: Exp A + ranking loss λ=0.01 | — | standalone=131.20 / ensemble RAW=100.43 | — | — | — | — |
| 14 | Affine vs isotonic, per tier, 5-seed | Tier0 (baseline) | — | RAW mean=101.18±6.61 / ISO mean=88.36±9.50 / **AFF mean=86.24±8.74** | — | — | — | — |
| 14 | Affine vs isotonic, per tier, 5-seed | Tier0.5 (λ_rank=0.1) | — | RAW mean=101.67±3.74 / ISO mean=85.65±9.93 / **AFF mean=85.15±8.45** | — | — | — | — |
| 14 | Affine vs isotonic, per tier, 5-seed | Tier2 (λ_mono=0.05) | — | RAW mean=101.69±5.59 / **ISO mean=82.12±9.07** / AFF mean=84.87±11.20 | — | — | — | — |
| 14 | Affine vs isotonic, per tier, 5-seed | Tier3 (λ_wiener=0.05) | — | RAW mean=104.77±5.52 / **ISO mean=84.90±2.69** / AFF mean=93.57±5.30 | — | — | — | — |
| 14 | Affine vs isotonic, per tier | Tier1 (SOH head) | — | not calibratable — no `val_pred` saved (see Open Items #5) | — | — | — | — |
| 14 | Affine on headline ensemble, 5-seed | tier2+l2 combo, w=0.16 frozen | — | RAW mean=93.64±6.10 / ISO mean=**74.66±7.79** / AFF mean=81.59±10.17 | — | — | — | — |
| — | README benchmark | PCR | — | 90 | — | — | — | — |

## Key findings (narrative, no repeated tables)

- **Task #8 — per-cell error correlation audit**: baseline/Tier2/combo (all
  top3-CNN-family) are ~0.95 correlated on test — redundant. `qdlin_diff_min`/
  `qdlin_diff_l2` are **negatively** correlated with that family (−0.21 to
  −0.42) — genuine complementary errors that the 17-member NNLS (fit on only
  41 val points) had been zeroing out, explaining why the broad ensemble
  underperformed baseline.
- **Task #9/#10 — the headline does NOT survive multi-seed on RAW.** Seed=0's
  76.23 looked like a clean win, but the honest 5-seed mean is 93.64±6.10 —
  fails the README bar (90). **CAL does survive**: 74.66±7.79, clearing 90
  with ~2 std margin — but every CAL number in this report carries the
  standing calibration caveat below. This is the clearest demonstration this
  session of why the project's multi-seed rule exists.
- **Task #11 — adding a 3rd ensemble member does not help.** Tested every
  plausible 3rd member (Tier0, all Tier0.5/2/3 λ, `qdlin_diff_min`) via honest
  val-only weight search; only `tier05_lam0.1` and `tier3_lam0.05` got
  non-trivial weight. Full 5-seed confirmation: RAW mean improves slightly
  (88.98/88.54 vs 93.64) but std grows too (7.3–7.8 vs 6.1) — no clean
  separation from 90. CAL is **not** improved by either 3-way variant (2-way's
  74.66±7.79 stays best). Verdict: mixed, not decisive — not adopted as a
  new headline.
- **Task #12 — Tier1 (SOH-aux head) root cause found.** It is the only tier
  that loses to baseline at every λ. Per-cell breakdown shows this is a
  **step effect, not a gradual one**: long_bias jumps from −18.6 to −41.4 the
  moment the SOH loss is turned on at all (λ=0.01), then stays flat
  (−41→−45→−47) as λ increases 10×. Interpretation: the auxiliary SOH head
  shares the backbone with the RUL head: any nonzero λ_soh is enough to pull
  the shared representation toward the majority (short-life) pattern, at the
  cost of extrapolation to the long-life tail — the same underlying weakness
  documented in Task #13's root-cause work on seed 1. Viz saved at
  `tier1_soh_bias_viz.png` (this folder).
- **Task #13 — full Huber-loss investigation, seed=1, now closed with a clean
  negative result.** Motivated by seed 1 being the multi-seed run's worst
  seed (a single long-life test cell, true=2237 cycles, `l2_under` error
  −424.1 there vs seed 0's −72.8 — traced to range-compression in the
  val-selection checkpoint itself, confirmed not a pure single-point artifact
  via top-1/top-2 outlier-exclusion tests). Tried 5 variants, none beat the
  original MSE-based ensemble (98.01 RAW):
  - Huber alone (simple arch): 123.20 vs MSE's 130.77 — modest per-member
    gain (−5.8%), but the worst-cell error only shrank −424→−391 (~8%).
  - Ensembled with Tier2 (frozen w=0.16): 98.27 — essentially a wash.
  - Ensembled with Tier2 (val-tuned w=0.08): 99.54 — tuning on this seed's
    own tiny val set made it *worse*, another small-n instability data point.
  - Grafting `qdlin_diff_l2` + asymmetric Huber onto Tier2's SOH-aux+
    monotonicity architecture (Exp A/B/C, with/without ranking loss): all
    three came out **worse than the simple-arch Huber alone** (129–131
    standalone) — the architecture that helps Tier2 (101.46 with its
    original top3 features) does not transfer when fed a single scalar
    feature; it needs richer input to earn its keep.
  - **Conclusion: loss function is not the root cause.** The real limiter is
    extrapolation to rare long-life cells with only 91 train cells — a
    data/architecture ceiling, not something a loss-function swap fixes.
    This closes the loss-shaping/PINN track; next session moves to the
    architecture-search plan (staged custom backbones — axis-aware pooling →
    CNN-per-cycle+temporal → ConvTransformer — deliberately not the library
    `RULPredictor`s, see that plan's rationale). See
    [day_0901/plan.md](../day_0901/plan.md).
- **Task #14 — affine calibration tested as an isotonic alternative (added
  2026-08-28, day_0901 session), no universal winner.** Affine =
  `calibrated = a*raw_pred + b`, least-squares fit on val only (2 params,
  vs. isotonic's free monotonic step function) — the hypothesis was that
  fewer params overfit the 41-point val set less. Result: **no consistent
  winner, it depends on whether each model's own bias is linear or not.**
  - Per single tier (Tier0/0.5/2/3, 5-seed each): affine wins Tier0 (86.24
    vs 88.36) and Tier0.5 (85.15 vs 85.65, narrow); isotonic wins Tier2
    (82.12 vs 84.87) and Tier3 clearly (84.90 vs 93.57 — Tier3's bias is
    visibly non-linear, isotonic's flexibility earns its keep there).
  - On the **headline ensemble itself** (tier2+l2, w=0.16 frozen):
    isotonic wins clearly, 74.66±7.79 vs affine's 81.59±10.17 — affine
    loses on 5/5 seeds. The headline's calibration method stays isotonic.
  - Also tried: NNLS-searched (per-seed dynamic) ensemble weight in place
    of the frozen w=0.16, both with and without a 3rd member
    (tier05_lam0.1 / tier3_lam0.05), then affine-calibrated. Every variant
    tested worse than the frozen-weight+isotonic headline (best new
    combo: `tier2+l2+tier3_lam0.05` = 81.52±9.01). NNLS improves RAW
    substantially and consistently (84.18–86.25 vs 93.64, 5/5 seeds) but
    that gain does not survive calibration — the dynamic weight search and
    the calibration step are both correcting the same linear bias, so
    combining them "double-dips" and nets out worse than leaving the
    weight fixed and letting calibration do that job alone.
  - **Standing rule going forward**: no default calibration method — pick
    whichever (affine or isotonic) wins on that candidate's own val RMSE,
    never fix one method project-wide. Full per-seed numbers, and the
    NNLS-weight combo grid, in task #14's table above.
- **CAL caveat (applies to every CAL number in this report)**: isotonic
  regression fit on a 41-point val set — an earlier LOO check on a similar
  ensemble found honest val RMSE 151.25 vs. raw 97.57. RAW is the
  trustworthy number; CAL is directional only.

## Open items

1. ~~Multi-seed the session headline~~ — **done**, see task #10. Result:
   RAW fails README (93.64±6.10), CAL clears it (74.66±7.79, caveat applies).
2. ~~3rd ensemble member~~ — **done**, see task #11. Mixed/not decisive,
   2-way stays the reported headline.
3. ~~Tier1 root cause~~ — **done**, see task #12 and `tier1_soh_bias_viz.png`.
4. ~~Huber-loss / "Huber+PINN" track~~ — **done and closed**, see task #13.
   Negative across every variant tried; loss function ruled out as the fix
   for the long-life extrapolation weakness.
5. Tier 1's 3 results have no `val_pred` (bug, fixed for Tier2/3 onward) — low
   priority, Tier1 already fails its RAW gate.
6. Two flagged Tier0+0.5 grid candidates from day_0826 remain open if that
   angle is revisited: sw=1.5/λ=0.005 (best CAL, 72.85) and sw=1.5/λ=0.01
   (best RAW, 95.82).
7. **Next session**: architecture-search track — see
   [day_0901/plan.md](../day_0901/plan.md) for the carried-forward plan
   (staged custom backbones, cheapest first, one step at a time — library
   `RULPredictor`s deliberately not used, see that plan for why).
8. ~~Affine vs isotonic calibration~~ — **done**, see task #14. No universal
   winner; affine wins Tier0/Tier0.5 and isotonic wins Tier2/Tier3/the
   headline ensemble — pick per-candidate on val RMSE, not a fixed rule.
