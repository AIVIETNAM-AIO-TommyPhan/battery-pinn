# Plan — 2026-08-27

Carried over from [day_0826/plan.md](../day_0826/plan.md). That session's task #6
had inserted **Tier 0.5** (cross-cell ranking loss) into the staged PINN plan
(`day_0825/Wiener_PINN_Project_Checklist.md`) and run its Step-1 diagnostic only.
This session runs Tier 0.5's actual training sweep, then continues the staged
plan through Tier 1 (SOH head), Tier 2 (monotonicity), and Tier 3 (Wiener) —
closing out the PINN/loss-shaping track that day_0825/0826 opened.

## Scoreboard going in

| Dataset | Status | Ours | README target |
|---|---|---|---|
| MATR1 | ❌ PINN/loss-shaping track fully closed — no config beats README on RAW once multi-seeded (CAL beats it, with the standing caveat) | **RAW 93.64±6.10** / **CAL 74.66±7.79** (5-seed, Tier2+l2 ensemble, w=0.16 frozen — task #10) | 90 (PCR) |

## Task table (easiest → hardest)

| # | Task | What / why | Est. | Status |
|---|---|---|---|---|
| 1 | Tier 0.5 training sweep, λ_rank ∈ {0, 0.001, 0.005, 0.01, 0.05, 0.1}, seed=0 | λ=0 reused from baseline (95.42, not retrained — definitionally identical). Best: **λ=0.01 → 94.51** (only lever this session that beat baseline on RAW alone without a big long_bias cost). λ=0.005 is the outlier: short_bias +76.51 (best short-life move) but long_bias −36.00 (worst long-life move) | ~5×25min GPU | ✅ done |
| 2 | Combined grid: safety_weight(Tier0) × λ_rank(Tier0.5), {1.5,2.0}×{0.005,0.01} | Best RAW: sw=1.5,λ=0.01 → 95.82 (short_bias 75.14, long_bias −52.74). Best CAL: sw=1.5,λ=0.005 → 72.85 (but that combo's RAW was the worst of the grid, 101.02) | ~4×25min GPU | ✅ done |
| 3 | Tier 1 (SOH auxiliary head), λ_soh ∈ {0.01,0.05,0.1}, seed=0 | All three worse than baseline on RAW (102.21/103.99/106.58), monotonically worse as λ_soh increases. Short_bias improves some (+81 to +88 vs baseline +99.84), long_bias worsens a lot (−41 to −47 vs −18.56). Ran before the val_pred-saving fix — **cannot be calibrated without retraining** | ~3×25min GPU | ✅ done (calibration gap noted) |
| 4 | Tier 2 (+monotonicity on SOH head), λ_mono ∈ {0.001,0.01,0.05}, λ_soh=0.01 fixed | λ=0.05 is the standout: RMSE=99.61, short_bias +99.64 (flat vs baseline), but **long_bias −8.27 — the only config all session to improve long_bias without hurting short_bias**. Best individual CAL of the whole PINN track: **70.93** | ~3×25min GPU | ✅ done |
| 5 | Tier 3 (+Wiener drift/diffusion), λ_wiener ∈ {0.001,0.01,0.05}, safety_weight=2.0 fixed | Best: λ=0.05 → RMSE=94.54 (best RAW of the pure-tier sweeps), short_bias +88.86, long_bias −42.91 | ~3×25min GPU | ✅ done |
| 6 | Combined Tier0+0.5+1+2 in one model: safety_weight ∈ {1.0,1.5,2.0} × fixed λ_rank=0.01/λ_soh=0.01/λ_mono=0.05 | No config beat Tier 2 alone. sw=1.5's CAL (71.91) came closest to Tier2's 70.93. sw=2.0 gave the best short_bias of any single model (+82.59) at the cost of the worst long_bias (−52.81) | ~3×25min GPU | ✅ done |
| 7 | Ensemble sweeps across every tier/config saved today (up to 17 members with val_pred) | Broader ensembles underperform narrower ones — 17-member NNLS nearly degenerated to 2 members and was *worse than baseline* (95.64). Best narrow ensemble: simple-average of {baseline, tier0.5-best, tier2-best, tier3-best} → RAW=94.25. Calibrate-then-ensemble (top3+min+l2 trio, fixed 0.5/0.25/0.25) → 81.42, beating both other orderings tried | ~30min analysis | ✅ done |
| 8 | Per-cell error correlation audit across all season's models, prompted by "is there real complementarity to ensemble?" | **Key finding**: baseline/tier2/combo (all top3-CNN-family) are ~0.95 correlated — redundant. `qdlin_diff_min`/`qdlin_diff_l2` are **negatively** correlated with the top3-family (−0.21 to −0.42) — genuine complementary errors. NNLS on the 41-pt val set had been zeroing out min/l2 entirely, missing this | ~15min analysis | ✅ done |
| 9 | Forced-weight exploit of the negative correlation: Tier2 + `qdlin_diff_l2`(weight=2.0), weight on val only, frozen to test | Seed=0 candidate: RAW=76.23, CAL=67.16. **Superseded by task #10's multi-seed result** — see below | ~20min analysis (+ a caught-and-fixed test-tuning mistake, see Open Items) | ✅ done, superseded |
| 10 | Multi-seed confirmation of task #9, 5 seeds, w_l2=0.16 frozen | **RAW mean=93.64±6.10 — fails README (90).** CAL mean=74.66±7.79 — clears README, caveat applies. This is the session's real headline number | ~2×25min×5 GPU | ✅ done |
| 11 | 3rd ensemble member search + 5-seed confirmation (tier05_lam0.1, tier3_lam0.05) | Mixed/not decisive: RAW mean improves (88.98/88.54) but std grows (7.3-7.8); CAL not improved by either. 2-way stays the reported headline | ~2×25min×4 GPU | ✅ done, not adopted |
| 12 | Tier1 root-cause investigation (why does SOH-aux head lose to baseline at every λ?) | Step effect, not gradual: long_bias jumps −18.6→−41.4 the instant λ_soh>0, then flat. Shared-backbone interference, not a scale-with-λ problem. Viz: `tier1_soh_bias_viz.png` | ~20min analysis | ✅ done |
| 13 | Huber-loss / "Huber+PINN" investigation (seed=1, the multi-seed run's worst seed) | **Closed, negative.** 5 variants tried (Huber-alone, ensembled frozen/val-tuned w, Huber+PINN-arch hybrid ×3), none beat the original MSE ensemble (98.01). Loss function ruled out as root cause — real limiter is long-life extrapolation with n=91 train | ~6×25min GPU | ✅ done, closed |

## Standing directions

- **Preprocessing is non-negotiable for reproducing any baseline number**: `diffed = clean_feature(feat - feat[:, :, [DIFF_BASE=9]])` before the CNN — confirmed straight from `run_one_scalar_safety`'s source this session. Any new architecture that skips this needs its own fresh baseline, not a comparison to 95.42.
- **Every tier this session repeated the same pattern**: pushing short-life bias down costs long-life bias, and vice versa — no single loss lever escaped this trade-off. Tier 2 λ=0.05 and today's ensemble discovery (task #9) are the only two exceptions found (both improve one side while leaving the other roughly flat).
- **`val_pred` must be saved by every new training script** — Tier 1's 3 results can't be calibrated because this was missed; caught and fixed mid-session for Tier 2/3 (see Open Items in day_0826 for the earlier version of this same lesson).
- **A separate architecture-ladder track opened today**, deliberately kept apart from this PINN/loss-shaping track: see [day_0901/plan.md](../day_0901/plan.md), [day_0901/Project_Architecture_Ladder_Spec.md](../day_0901/Project_Architecture_Ladder_Spec.md) (Track A — project tensor, `SmallCNNScalarBranch`-based ladder: axis-aware pooling → CNN-per-cycle+temporal → ConvTransformer, ConvLSTM1D added as a screened variant not a mandatory first step) and [day_0901/Thesis_Aligned_Architecture_Ladder.md](../day_0901/Thesis_Aligned_Architecture_Ladder.md) (Track B — backlog, not started, predicts a different target so not comparable to 95.42/90).

## Open items carried into next session

1. ~~Multi-seed the session headline~~ — done, task #10.
2. **Tier 1's 3 results have no `val_pred`** — either retrain (~75min) for completeness or accept the calibration gap; Tier 1 is not the headline candidate so this is low priority.
3. **Caught mid-session**: an early version of the task #9 weight search picked `w_l2` by directly minimizing *test* RMSE (73.88) — a methodology violation. Corrected by re-picking the weight on val only (0.34) before reporting the frozen test number (76.23). No results derived from the tainted test-tuned weight were kept as claims.
4. Two flagged Tier0+0.5 combined-grid candidates from day_0826, still open if that specific angle is revisited: `safety_weight=1.5, lambda_rank=0.005` (best calibrated then, 72.85) and `safety_weight=1.5, lambda_rank=0.01` (best raw then, 95.82).
5. **Session closed.** PINN/loss-shaping track (day_0825→day_0827) is done —
   headline is RAW 93.64±6.10 (fails 90) / CAL 74.66±7.79 (beats 90, caveat).
   Huber-loss branch explored and closed negative (task #13). Next session
   moves to the architecture-search track — see
   [day_0828/plan.md](../day_0828/plan.md).

## Standing constraints

- **Sanity gate**: reproduce a known number exactly (or within a stated tolerance) before attributing any change to it — used throughout (λ=0 reuse instead of retraining; the Architecture Ladder Spec's Stage 0 gate).
- **Metrics**: RMSE, MAE, MAPE, R² together, plus this track's own short_bias/long_bias (bias = pred − true, by life tercile).
- **Multi-seed discipline**: seed 0 for screening only; nothing is a claim until confirmed on the full seed set with mean±std, never a best-seed cherry-pick — this is exactly why task #9 stays 🟡.
- **Never tune a weight or hyperparameter on test** — violated briefly and caught within the same session (Open Items #3); the frozen, honestly-derived number is the only one carried forward.
- **Windows/joblib**: never nest `n_jobs=-1`.
- **GPU**: one training job at a time (4GB card) — a background job died silently once this session when this was nearly violated; always confirm no other python training process before launching.
- **Benchmark bar**: MATR1 README = 90 (PCR). Task #9's RAW=76.23 is the first number this session to clear it — pending multi-seed before it can be claimed as such.
