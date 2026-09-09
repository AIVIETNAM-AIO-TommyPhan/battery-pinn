# Plan — 2026-09-01

New direction, separate from the loss-shaping/PINN track (Tier 0.5→3, run on
2026-08-27). **That track is now fully closed** — see
[day_0827/report.md](../day_0827/report.md) for the complete write-up. Final
numbers: RAW mean=93.64±6.10 (5-seed, fails README=90), CAL mean=74.66±7.79
(clears 90, standing calibration caveat applies). A follow-up investigation
into whether a Huber loss (instead of MSE) fixes the long-life extrapolation
weakness was also run and closed negative — 5 variants tried, none beat the
original ensemble; loss function was ruled out as the root cause. This
session's goal: find a backbone that beats
`SmallCNNScalarBranch` (95.42 RMSE) by exploiting the cycle-axis / voltage-
position-axis distinction the current model ignores, staged cheapest-first.
Full architecture spec, ablation tables, exact pseudocode, and decision gates
are in [Project_Architecture_Ladder_Spec.md](Project_Architecture_Ladder_Spec.md)
— this file tracks status only; read the spec before implementing any stage.
A separate, not-yet-coded backlog outline for a literal thesis-pipeline
reproduction (different prediction target, not comparable to 95.42/README=90)
lives in [Thesis_Aligned_Architecture_Ladder.md](Thesis_Aligned_Architecture_Ladder.md).

## Scoreboard going in

| Dataset | Status | Ours | README target |
|---|---|---|---|
| MATR1 (this track) | ✅ Stage 0-4 done, backbone selected — ✅ beats README on 5-seed mean (caveat: wide std) | **V2** (axis-aware attention pooling, no PE) = **82.85 ± 11.43 RMSE raw** (5-seed), **74.10 ± 9.46 affine-calibrated** — beats both Stage 0 baseline (101.18±6.61, ~19% better) and README=90 on the RAW 5-seed mean itself (best single seed, seed4, = 65.97). Caveat: std is wide (±11.43) driven by a seed2 outlier (attention collapse, root-caused) — not yet as stable a claim as the margin suggests. No later stage/tier has beaten V2. Task #7 (PINN re-test on V2) in progress. | 90 (PCR) |

## Task table (easiest → hardest)

| # | Task | What / why | Est. | Status |
|---|---|---|---|---|
| 1 | Stage 0 — sanity baseline, `SmallCNNScalarBranch`, 5-seed | **Done.** Corrected a protocol error in this file/spec before running: the historical 95.42 was produced with `patience=350`, not `patience=1e9` (confirmed from `variant_c_tier0_safety_loss_v1.ipynb` source + `day_0825/report.md`) — running with `patience=1e9` would not have reproduced it. Fresh seed=0 run (patience=350) gave RMSE=95.4175 vs reference 95.42 (diff=0.0025) → gate PASS. Seeds 1-4 reused from `top3_multiseed_results.pkl` (same model/protocol/code path, deterministic seeding) instead of retraining, saving ~1.5-2h GPU time. **Result: RMSE 101.18±6.61, MAE 82.39±6.92, R²=0.931±0.009, short_bias +83.66±18.79, long_bias +9.49±24.92, n_params=15,153.** Saved to `ipynb/exp_v3/matr1_feature_cache/stage0_smallcnn_scalar_branch_5seed.pkl`. See `report.md`. | ~25min (1 fresh run) | ✅ done |
| 2 | Stage 1 — axis-aware pooling head (V1-V4), screen seed=0 | **Done.** V1 (PE+attention)=81.77 pass, V2 (attention, no PE)=75.32 pass (best), V3 (PE+conv1d-pool)=91.35 pass, V4 (PE-only-mean)=96.80 fail (short_gain 14.88<15, just missed) | 4 run × ~25min | ✅ done |
| 2b | Stage 1 — confirm, 5-seed | **Done, both V1 and V2** (user approved confirming both, not just best-RMSE, since V1 had the better bias profile). **V1: RMSE 89.52±9.53** (seed4 outlier, long_bias -87.13). **V2: RMSE 82.85±11.43** (seed2 outlier, root-caused to attention collapsing onto cycles 45-47 for nearly every test cell, argmax_std=0.51 vs seed0's 1.41). Both `stage1_promote=True`. **V2 is the session's best backbone.** | 8 run × ~15-20min | ✅ done |
| 3 | Stage 2 — CNN-per-cycle + temporal aggregation (V1-V4), screen seed=0 | **Done** (run regardless of Stage 1 promoting, per user request to see every stage). V1 (mean)=92.05 pass, V2 (attn+PE)=94.56 pass, V3 (causal LSTM)=97.61 **fail**, V4 (ConvLSTM1D)=90.18 pass (best bias profile of Stage 2: short_bias+25.38, long_bias-4.63, but ConvLSTM training is visibly unstable — loss/val jump around before settling). None beat V2 (75.32/82.85 5-seed) | 4 run × ~15-70min (ConvLSTM slow: 100-step Python loop per forward) | ✅ done |
| 3b | Stage 2 — confirm, 5-seed | Not run — Stage 2 didn't beat Stage 1, no candidate worth the 5-seed cost | — | ⬜ not started (not needed) |
| 4 | Stage 3 — ConvTransformer-inspired (V1-V3), screen seed=0 | **Done.** V1 (causal conv only)=120.68 **fail** (short_bias +111.76, worse than baseline), V2 (attention only)=101.23 **fail** (short_bias +112.19), V3 (conv+attn)=112.06 **fail**. **All 3 failed** — first stage with zero passing variants | 3 run × ~20-30min | ✅ done |
| 4b | Stage 3 — confirm, 5-seed | N/A — no screen pass | — | ⬜ not applicable |
| 5 | Backlog (optional) — GCN/MvNGCN-inspired | **Gate not met**: neither Stage 2 nor Stage 3 promoted (`stage2_promote=False`, `stage3_promote=False`) — backlog stays closed | — | ⬜ not opened (gate closed) |
| 6 | Stage 4 — summary + decision | **Done.** Ranking eligible candidates (rmse_mean, then bias gain, then params, then train_seconds) across every stage's seed-0/5-seed result: **V2 (Stage 1, no PE, attention pooling) wins outright** — lowest RMSE at every calibration level (raw 82.85±11.43, isotonic 78.27±5.06, affine 74.10±9.46), beats Stage 0 baseline (101.18±6.61) by ~19%. No later stage (2 or 3) beat it; Stage 3 failed entirely. **V2 is the selected backbone for this track.** Known open weakness: seed-to-seed variance from attention collapsing onto 1-2 end-of-sequence cycles for an unlucky seed (seed2, RMSE 99.58) — confirmed via attention-weight diagnostics, not yet solved (an entropy-regularization fix was tried and made things worse, see report.md) | ~30min, no training | ✅ done |
| 7 | Post-selection — PINN + feature additions, seed=0 screen | **Unblocked by Task #6, in progress.** Tier0 (safety-weight) on V2: **FAIL** at both sw=1.5 (75.32→93.48) and sw=2.0 (75.32→100.93, long_bias flips to -25.86) — same failure mode as the old backbone, just via a different path (V2's favorable same-signed bias didn't protect it once sw was strong enough). Tier0.5/1/2/3 (ranking loss / SOH head / +monotonicity / +Wiener) running now on V2, exact loss formulas reused verbatim from `day_0825/Wiener_PINN_Project_Checklist.md` (the only surviving spec — original training code no longer exists). Single run per tier at the old track's best-known λ (not re-swept) | 5 run × ~15-30min | 🟡 in progress |

## Standing directions

- **Scope**: applies to `SmallCNNScalarBranch` (95.42 baseline, has top3 scalar branch) — never the CNN-only `SmallCNN` (130.83 baseline, no scalar branch). Scalar branch (top3: `qdlin_diff_std`, `voltage_slope_50_90`, `voltage_soc_90`, in that order) is kept in every stage's architecture, always processed through the existing `Linear(3,16)→ReLU→Linear(16,16)→ReLU` before concat — never concatenated raw.
- **Preprocessing is baseline-defining, not optional**: every stage's model input is `clean_feature(x - x[:, :, [9], :])` (`DIFF_BASE=9`), never the raw `feature_cache.pkl` tensor directly. Confirmed from `run_one_scalar_safety`'s actual source this session — this was the single biggest gap caught during spec review and is now fixed in `Architecture_Ladder_Spec.md`.
- **Protocol pinned** (do not vary except as a declared ablation): `EPOCHS=1000`, `EVAL_EVERY=50`, `PATIENCE=350` (**corrected 2026-09-01** — the historical 95.42 run used `patience=350`, not `1e9`; `Project_Architecture_Ladder_Spec.md`'s original pin was an undetected error, caught by Stage 0's fresh reproduction. Fresh seed=0 hit its best-val checkpoint at epoch 200, matched 95.42 exactly, then plateaued/rose through epoch 550 where patience=350 correctly stopped it), `AdamW(lr=1e-3)` with no explicit `weight_decay` (PyTorch default 0.01), seeds `[0]` for screening / `[0,1,2,3,4]` for confirmation.
- **Decision gate** (`passes_primary_gate()` in the spec): `rmse_new <= rmse_base*1.10 AND (short_bias_gain>=15 OR long_bias_gain>=15)`, OR `rmse_new <= rmse_base*0.95` — evaluated on the **mean of the seed set**; `std` always recorded, and any candidate with high variance/seed-failure gets flagged `unstable=True` and is never promoted regardless of mean.
- **Sequential execution only**: never two training jobs at once on the 4GB GPU (established the hard way this session — a background job silently died once when this was violated).
- **`T_rest` (relaxation/off-cycle timing) confirmed NOT available**: `time_in_s` resets to 0 every cycle in the processed pickles, no absolute cross-cycle timestamp exists. Dropped from the plan entirely, not carried as "if available."
- **Deliberately not using** `CNNRULPredictor`/`LSTMRULPredictor`/`TransformerRULPredictor` (library models) as-is: their raw per-cycle input is `6×1000=6000`-dim, which would need hundreds of thousands of parameters just for the first LSTM/Transformer projection layer — high overfitting risk on 91 training cells, and exactly the kind of capacity blowup that made the old `YOriOnly` (1.13M params) seed-unstable before `SmallCNN` replaced it. Stage 2/3 compress each cycle to a 32-dim token via a small Conv1d encoder first instead.
- **Real compute-time note**: today's ~1300-1500s/run figures (used for the "número run × phút" estimate) came from early-stopped runs (`patience=200`). This track's Stage 0+ use `patience=1e9` (full 1000 epochs, no early stop) — expect roughly 2.5-3x longer per run (~60min, not ~25min) until Stage 0's actual `train_seconds` is measured.
- **Thesis-alignment review (resolved)**: user proposed re-ordering this whole track to match a specific thesis's pipeline (channel-wise 11-timepoint + cycle-wise C/P/x3/x5 features, L=10 sliding window, ConvLSTM→BiAR-SeqInSeq→BFPE-AR→ARNS→CTARNS→MvNGCN, next-cycle SoH forecasting target). Rejected as a full replacement: the thesis's task (predict next-cycle capacity from a 10-cycle window) is a *different problem* than this project's (predict total life from cycles 0-99) — its RMSE numbers aren't comparable to 95.42/README=90, and building its feature pipeline (11-timepoint channel-wise) is a large new research effort with no evidence it transfers to MATR data. **Resolution**: keep this track's task/target/preprocessing exactly as-is, only borrow the *architectural idea* of ConvLSTM — added as Stage 2's V4 (`ConvLSTM1D`, a genuine fused conv+recurrent cell keeping the per-cycle spatial width intact, not a flattened-vector strawman), screened fairly alongside V1-V3 under the same gate. Not elevated to a mandatory first phase, given `BatteryML/BatLiNet/model_convlstm.py`'s prior 0/3 loss to plain LSTM/CNN-Attn-LSTM on MATR-family data — local empirical evidence outweighs a different dataset/task's reported numbers.

## Open items carried from day_0827 (loss-shaping track — closed, see report.md)

1. Tier 0.5 (ranking loss) + Tier 1 (SOH head) + Tier 2 (monotonicity) + Tier 3
   (Wiener) all fully swept and written up. No single tier beat README
   cleanly; the ensemble discovery (Tier2 + `qdlin_diff_l2`) did on RAW at
   seed=0 (76.23) but failed multi-seed confirmation (93.64±6.10). CAL
   (74.66±7.79) clears README with the standing calibration caveat. This is
   the number to cite if anyone asks "what did the loss-shaping track reach."
2. Two flagged combined-grid candidates from that track, still open if it's
   ever revisited: `safety_weight=1.5, lambda_rank=0.005` (best calibrated,
   72.85 — but RAW was the worst of the grid, 101.02) and
   `safety_weight=1.5, lambda_rank=0.01` (best RAW, 95.82, short_bias 75.14
   but long_bias -52.74).
3. Huber-loss follow-up (motivated by seed-1 being the multi-seed run's worst
   seed): closed negative. Root cause of the extrapolation weakness is a
   data/architecture ceiling (91 train cells, few long-life examples), not
   the loss function — this is part of the motivation for this
   architecture-ladder track existing at all.

## Note on library RULPredictors (resolved, do not revisit lightly)

A separate plan draft (`gleaming-nibbling-pony.md`, not part of this file's
staged ladder) proposed trying `CNNRULPredictor`/`LSTMRULPredictor`/
`TransformerRULPredictor` from `batteryml/models/rul_predictors/` directly, in
place of Stage 2/3's custom small encoders. **This was considered and
rejected** — see "Deliberately not using..." in Standing Directions above:
those library models' raw per-cycle input is 6000-dim, requiring hundreds of
thousands of parameters at the first layer, which is exactly the capacity
blowup that made the old `YOriOnly` (1.13M params) seed-unstable. Stage 2/3's
Conv1d-encoder-first approach exists specifically to avoid that. Follow this
file's staged plan, not the library-predictor shortcut, unless a specific new
reason to revisit is documented first.

## Standing constraints

- **Sanity gate discipline**: any rebuilt pipeline must reproduce its known number exactly (or within a stated tolerance) before attribution of any change — this is why Stage 0 exists as its own gated step.
- **Metrics**: report RMSE, MAE, MAPE, R² together; here also short_bias/long_bias (bias = pred − true, split by life tercile) per this track's own gate.
- **Multi-seed discipline**: seed 0 for screening only; a candidate is never claimed until confirmed on the full seed set, mean±std reported, never a best-seed cherry-pick.
- **Windows/joblib**: never nest `n_jobs=-1`.
- **Benchmark bar**: MATR1 README = 90 (PCR) — the number every claim here is ultimately judged against, not just the 95.42 in-process baseline.
