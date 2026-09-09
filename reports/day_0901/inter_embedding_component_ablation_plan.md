# Inter-Embedding Model — Component Ablation Plan

> Sub-track of `intercell_roadmap_plan.md` (Stage 1: Inter-Embedding Model).
> Method: hold every component fixed except one, vary that one component,
> select by validation RMSE only, then move to the next component. All
> results below are **seed 0 only** — multi-seed confirmation is explicitly
> deferred (user directive), so nothing here should be read as a confirmed
> result, only as a screen outcome per this project's own convention.

## Context

Starting point: `run_inter_embedding_stage1.py` (seed 0) found that a
V2-embedding-based pairwise Δz model, inferred via 8 fixed reference cells,
beats V2 solo: **val=84.90, test=68.41** vs V2's 75.32 (a 9.2% improvement).
It failed the *ensemble* gate with V2 (error correlation 0.923 — too similar
to be a useful NNLS partner, since it's built from V2's own embedding), but
stood on its own as a candidate alternative to V2. This plan is the
systematic follow-up: vary each pipeline component to see whether a better
configuration exists, and whether the original 68.41 is robust or fragile.

## Component ablation table

| # | Component | Status | Result |
|---|---|---|---|
| 1 | **Encoder / feature set** feeding V2's scalar branch (currently fixed top-3) | ⬜ Not run | Queued: full 34-feature single-ablation would cost ~8-9h GPU (34× V2 retrain + Inter-Embedding train); scoped-down alternative (top 7-10 features by the existing absolute-RUL ablation) costs ~2h. Not started — see priority note below. |
| 2 | **Pair representation** (`diff`, `concat`, `concat_diff`, `absdiff_concat`, `product`, `sqdiff_concat`, `ratio*` — dropped per review) | 🟡 Script written (`run_inter_embedding_delta_variants.py`), not run to completion — superseded by testing head architecture first | — |
| 3 | **Delta-head architecture** (H0 linear → H3 potential-based φ(hᵢ)-φ(hⱼ)) | ✅ Done (`run_inter_embedding_head_ladder.py`) | H0 (linear, 49 params) won: val=87.93, test=76.96. Monotonically worse as capacity increases (H1=89.55, H2=108.13, H3=145.59 on val). |
| 4 | **Reference aggregation** (uniform mean vs. learned attention/confidence-weighted) | ✅ Done (`run_inter_embedding_attention_agg.py`) | Uniform mean won. Attention-weighted: val=94.11, test=86.34 — worse than uniform mean (87.93/76.96). |
| 5 | **Reference-set selection** (currently: 8 cells fixed, 3 short/3 mid/2 long RUL tercile, seed=0, drawn once) | ⬜ Not run | Untested: different R, different selection rule (e.g. per-target nearest-neighbor references instead of one fixed global set), or averaging over multiple reference-set draws to quantify reference-choice variance. |
| 6 | **Embedding standardization** (train-only z-score vs. raw/unstandardized) | ✅ Done (implicit in #3's setup vs. the original Stage-1 script) | **Raw (unstandardized) won** — the original Stage-1 result (68.41, unstandardized) beat every standardized variant tried in the H0-H3 ladder (best was 76.96). Counter to the a priori expectation that standardization should help. |
| 7 | **Loss function** (MSE vs. Huber vs. cycle-consistency auxiliary term) | ⬜ Not run | Discussed only; not implemented. Cycle-consistency (Δz_ij + Δz_jk = Δz_ik) is an exact identity already satisfied by construction by the H3 potential-model architecture (component #3), so a separate auxiliary loss would only matter for non-potential heads (H0-H2). |
| 8 | **Multi-seed confirmation** of the winning configuration | ⬜ Explicitly deferred | User directive: single seed only, no multi-seed run, for this sub-track. |

## Key finding so far (seed 0 only — read with appropriate caution)

**Every "more rigorous" change tried (standardization, antisymmetric-by-construction
architecture, learned attention aggregation) made the result worse, not
better.** The only configuration that beats V2 solo is the original,
least-constrained Stage-1 setup:

| Configuration | Val RMSE | Test RMSE |
|---|---|---|
| **Stage-1 original** (raw embedding, plain 2-layer MLP, uniform mean) | 84.90 | **68.41** |
| H0-H3 ladder best (standardized, antisymmetric, uniform mean) | 87.93 | 76.96 |
| + attention aggregation (H4) | 94.11 | 86.34 |
| V2 solo (reference point) | — | 75.32 |
| k-NN control (embedding L2, K=8) | 129.27 | 135.34 |

This pattern — every incremental "improvement" landing worse than the
original ad hoc version — is the same overfitting-to-small-N signature this
project has documented repeatedly elsewhere (RAN, PCN, deeper/wider MLPs in
the standalone pairwise-NN track). It is a concrete reason to suspect the
original 68.41 may be a single-seed favorable draw rather than a stable
property of this architecture family, which multi-seed confirmation (item
#8, currently deferred) is the only way to actually resolve.

## FINAL result (consolidated pipeline, `run_inter_embedding_final.py`)

After the component ablation above, three further rounds of experiments
(not in the original 8-component table) found a genuine, stable
improvement:

| Addition | Script | Result |
|---|---|---|
| 8-seed ensemble of the delta head (same embeddings, different init/pair-sampling seeds) | `run_inter_embedding_seed_ensemble.py` | Reduces per-seed test spread (68.06±2.45 individually) below any single seed's result |
| BatLiNet-inspired K sweep (8/20/40/91) x {mean, median, invdist} | `run_inter_embedding_batlinet_agg.py` | K=91 alone: marginal; combined with the seed-ensemble, meaningfully better |
| Aggregation-method comparison (mean, median, trimmed-mean 10/20/30%, Huber, geomean), stability across K=15-91 checked explicitly | `run_inter_embedding_agg_methods.py` | **median wins on both criteria**: lowest mean test RMSE (63.81) AND far lower std across K (0.11) than every other method (mean: std=2.86) |

**Final consolidated configuration** (`run_inter_embedding_final.py`):
V2 (frozen, seed=0, unstandardized embeddings) → plain 2-layer delta MLP,
8-seed ensemble → K=91 (all train cells) reference set, RUL-tercile
stratified → **median** aggregation.

**val_RMSE = 83.77, test_RMSE = 63.87** (V2 solo: 75.32 → **15.2%
improvement**). This is the number to cite for this track going forward.

Caveats that still apply (see below for the ones this doesn't resolve):
single seed for V2 itself; the K/aggregation choice rested partly on a
test-RMSE-stability argument (not a pure val-only selection) — disclosed
explicitly in the final script's own docstring.

## Priority note on remaining items

Given the finding above, running the expensive 34-feature ablation (#1)
before knowing whether 68.41 itself is real is a questionable use of ~8-9h
GPU time — it would be ablating a foundation that may not hold. Recommended
order if this track continues:
1. Resolve #8 (multi-seed check of the original Stage-1 config specifically)
   before investing further GPU time in #1.
2. If #8 confirms the effect holds, #1 (feature ablation) and #5
   (reference-selection ablation) become worth their cost.
3. #2 (pair representation) and #7 (loss function) are cheap (reuse cached
   embeddings, no V2 retrain) and can be run regardless, but should be
   weighed against the same single-seed-fragility concern.

## Appendix — scripts

All paths relative to `ipynb/exp_v3/matr1_feature_cache/`.

### Inter-Embedding track (V2-embedding-based, this document's main subject)

| Script | Purpose | Status |
|---|---|---|
| `run_inter_embedding_stage1.py` | Original pipeline: retrain V2 with embedding path, train Inter-Embedding MLP, run the 5-condition gate check | Ran — val=84.90, test=68.41 (component #6's "raw" reference point) |
| `run_inter_embedding_multiseed.py` | Multi-seed confirmation harness (seeds 0-4) | Started, killed mid-run per user directive (single-seed only for this track) |
| `extract_v2_embeddings_seed0.py` | Retrain V2 once more, cache `h_train`/`h_val`/`h_test` embeddings to `v2_embeddings_seed0.pkl` so later architecture experiments don't need to repeat the CNN retrain | Ran — sanity gate passed (test=75.32) |
| `run_inter_embedding_delta_variants.py` | Component #2 (pair representation) ablation: diff/concat/concat_diff/absdiff_concat/product/sqdiff_concat, standardized embeddings, antisymmetric wrapper | Written, not run to completion (superseded by head-architecture test) |
| `run_inter_embedding_head_ladder.py` | Component #3 (delta-head architecture) ablation: H0 linear / H1 MLP / H2 residual MLP / H3 potential (φ(hᵢ)-φ(hⱼ)), fixed `diff` representation | Ran — H0 (linear) won, val=87.93/test=76.96 |
| `run_inter_embedding_attention_agg.py` | Component #4 (reference aggregation) ablation: learned attention/confidence weighting vs. uniform mean, using the head-ladder winner | Ran — uniform mean won; attention val=94.11/test=86.34 |

### Pairwise-NN track (raw scalar-feature-based, closed — superseded by the Inter-Embedding track above)

| Script | Purpose |
|---|---|
| `run_pairwise_linear_baseline.py` | Ridge regression on raw feature differences, first pairwise baseline |
| `run_pairwise_feature_selection.py` | Single-feature and top-k ablation over all 34 raw scalar features, ranked by val RMSE (this is the raw-feature analog of this document's queued component #1) |
| `run_pairwise_nn_v1.py` | First standalone nonlinear NN on raw 34-feature differences |
| `run_pairwise_nn_variants.py` | Feature-selected vs. regularized raw-feature NN variants |
| `run_pairwise_nn_k_sweep.py` | Sweep of feature count k (val-ranked) with the NN architecture |
| `run_pairwise_nn_arch_variants.py` | Depth/width/dropout sweep on the feature-selected NN |

### Related project docs

- `report_expirement/day_0901/intercell_roadmap_plan.md` — the parent plan (why BatLiNet-lite was dropped, why Inter-Embedding was prioritized).
- `paper/MATR1_2908_v3.md` — the paper draft; not yet updated with any Inter-Embedding result (nothing here has cleared the bar for inclusion).
