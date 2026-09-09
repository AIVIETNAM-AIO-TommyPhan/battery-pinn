# Plan: Inter-Cell Roadmap for MATR1 — staged, risk-ordered implementation

## Context

The user shared `matr1-roadmap-intercell.md` (in `Downloads/`, not yet part of the
repo) proposing an **inter-cell** research direction for MATR1: instead of
predicting RUL from one cell's data alone, exploit relationships between the
target cell and other (reference) cells — either from raw voltage-capacity
curves (BatLiNet-lite, §2 of the roadmap) or from an existing encoder's
embeddings (Inter-Embedding Model, §3), then optionally blend with the
existing intra-cell prediction (§4). This has never been implemented — no
script, notebook, or `report_expirement/` file for it exists (verified via
grep across `report_expirement/`); the closest project-native trace is a
one-line backlog note in `report_expirement/day_0826/plan.md` item 13
("BatLiNet-style relative-regression branch... explicitly deferred, not
started").

**Why BatLiNet-lite (roadmap §2) is dropped from this plan entirely, not just
reordered.** Per this project's own standing convention
(`[[feedback_batlinet_role_learning_not_target]]`): BatLiNet is a source of
*insight* to test as a hypothesis in a new model, never a blueprint to
re-implement directly. The roadmap's §2 ("BatLiNet-lite") is written as a
literal port of BatLiNet's dual-branch architecture (new intra-branch CNN
`f` + new inter-branch CNN `g` on raw voltage-capacity curves) — exactly the
"rebuild BatLiNet" pattern the project has already ruled out. Separately,
this session has hard, session-native evidence against building *any* new
relational/inter-cell encoder from scratch on this 91-cell training set:
**RAN** (day_0808-09) and **PCN** (day_0809-10, its RAN-derived successor)
both did exactly that and failed decisively
(`report_expirement/day_0810/report_total.md`: RAN 2-encoder 5-seed mean
124.20±20.38, PCN all-pool mean≈133.5, neither ever approached the README
bar of 90, both closed as negative results). BatLiNet-lite's dual CNN
branches are architecturally the same kind of bet. **The insight worth
extracting from BatLiNet is "compare cells pairwise / use relative
information between cells," not "use this specific dual-CNN architecture."**
This plan tests that insight the cheap way: the **Inter-Embedding Model**
(roadmap §3) reuses the already-validated V2 encoder (5-seed confirmed,
`82.85±11.43`, this session's own headline architecture) and only trains a
small MLP on frozen embeddings to learn pairwise RUL differences — this
already *is* "building from BatLiNet's insight," done as a new, appropriately
-scaled test rather than a re-implementation. If this hypothesis is
confirmed to add value, later refinement means iterating on *this* mechanism
(better pairing/reference selection, a richer Δ-model), not switching to a
brand-new from-scratch dual-CNN encoder. This plan sequences the roadmap as:
**cheap diagnostic → Inter-Embedding Model → multi-seed confirmation →
hybrid ensemble with V2** — roadmap §2 (BatLiNet-lite) is not part of this
plan at any stage.

**A concrete blocker found during exploration, not present in the roadmap.**
V2/V1's `forward()` only returns the final scalar prediction, never the
48-dim embedding (`ipynb/exp_v3/matr1_feature_cache/run_stage1_axis_aware_confirm.py:170-173`),
and none of this session's training scripts persist model weights to disk —
`best_state` (the best-val checkpoint) is held only in memory and explicitly
`del`eted at the end of each run
(`run_stage1_axis_aware_confirm.py:230,247,264`). **The already-trained V2/V1
models from this session cannot have their embeddings extracted — V2 (all 5
seeds) must be retrained** with a modified model class that exposes the
embedding, to get the inputs the Inter-Embedding Model needs. Training is
fully deterministic (fixed seeds, `cudnn.deterministic=True`) and the
existing scripts already print RMSE per checkpoint, so retraining seed 0 is
also a free correctness check: if it does not reproduce the already-reported
75.32 RMSE, something about the retrain diverged and must be fixed before
trusting any embedding pulled from it.

## Stage 0 — Zero-training diagnostic (do this before writing any model code)

Mirrors this project's own established convention (Tier 0.5's Step-1 ranking
diagnostic before any ranking-loss training,
`report_expirement/day_0825/Wiener_PINN_Project_Checklist.md` §4.5) of
checking for exploitable signal before investing in training.

- Use the 34 already-cached scalar features in
  `ipynb/exp_v3/matr1_feature_cache/feature_cache.pkl` (`train_base_scalar_raw`)
  and the pre-transformed labels (log+zscore, matching the roadmap's `z_i`
  space) for the **91 train cells only** (no val/test — matches the
  roadmap's own "don't use val/test to generate pairs" rule, §2.5/§3.3 of the
  spec, applied here to the diagnostic too).
- For all `Δz_ij = z_i - z_j` pairs (or a large random sample, e.g. 2000 of
  the ~4095), fit a simple linear model (ridge, small alpha) predicting
  `Δz_ij` from the corresponding scalar-feature differences
  (`Δfeature_ij` for the top-3 features, and separately for all 34) and
  report in-sample R².
- **Gate**: if R² for this cheap linear pairwise-difference model is very
  low (rule of thumb: below ~0.15, informed by how weak individual scalar
  features already are at explaining absolute RUL — §2.2 of the paper), this
  is an early warning that inter-cell *difference* information may not carry
  much more signal than intra-cell information already does at this sample
  size, and the Inter-Embedding Model (Stage 1) should be scoped as a
  screen-only, single-seed experiment rather than a larger investment. If R²
  is meaningfully higher, proceed to Stage 1 with more confidence.
- Cost: minutes, no GPU, no new training. This is the single cheapest test
  of the roadmap's core premise and should run first regardless of which
  later stage the user wants to prioritize.

## Stage 1 — Inter-Embedding Model (roadmap §3), seed 0 screen only

1. Add an embedding-exposing forward path to the V2 model class
   (`SmallCNNScalarBranchAxisAware` in `run_stage1_axis_aware_confirm.py`) —
   e.g. `forward(x, scalar, return_embedding=False)` returning
   `(out, h)` when requested, `h = concat[signal_embedding, scalar_embedding]`
   (48-dim), with no other change to the architecture or training loop.
2. Retrain V2 at seed 0 with this modified class, same protocol as the
   confirmed run (patience=350, per `report_expirement/day_0901`'s pinned
   protocol — not patience=200, to match the already-reported 75.32 number
   exactly). **Sanity gate**: confirm test RMSE reproduces 75.32 before
   proceeding — if it doesn't, stop and debug the embedding-path change
   before building anything on top of it.
3. Extract frozen embeddings `h_i` for all 91 train + 41 val + 42 test cells
   from this checkpoint (kept in memory this run, or saved to disk this
   time — recommend saving `state_dict` to disk now, unlike prior scripts,
   specifically so this doesn't have to be redone).
4. Build the pairwise training set exactly per spec
   (`pairwise-ranknet.md` §4 / `v2-soft-contrast.md` §7 pattern, reused
   here): train cells only, per-step sampling of ~32 anchors × 3-5 targets,
   margin-based labels not needed here since this is regression on `Δz_ij`
   directly (roadmap §3.3), not ranking — **pre-declare** the anchor/target
   sampling count and any margin *before* looking at any result.
5. Train a small MLP (`Δh_ij = h_i - h_j → Δz_ij`, 2 layers, similar size to
   V2's existing scalar branch) — pre-declared hyperparameters (hidden dim,
   lr, epochs, patience), not tuned by looking at test.
6. Inference (roadmap §3.5): for each test/val cell `i`, pick `K` reference
   cells from **train only** (pre-declare `K`, e.g. 8 or 16, and the
   reference-selection rule — nearest by feature distance, or random —
   before seeing results), predict `Δz_{i,r}` for each, recover
   `z_i^(r) = z_r + Δz_{i,r}`, average.
7. **Required control**: a trivial k-NN baseline using the *same* K
   reference cells and the *same* aggregation (plain average of reference
   RULs, no learned Δ-model at all) — the Inter-Embedding Model must beat
   this control, not just beat "nothing," or it isn't adding value beyond
   what a nearest-neighbor heuristic already gives for free.
8. **Gate to Stage 2**: report raw test RMSE alongside the k-NN control
   and V2's own 75.32, plus error correlation with V2 (for later ensemble
   potential, §4 of the roadmap). Single seed only at this stage — do not
   claim a result, only a screen outcome, consistent with this project's
   multi-seed discipline.

## Stage 2 — Multi-seed confirmation (only if Stage 1 clears its gate)

Retrain V2 (with the embedding-exposing forward) for seeds 1-4, repeat the
Inter-Embedding Model training and inference per seed, and report 5-seed
mean±std for: the Inter-Embedding Model alone, the k-NN control, and an NNLS
ensemble of {V2, Inter-Embedding Model} per seed (reusing this session's
now-established NNLS convention from paper §5, including its known
calibration-selection risk — do not repeat the isotonic-overfits-val mistake
from §5.0 without at minimum checking val vs. test agreement first).

## Stage 3 — Hybrid ensemble (roadmap §4, adapted)

Combine the Inter-Embedding Model (Stage 1/2) with V2 via NNLS, fit per seed
on validation only — the same protocol already established and validated in
paper §5.1, including its known caveats (candidate-selection should be done
on validation where possible, given §5.1's own finding that seed-0
test-informed selection is a real, disclosed risk). If Stage 1/2 show real,
confirmed signal and there is appetite to push further, the next step is
refining the Inter-Embedding mechanism itself (better reference-cell
selection, a richer Δ-model, more references K) — not building a new
from-scratch encoder.

## What this plan does NOT do

- **Does not build BatLiNet-lite (roadmap §2) at any stage.** BatLiNet is
  treated strictly as the source of the "compare cells pairwise" insight,
  already carried into Stage 1's design — not as an architecture to
  re-implement. If a future need arises to revisit this decision, that is a
  new, separate discussion, not a later stage of this plan.
- Does not touch or depend on `PairwiseRankNet` (the separate, still-unbuilt
  roadmap from `pairwise-ranknet.md`) — the Inter-Embedding Model works
  directly on V2's embeddings, per the roadmap's own "V2 or PairwiseRankNet"
  wording choosing V2 since it already exists and is validated.
- Does not modify `paper/MATR1_2908_v3.md` — this is a new, separate research
  thread; the paper is not touched until/unless a stage here produces a
  result worth reporting.
- Does not start any multi-seed run (Stage 2) before Stage 0/1's gates are
  checked — this is the core point of the staged, risk-ordered approach.

## Verification

- Stage 0: rerun the ridge-diff regression with a different random pair
  sample to confirm the R² isn't sensitive to sampling noise before treating
  it as informative.
- Stage 1 step 2's sanity gate (retrained V2 seed 0 reproduces 75.32) is the
  hard go/no-go check before any embedding is trusted.
- Stage 1 step 7's k-NN control is the correctness check that the learned
  Δ-model is adding value, not just re-deriving what nearest-neighbor
  averaging already gives.
