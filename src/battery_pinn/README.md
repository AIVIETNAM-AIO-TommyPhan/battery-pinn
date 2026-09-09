# src/battery_pinn — scaffold only, not yet populated

This package is a **placeholder for a future refactor**, not a working
library yet. Every script in `scripts/` is currently self-contained: the
CNN backbone (`SmallCNNScalarBranchAxisAware` / its Tier-3 variant), the
Tier-1/Tier-3 loss terms (SOH auxiliary head, monotonicity, Wiener
drift/diffusion), the feature-cleaning helpers (`smoothing`,
`_remove_glitches`, `filter_cycles`, `clean_feature`), and the ensemble
calibration logic (`affine_fit` + NNLS) are copy-pasted with small,
**deliberate, dataset-specific differences** into each `run_*.py` script
(e.g. CRUSH's asymmetric `UNDER_PENALTY` term has no MATR1/HUST
equivalent; λ_SOH differs per dataset by explicit, disclosed screening —
see `reports/day_0906/report.md`).

## Why this isn't extracted yet

Consolidating these into one shared implementation is real, valuable
future work, but risky to do casually: a subtle difference lost in the
merge (a default hyperparameter, an off-by-one in which cycle index
`clean_feature` treats as the edge, a variant of the Wiener loss's
`mono_tolerance`) would silently change results that are already reported
with specific `.pkl`-traceable numbers in `reports/`. Per this project's
own standing rule, any such extraction must re-verify that each dataset's
headline number reproduces *exactly* against the refactored code before
being trusted (a "sanity gate," in the reports' terminology) — that
verification pass has not been done.

## Intended layout, when this is done

- `models/` — `SmallCNNScalarBranchAxisAware` and its Tier-3 variant,
  `CycleAttentionPooling`, `LearnableCyclePositionalEncoding`
- `losses/` — SOH auxiliary (Smooth-L1), monotonicity, Wiener
  drift/diffusion, CRUSH's asymmetric RUL term
- `features/` — `smoothing`, `_remove_glitches`, `filter_cycles`,
  `clean_feature`
- `calibration/` — `affine_fit`, the NNLS ensemble step
- `data/` — `DataBundle`-loading and feature-cache-schema helpers
- `utils/` — misc (seeding, metric computation)

Until that refactor happens (and is re-verified per-dataset), treat
`scripts/<dataset>/` as the source of truth, not this package.
