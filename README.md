# Battery PINN (Tier-3): SOH-Auxiliary, Monotonicity & Wiener-Diffusion RUL Prognostics

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code and results for a physics-informed RUL-prognostics research track built
on top of a private fork of [microsoft/BatteryML](https://github.com/microsoft/BatteryML),
covering six datasets: **MATR1, MATR2, HUST, CRUSH, CRUH, SNL**. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the full V1/V2/Inter-Embedding
diagram with SOH heads, mapped block-by-block to `src/battery_pinn/`. Does **not**
contain the BatteryML library itself (`batteryml/`, `BatLiNet/`) — see
"Dependencies" below.

Every number in this README is read directly from `reports/day_0906/report.md`
(or an earlier day's report where noted), each traceable to a specific
`.pkl`/log file per that report's own discipline — nothing here is
restated from memory. Single-seed numbers are labeled as screens, not results.

---

## Key methodological components

1. **Tier-3 physics-informed objective** — the project's primary
   physics-informed claim, in the sense of Raissi et al. (2019): a governing
   equation as a soft training constraint, not just a physically-named
   auxiliary target. Three terms added to the RUL loss:
   - **SOH auxiliary head** (Smooth-L1 on a 100-cycle SOH trajectory)
   - **Monotonicity constraint** — penalizes any predicted SOH increase
     beyond a small tolerance
   - **Wiener-process diffusion loss** — treats the capacity-fade increment
     `dD(t) = μ dt + σ dW(t)` as a Gaussian-increment SDE and fits `(μ, σ)`
     via a Gaussian negative log-likelihood on the drift/diffusion residual
2. **Axis-aware CNN backbone (V1/V2)** — a 3-layer Conv2d backbone over the
   ΔQ(V) signal with attention pooling over the cycle axis, plus a small
   scalar branch on 3 physics-derived voltage–capacity features. V1 adds a
   learnable cycle positional encoding; V2 omits it. This backbone alone
   already beats the README benchmark on MATR1 and HUST.
3. **Inter-Cell Embedding** — a small MLP that learns pairwise RUL
   *differences* from frozen backbone embeddings, then infers a target
   cell's RUL by median-aggregating over a stratified reference set of
   training cells. Built on a **Tier-1** (SOH-auxiliary-only) embedding,
   not Tier-3 — empirically the Tier-3 embedding's temporal constraints
   compress the relative-distance structure this module needs.
4. **Ensemble calibration, with an explicit caveat** — components are
   affine-calibrated on val, then combined by either NNLS-fit weights or a
   plain mean (`src/battery_pinn/calibration/ensemble.py`). NNLS is not a
   safe default: on CRUSH's 8-seed redesigned-val run, simple mean beat
   NNLS on both mean RMSE (339.8 vs. 341.0) and stability (std 13.9 vs.
   21.8) — small val sets let NNLS's per-component free weight overfit.

---

## Results

| Dataset | README target | Ours | RMSE | MAE | MAPE (%) | R² | Beats README? |
|---|---:|---|---:|---:|---:|---:|---|
| **MATR1** | 90 (PCR) | Tier-3 backbone + Inter(Tier-1), 5-seed, official | **71.1 ± 5.3** | 60.9 ± 4.8 | 9.3 ± 0.8 | 96.3 ± 0.5% | ✅ ~21% better |
| **HUST** | 322 | Tier-1 ensemble, 8-seed | **289.8 ± 12.4** | 234.0 ± 13.1 | 12.8 ± 0.7 | 52.1 ± 4.1% | ✅ ~10% better (via Tier-1; Tier-3 is worse here, 316.9±16.8) |
| **CRUSH** | 330 (local repro 355) | Tier-3, redesigned val/train, simple-mean, 8-seed | **339.8 ± 13.9** | 204.3 ± 12.4 | 50.2 ± 7.1 | 59.3 ± 3.4% | 🟡 beats local repro by ~4.5%, still ~3% short of README |
| MATR2 | 149 | No-SOH ensemble, **single-seed screen** | 210.7 | — | — | — | ❌ no |
| CRUH | 60 | No-SOH ensemble, **single-seed screen** | 100.7 | 72.5 | 12.8 | 68.0% | ❌ no |
| SNL | 200 | Best of 4 val-construction screens, **single-seed** | 370.6 | 250.6 | 56.8 | 31.0% | ❌ no |

MATR2/CRUH/SNL are explicitly flagged single-seed screens, not multi-seed
claims, per this project's own reproducibility discipline — see
`reports/day_0906/report.md`'s Error Analysis (§4) for why SNL and CRUH, in
particular, are dominated by one structurally-different low-n subgroup
(SNL's LFP chemistry; CRUH's CALCE lab) that neither a better val split nor
train augmentation has fixed so far.

---

## Full structure

```text
battery-pinn-release/
├── README.md, LICENSE, requirements.txt
│
├── configs/                       placeholder — no hyperparameter YAML/JSON
│                                   files yet; every script currently
│                                   hardcodes its own constants at the top
│                                   (pre-declared, not tuned on test — see
│                                   each script's docstring)
│
├── data/
│   ├── raw/                       placeholder — raw cell data lives in the
│   │                               BatteryML fork's batteryml/data/processed/,
│   │                               not copied here (see Dependencies)
│   └── processed/                 placeholder — feature-cache .pkl files
│                                   (large, regenerable) are not copied here
│
├── paper/
│   ├── manuscript.md              draft manuscript (formerly paper_0904.md)
│   ├── figures/                   placeholder — no extracted figure files yet
│   └── tables/                    placeholder — result tables currently
│                                   live inline in reports/*/report.md, not
│                                   extracted to standalone files
│
├── reports/                       one folder per research session
│   ├── day_0825/  (4 files: plan.md, report.md, PINN4SOH_MATR_RUL_Adaptation_Spec.md,
│   │               Wiener_PINN_Project_Checklist.md)
│   ├── day_0826/  (plan.md, report.md)
│   ├── day_0827/  (plan.md, report.md)
│   ├── day_0831/  (plan.md, report.md)
│   ├── day_0901/  (6 files: plan.md, report.md, Project_Architecture_Ladder_Spec.md,
│   │               Thesis_Aligned_Architecture_Ladder.md,
│   │               inter_embedding_component_ablation_plan.md,
│   │               intercell_roadmap_plan.md)
│   └── day_0906/  (plan.md, report.md — the most current; includes the
│                    CRUSH val/train redesign, the cross-dataset "Dataset &
│                    Split Composition Analysis" section, and the CALCE/LFP
│                    per-source error-analysis finding)
│
├── src/battery_pinn/               reusable package — SCAFFOLD, mostly empty
│   ├── README.md                   explains what's extracted vs. not, and why
│   ├── features/
│   │   └── cleaning.py             smoothing, remove_glitches, filter_cycles,
│   │                               clean_feature (shared, byte-identical
│   │                               across every dataset's script)
│   ├── models/
│   │   ├── backbone.py             LearnableCyclePositionalEncoding,
│   │   │                           CycleAttentionPooling,
│   │   │                           SmallCNNScalarBranchAxisAwareTier3
│   │   └── inter_embedding.py      InterEmbeddingModel, build_reference_set,
│   │                               run_inter_embedding
│   ├── losses/
│   │   └── tier3.py                asymmetric_rul_loss, soh_auxiliary_loss,
│   │                               monotonicity_loss, wiener_loss,
│   │                               tier3_total_loss
│   ├── calibration/
│   │   └── ensemble.py             affine_fit + ENSEMBLE_METHODS registry
│   │                               (affine_nnls / affine_mean / simple_mean)
│   │                               + combine() dispatcher
│   ├── data/                       empty — DataBundle/cache-schema helpers
│   │                               not yet extracted
│   └── utils/
│       └── metrics.py              rmse, mae, mape, r2, full_metrics
│   Only ONE pipeline (CRUSH's Tier-3 ensemble, scripts/crush/ensemble/
│   run_crush_v1v2inter_tier3_ensemble.py) has been refactored to import
│   from this package and re-verified to reproduce its original seed-0
│   numbers exactly. Every other dataset's scripts remain self-contained
│   (each duplicates its own copy of the model/loss/cleaning code) — see
│   src/battery_pinn/README.md for why extracting the rest is deferred.
│
└── scripts/                        experiment entry points, one folder per
                                    dataset, each split into subfolders by
                                    purpose (data_build → validation →
                                    ensemble, in the order a fresh run would
                                    use them)
    │
    ├── matr1/                     44 scripts total
    │   ├── pairwise/       (6)    BatLiNet-insight pairwise-difference
    │   │                          experiments (linear baseline, NN variants,
    │   │                          feature selection, k-sweep) — the
    │   │                          precursor line of work to Inter-Embedding
    │   ├── embedding/      (15)   Inter-Cell Embedding: aggregation-method
    │   │                          sweeps, k-reference tuning, multiseed
    │   │                          confirmations, the final adopted version
    │   ├── stage_sweep/    (11)   architecture-ladder screens (stage0
    │   │                          sanity → stage3 conv-transformer), V2
    │   │                          Tier0-3 PINN screens, single-feature and
    │   │                          entropy-regularization ablations
    │   ├── validation/     (1)    diag_v2only_nostop.py — no-early-stop
    │   │                          val/test correlation diagnostic
    │   ├── ensemble/       (5)    production V1+V2+Inter-Embedding ensembles
    │   │                          (Tier-1 SOH-only, Tier-3, cross-tier
    │   │                          variants) — MATR1's headline number
    │   │                          (71.1±5.3) comes from here
    │   └── utils/          (1)    _assemble_notebook.py (scratch-cell →
    │                              notebook assembly helper, project convention)
    │
    ├── matr2/                     5 scripts — split out as its own dataset
    │   │                          (train_base pools MATR1+MATR34-b4+HUST;
    │   │                          test is pure MATR34-b4, the official
    │   │                          "unseen protocol" batch)
    │   ├── data_build/     (2)    build_matr2_feature_cache.py,
    │   │                          build_matr2_variant_caches.py
    │   ├── validation/     (1)    diag_matr2_v2only_nostop.py
    │   └── ensemble/       (2)    run_matr2_smallcnn_screen.py,
    │                              run_matr2_v1v2inter_ensemble.py
    │                              (headline: 210.7, single-seed screen only)
    │
    ├── hust/                      8 scripts
    │   ├── data_build/     (2)    build_hust_feature_cache.py (official
    │   │                          55/22 split + 20-cell MATR-donor
    │   │                          augmentation), build_hust_soh_cache.py
    │   ├── validation/     (1)    diag_v2only_nostop.py
    │   └── ensemble/       (5)    Tier-1 (headline: 289.8±12.4, 8-seed),
    │                              Tier-3, cross-tier, k-fold, smallcnn screen
    │
    ├── crush/                     25 scripts — the most iterated-on dataset
    │   │                          this session (leakage found+fixed, val
    │   │                          redesigned, train reinforced)
    │   ├── data_build/     (6)    feature-cache builders (2 variants),
    │   │                          fromraw rebuild, SOH cache, CALCE/SNL
    │   │                          donor searches
    │   ├── validation/     (10)   4 cluster-val variants (older), dedup
    │   │                          (leakage fix), 3 val-redesign attempts
    │   │                          (quantile/similarity/random — only random
    │   │                          worked), train_reinforce, the no-early-stop
    │   │                          correlation diagnostic
    │   └── ensemble/       (9)    No-SOH/Tier-1/Tier-3/asymloss production
    │                              scripts + eval_crush_reinforced_altens.py
    │                              (post-hoc NNLS-vs-mean comparison).
    │                              **run_crush_v1v2inter_tier3_ensemble.py
    │                              is the one script in this repo refactored
    │                              to import src/battery_pinn** (headline:
    │                              339.8±13.9, 8-seed, simple-mean ensemble)
    │
    ├── cruh/                      9 scripts
    │   ├── data_build/     (3)    feature cache, SNL donor-pool cache
    │   │                          (shared with CRUSH's donor search),
    │   │                          CALCE donor search
    │   ├── validation/     (4)    cluster-val, CALCE-augmented cache builds
    │   │                          (single + full 8-donor versions), SNL sweep
    │   └── ensemble/       (2)    smallcnn screen, V1V2Inter ensemble
    │                              (headline: 100.7, single-seed screen;
    │                              augmented-cache result: 102.2, no better)
    │
    └── snl/                       7 scripts — SNL as a standalone modeling
        │                          target (distinct from its OTHER role as
        │                          a cross-dataset donor pool for CRUH/CRUSH)
        ├── data_build/     (1)    build_snl_standalone_feature_cache.py
        ├── validation/     (2)    build_snl_cluster_val.py (earlier),
        │                          build_snl_split.py (this session — also
        │                          fixed a NaN-feature bug, see its docstring)
        └── ensemble/       (4)    smallcnn screen, V1 k-fold, official-XGB
                                   and sklearn baselines (best: 370.6,
                                   single-seed; 4 total screens, none beat
                                   README=200 — see reports/day_0906/report.md
                                   Error Analysis §4)
```

---

## Dependencies

These scripts are **not standalone** — every one does
`sys.path.insert(0, REPO)` and imports `batteryml.*` / `BatLiNet` from a
local checkout of the private BatteryML fork, and loads a dataset's
`feature_cache_*.pkl` from a path inside that fork's `ipynb/exp_v3/<dataset>/`
tree. To actually run anything here, you need:

1. A local checkout of the BatteryML fork (not public; contact the author),
   with `batteryml/data/processed/<SOURCE>/*.pkl` populated per dataset.
2. The relevant `feature_cache_*.pkl` file(s) already built (or rebuild them
   with the matching `data_build/` script).
3. Python packages in `requirements.txt`.
4. Edit the `REPO = r"..."` path at the top of whichever script you run to
   point at your local fork checkout.

## Data sources

**MATR1, MATR2, HUST, CRUSH, CRUH, SNL are not raw datasets themselves** —
CRUSH and CRUH are this project's/BatteryML's own composite splits built
from several of the raw lab sources below (CRUSH = CALCE+RWTH+UL_PUR+SNL+HNEI;
CRUH = CALCE+RWTH+UL_PUR+HNEI, per `configs/baselines/*/crush.yaml` /
`cruh.yaml` in the BatteryML fork); MATR1/MATR2/HUST/SNL draw on the raw
sources named per row below. This table is copied from the BatteryML fork's
own `dataprepare.md` (not fabricated) — go there for the full download/
preprocessing walkthrough (`python scripts/preprocess.py` after placing raw
files under `data/raw/<SOURCE>/`).

| Raw source | Used by | Where to get it | Citation |
|---|---|---|---|
| MATR (4 batches) | MATR1, MATR2 | [data.matr.io](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204) (3 batches) + [4th batch](https://data.matr.io/1/projects/5d80e633f405260001c0b60a/batches/5dcef1fe110002c7215b2c94) | Severson et al., *Data-driven prediction of battery cycle life before capacity degradation*, Nature Energy 4, 383–391 (2019); 4th batch from Attia et al., *Closed-loop optimization of fast-charging protocols*, Nature 578, 397–402 (2020) |
| HUST | HUST | [Mendeley Data](https://data.mendeley.com/datasets/nsc7hnsg4s/2) | Ma et al., *Real-time personalized health status prediction of lithium-ion batteries using deep transfer learning*, Energy & Environmental Science 15(10), 4083–4094 (2022) |
| CALCE | CRUSH, CRUH | [calce.umd.edu/battery-data](https://calce.umd.edu/battery-data#Citations) (CS2/CX2 series) | He et al., *Prognostics of lithium-ion batteries based on Dempster–Shafer theory and the Bayesian Monte Carlo method*, 196(23), 10314–10321 (2011) |
| RWTH | CRUSH, CRUH | [RWTH Aachen publications](https://publications.rwth-aachen.de/record/818642) | Li et al., *One-shot battery degradation trajectory prediction with deep learning*, Journal of Power Sources, 230024 (2021) |
| SNL | SNL (standalone), CRUSH, CRUH (donor) | [batteryarchive.org](https://www.batteryarchive.org/) — **access request required, no longer a direct download** (see caveat below) | Preger et al., *Degradation of commercial lithium-ion cells as a function of chemistry and cycling conditions*, J. Electrochem. Soc. 167, 120532 (2020) |
| UL-PUR | CRUSH, CRUH | batteryarchive.org — same access-request caveat | Juarez-Robles et al., *Degradation-safety analytics in lithium-ion cells: Part I*, J. Electrochem. Soc. 167, 160510 (2020) |
| HNEI | CRUSH, CRUH | batteryarchive.org — same access-request caveat | Devie et al., *Intrinsic variability in the degradation of a batch of commercial 18650 lithium-ion cells*, Energies 11, 1031 (2018) |

**Caveat on SNL/UL-PUR/HNEI (confirmed via both BatteryML's `dataprepare.md`
and BatLiNet's own `scripts/download.py`, whose `SNL_LINKS`/`UL_PUR_LINKS`/
`HNEI_LINKS` are commented out of its default download list):**
batteryarchive.org stopped serving these three datasets via direct URL —
you must request access from them directly. The specific per-cell CSV
filenames these scripts expect (e.g.
`SNL_18650_LFP_25C_0-100_0.5-1C_a_cycle_data.csv`) are still documented in
`dataprepare.md`'s file-tree listing and in BatLiNet's `download.py`
(dead links, but the filenames/naming convention are accurate), useful for
knowing what to ask for.

### Where the val set comes from (it is not part of the raw download)

The table above gets you `batteryml/data/processed/<SOURCE>/*.pkl` and,
after each dataset's `data_build/` script, a `feature_cache_*.pkl` with an
official **train/test** split (from BatteryML's own splitter classes, e.g.
`HUSTTrainTestSplitter`, `SNLTrainTestSplitter`, `MATRPrimaryTestTrainTestSplitter`).
**None of these official splitters define a val set** — every dataset's val
here is carved out of the official train pool by this project's own
`validation/` scripts, and the method differs per dataset (see
`reports/day_0906/report.md`'s "Dataset & Split Composition Analysis" for
the full per-split RUL-distribution and composition tables):

| Dataset | Val size | Construction | Script |
|---|---:|---|---|
| MATR1 | 41 | size matches the official train-id list exactly; exact cell-level provenance of the larger train_base (91) is an open gap (no ids stored in `feature_cache.pkl`) | *(not in `scripts/matr1/`, cache pre-dates ID-tracking)* |
| MATR2 | 40 | nearest-neighbor RUL-match to test, drawn from the same MATR1+MATR34-b4+HUST pool as train_base — disclosed test-informed choice | `scripts/matr2/data_build/build_matr2_feature_cache.py` |
| HUST | 12 (of 55 official-train) | RUL-tercile-stratified, pre-declared before touching test | `scripts/hust/data_build/build_hust_feature_cache.py` |
| CRUSH | 20 (of 70 train+val pool) | plain random draw — found THIS session after ruling out quantile-stratified and test-centroid-similarity val redesigns; fixed a val/test RUL-range mismatch (r=−0.370 → +0.785, see report.md Error Analysis §2) | `scripts/crush/validation/build_crush_val_random.py` |
| CRUH | 17 | cluster-based (K-means on scalar features + RUL); has a **known, unfixed** distribution mismatch (val mean RUL sits well below both train and test) — an open item, not yet given the CRUSH treatment | `scripts/cruh/validation/build_cruh_cluster_val.py` |
| SNL | 5 (of 16 official-train, labeled subset) | RUL-tercile-stratified; this session's version also fixed a NaN-feature bug in the donor-pool cache it builds from | `scripts/snl/validation/build_snl_split.py` |

If you rebuild a feature cache from scratch, running the `data_build/`
script alone reproduces the *official* train/test split; you additionally
need the matching `validation/` script (where one exists) to get the exact
val this project's reported numbers used.

## Quick start

```bash
git clone <this-repo>
cd battery-pinn-release
pip install -r requirements.txt
# edit REPO = r"..." at the top of the script you want to run, then e.g.:
python scripts/crush/ensemble/run_crush_v1v2inter_tier3_ensemble.py \
    feature_cache_crush_hust10_calce15_fullval_withsoh_randval_reinforced 0 1.4
```

There is no single "run everything" entry point (unlike BatLiNet's
`run_all_configs.sh`) — each dataset's `ensemble/` script is invoked
individually, per the CLI documented in its own docstring.

## Not included

- `batteryml/`, `BatLiNet/` library source (the fork's original code)
- Raw processed cell data (`batteryml/data/processed/`)
- Feature-cache `.pkl` files (large, regenerable from `data_build/` scripts + raw data)
- Per-run result `.pkl` / `.log` files (numbers are in each `reports/*/report.md`'s tables instead)
- Earlier, non-PINN experiment days (`day_0708`–`day_0823`) — architecture
  search / baseline work that predates the Tier-1/Tier-3 objective

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed
  neural networks: A deep learning framework for solving forward and
  inverse problems involving nonlinear partial differential equations.*
  Journal of Computational Physics, 378, 686–707. — the Tier-1/Tier-3
  distinction in this project's manuscript (`paper/manuscript.md` §2.2)
  follows this paper's definition of what makes an objective
  "physics-informed" (a governing equation as a soft constraint) rather
  than just physically-named supervision.
- Severson, K. A., Attia, P. M., Jin, N., et al. (2019). *Data-driven
  prediction of battery cycle life before capacity degradation.* Nature
  Energy, 4, 383–391. — source of the MATR1/MATR2 dataset splits used here.
- [microsoft/BatteryML](https://github.com/microsoft/BatteryML) — the
  benchmark codebase this project forks (feature extractors, official
  train/test splitters, README RMSE targets cited above).
- BatLiNet ("Accurate battery lifetime prediction across diverse aging
  conditions with deep learning") — source of the pairwise/relative-RUL
  insight this project's Inter-Cell Embedding module builds on (see
  `scripts/matr1/pairwise/` and `paper/manuscript.md` §6.2 for how the
  insight was adapted, not the architecture ported directly).

`paper/manuscript.md` does not yet have a formatted bibliography — treat
the list above as a pointer to the four sources actually used this
session, not a complete literature review.

## License

MIT — see [LICENSE](LICENSE). Note this project depends on (but does not
redistribute) microsoft/BatteryML, itself MIT-licensed.

## Status

Local git repo (`git init` done, `master` branch), not pushed to any
remote. Staged here for review before deciding where (if anywhere) to push it.
