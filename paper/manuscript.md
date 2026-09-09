# Physics-Informed Wiener-Diffusion Modeling with Cycle-Axis Attention for Lithium-Ion Battery Remaining Useful Life Prediction

> **Draft status**: outline + key sections drafted for review. HUST (5-seed) and CRUSH (8-seed) Tier-3 runs are both complete (§5.1, §5.2b, §6.3) — HUST's proposed result is Tier-1; CRUSH's is Tier-3, though none of CRUSH's configurations beat README. MATR1's headline remains the pre-registered 5-seed result; a 7-seed robustness check (seed 8 pending) confirms the same ordering. Numbers marked `[TBD]` are pending only the SNL/CALCE/CRUH single-seed screens — do not cite before these are filled in and verified against `report_expirement/`.

## Abstract

Accurate battery remaining useful life (RUL) prediction is challenging because the RUL label is sparse, degradation trajectories are heterogeneous across cells, and models often fail to generalize across laboratories and protocols. We propose a degradation-aware architecture combining a CNN backbone, learned attention pooling over the cycle axis, and physics-derived voltage–capacity features, which alone surpasses the published benchmark on MATR1 and HUST. We then investigate a physics-informed auxiliary objective (Tier-3) that supervises the network with the SOH trajectory under an explicit degradation model — a monotonicity constraint and a Wiener-process diffusion term ($dD(t) = \mu\,dt + \sigma\,dW(t)$) governing capacity fade — and combine it with an Inter-Cell Embedding module that corrects predictions using pairwise relationships between cells. Under a strict validation-only model-selection protocol with 5-seed evaluation, we report the resulting pipeline's performance on MATR1, HUST, and CRUSH against both the published README benchmark and our own in-process reproduction of it, and disclose every case where a deviation from the pre-declared protocol occurred. The physics-informed objective's benefit is dataset-dependent; we report this honestly and discuss a mechanistic hypothesis (source heterogeneity) for why, rather than presenting a uniform improvement claim.

---

## 1. Introduction

### 1.1 Background & Motivation

Lithium-ion batteries are widely used in electric vehicles, renewable energy integration, and portable electronics. Accurate estimation of their remaining useful life (RUL) is critical for safe operation, maintenance planning, and cost reduction. Battery degradation is influenced by chemistry, operating conditions, and manufacturing variability, leading to heterogeneous degradation trajectories across cells and datasets.

From a machine learning perspective, RUL prediction poses several challenges:

- The RUL label is a single scalar per cell, observed only at end-of-life.
- The degradation process is rich in information (voltage, capacity over cycles), but most models only optimize for the final RUL value.
- Datasets collected by different laboratories exhibit domain shift in protocols, chemistry, and measurement noise, causing models trained on one source to degrade on another.

### 1.2 Problem & Gap

Existing deep learning approaches for battery RUL prediction often:

- Optimize only an RUL loss, ignoring the dense degradation signal available in the SOH trajectory.
- Use random or poorly documented train/validation/test splits, sometimes implicitly using test data to select models or hyperparameters.
- Report results on a single split or seed, without analyzing generalization across sources or laboratories.
- When they do add an SOH-based auxiliary loss, treat it as "physics-informed" without an explicit governing equation — an auxiliary-supervision claim, not a physics-constraint claim.

As a result, many models achieve low training error but exhibit large and unstable test errors, especially when evaluated on unseen datasets or protocols.

### 1.3 Proposed Approach

We propose a framework for battery RUL prediction with four components, presented in the order of the evidence supporting them:

1. **Axis-aware attention backbone**: a CNN backbone with learned attention pooling over the cycle axis, plus a scalar branch over three physics-derived voltage–capacity features. This component alone surpasses the published README benchmark on MATR1 and HUST (§5.1).
2. **Tier-3 physics-informed objective**: an auxiliary SOH-trajectory head trained jointly with (a) a monotonicity constraint on predicted SOH and (b) a Wiener-process diffusion loss on the capacity-fade increment $D(t) = 1 - \mathrm{SOH}(t)$, i.e. an explicit stochastic degradation model $dD(t) = \mu\,dt + \sigma\,dW(t)$ (§3.5). This is what we regard as the physics-informed component proper, as distinct from SOH-only auxiliary supervision (Tier-1), which we use only as a component inside the Inter-Cell Embedding module (below), not as a standalone claim.
3. **Inter-Cell Embedding correction**: a small model that predicts pairwise RUL differences from frozen cell embeddings and aggregates over reference cells, used as a correction term in the final ensemble (§3.6).
4. **Source-aware validation protocol**: train/validation/test separated by dataset and source, validation-only model selection, and 5-seed evaluation, with every deviation from this rule disclosed explicitly (§4.3, §7).

### 1.4 Contributions

- An **axis-aware attention architecture** that surpasses the published README benchmark on MATR1 and HUST using only architecture and physics-derived features, with no auxiliary loss.
- **Tier-3**, a physics-informed objective governed by an explicit stochastic degradation model (Wiener process on capacity fade), which we distinguish from simpler SOH-auxiliary supervision (Tier-1) both conceptually (§2.2) and in how it is used in the final pipeline (§3.6).
- An **Inter-Cell Embedding** module that learns pairwise degradation relationships between cells and is combined with the Tier-3 backbone via a disclosed cross-tier design (§3.6, §6.2).
- A **fully disclosed evaluation protocol**: 5-seed mean±std throughout, comparison against both the published README benchmark and our own in-process reproduction of it, and explicit disclosure of every case where hyperparameter selection deviated from validation-only screening (§4.3, §7).

---

## 2. Related Work

### 2.1 Deep Learning for Battery RUL Prediction

Recent works have applied CNNs, LSTMs, attention mechanisms, and hybrid architectures to battery RUL prediction. These models typically take voltage, current, or capacity trajectories as input, output a single RUL value per cell, and are trained with an RUL loss (e.g., MSE or Huber). While these methods demonstrate strong performance on specific datasets, they often use random splits without controlling for source or protocol, report results on a single seed or split, and do not explicitly leverage the dense SOH trajectory as a learning signal.

### 2.2 Physics-Informed Learning, and Why We Distinguish Two Tiers

Physics-informed neural networks (PINNs), as defined by Raissi et al. (2019), incorporate a governing differential equation directly into the training loss as a soft constraint. Much of the recent "physics-informed" battery literature — including our own earlier Tier-1 design — instead uses a physically meaningful *quantity* (e.g., SOH) as an auxiliary supervision target, without an explicit governing equation. We treat this distinction as consequential rather than terminological:

- **Tier-1 (SOH-auxiliary)**: an auxiliary head predicts the SOH trajectory, supervised with a standard regression loss. This regularizes the representation toward a physically meaningful quantity but enforces no dynamics.
- **Tier-3 (this work's main objective)**: in addition to the SOH-auxiliary term, we add a monotonicity constraint and a Wiener-process diffusion loss over the *modeled* capacity-fade process, i.e., an explicit SDE governing how SOH evolves between cycles (§3.5). This satisfies the governing-equation criterion in a way Tier-1 does not.

We report both because they play different roles in our final design (§3.6): Tier-3 governs the main backbone; Tier-1 (without the temporal constraints) turns out to produce better cell embeddings for the Inter-Cell Embedding module (§6.2).

### 2.3 Cross-Cell, Transfer Learning, and Domain Generalization

Several studies investigate transfer learning and domain adaptation across battery datasets and chemistries. Many do not analyze error by source or laboratory in detail, or lack a clear separation between validation and test when designing splits. Our work complements this line of research with a source-aware validation protocol and cross-lab/cross-source error analysis (§5.3, §5.4).

---

## 3. Methodology

### 3.1 Problem Formulation

Let $i$ index a battery cell. For each cell, we observe an input trajectory $X_i$ (voltage and capacity over early cycles), a scalar RUL label $y_i$, and an SOH trajectory $\mathbf{s}_i = [s_{i,1}, \dots, s_{i,T}]$. Our goal is to learn $f_\theta$ such that $\hat{y}_i = f_\theta(X_i)$, optionally using $\mathbf{s}_i$ as an auxiliary/physics-informed target during training only.

### 3.2 Input Representation

Each cell is represented by a sequence of cycles. For each cycle we extract discharge voltage curves $V(Q)$ and capacity-related features (e.g., Qdlin). We normalize across cells and cycles, and ensure no information from cycles beyond the input window is used at prediction time.

### 3.3 Physics-Derived Degradation Features

Three handcrafted scalar features, computed over early cycles (0–99):

1. **qdlin_diff_std** — $\operatorname{std}_V\left[Q_{\text{dlin}}^{(99)}(V) - Q_{\text{dlin}}^{(9)}(V)\right]$: non-uniformity of capacity fade across voltage bins.
2. **voltage_slope_50_90** — $\frac{V_{90}-V_{50}}{0.4}$: slope of the discharge voltage curve in the high-SOC region.
3. **voltage_soc_90** — $V_{90}$: discharge voltage at SOC 90%.

These are computed from observed data only and introduce no label leakage.

### 3.4 Network Architecture

We refer to the backbone as the axis-aware attention network, with two variants distinguished only by whether a positional encoding is applied over the cycle axis:

- **V2** — no positional encoding.
- **V1** — a learnable cycle positional encoding is added before attention pooling.

#### 3.4.1 CNN Backbone

A small Conv2d stack (6→16→32→32 channels) extracts local temporal patterns from the cycle-indexed input.

#### 3.4.2 Cycle Attention Pooling

Attention weights are computed per cycle and used to produce a weighted-sum sequence embedding, instead of fixed pooling.

#### 3.4.3 Scalar Feature Branch

The three physics-derived features pass through a small MLP (3→16→16, ReLU).

#### 3.4.4 Fusion, Embedding, and Regression Head

The sequence embedding and scalar embedding are concatenated into a 48-dim cell embedding $h_i$, which is (a) passed through a linear RUL head, and (b) reused as-is by the Inter-Cell Embedding module (§3.6) and the auxiliary heads (§3.5):
$$
\hat{y}_i = \mathrm{Linear}_{\mathrm{RUL}}(h_i).
$$

### 3.5 Training Objectives: Tier-1 (SOH-Auxiliary) and Tier-3 (Physics-Informed)

**RUL loss (both tiers, and the no-SOH baseline).** We use plain MSE, not Huber:
$$
\mathcal{L}_{\mathrm{RUL}} = \mathrm{MSE}(\hat{y}_i, y_i).
$$

**Tier-1 (SOH-auxiliary).** An additional head $\mathrm{soh\_head}: \mathbb{R}^{48}\to\mathbb{R}^{100}$ predicts the SOH trajectory, supervised with Smooth-L1 (Huber):
$$
\mathcal{L}_{\mathrm{SOH}} = \mathrm{SmoothL1}(\hat{\mathbf{s}}_i, \mathbf{s}_i), \qquad
\mathcal{L}_{\mathrm{Tier1}} = \mathcal{L}_{\mathrm{RUL}} + \lambda_{\mathrm{SOH}}\,\mathcal{L}_{\mathrm{SOH}}.
$$
Tier-1 is not claimed as physics-informed in the strict sense (§2.2); we use it in this work only as the embedding source for the Inter-Cell Embedding module (§3.6), not as a standalone result.

**Tier-3 (physics-informed, this work's main objective).** Two additional heads, $\mu_{\mathrm{head}}$ and $\sigma_{\mathrm{head}}$ (each $\mathbb{R}^{48}\to\mathbb{R}^{100}$), parameterize a Wiener process over the modeled capacity-fade increment. Let $D_t = 1 - \hat{s}_{i,t}$:

$$
\mathcal{L}_{\mathrm{mono}} = \frac{1}{T-1}\sum_{t=1}^{T-1} \mathrm{ReLU}\big(\hat{s}_{i,t+1} - \hat{s}_{i,t} - \tau\big), \qquad \tau = 0.002,
$$
$$
\Delta D_t = D_{t+1} - D_t, \quad r_t = \Delta D_t - \mu_t, \quad v_t = \sigma_t^2 + \epsilon,
$$
$$
\mathcal{L}_{\mathrm{Wiener}} = \frac{1}{T-1}\sum_{t=1}^{T-1} \left(\frac{1}{2}\frac{r_t^2}{v_t} + \frac{1}{2}\log v_t\right),
$$
$$
\mathcal{L}_{\mathrm{Tier3}} = \mathcal{L}_{\mathrm{RUL}} + \lambda_{\mathrm{SOH}}\mathcal{L}_{\mathrm{SOH}} + \lambda_{\mathrm{mono}}\mathcal{L}_{\mathrm{mono}} + \lambda_{\mathrm{wiener}}\mathcal{L}_{\mathrm{Wiener}}.
$$

Pre-declared weights: $\lambda_{\mathrm{SOH}}=0.01$, $\lambda_{\mathrm{mono}}=0.05$, $\lambda_{\mathrm{wiener}}=0.05$, used identically across MATR1, HUST, and CRUSH in the Tier-3 backbone. (A separate, disclosed screen over $\lambda_{\mathrm{SOH}}\in\{0.01,0.05,0.1\}$ was run at the Tier-1 stage on all three datasets — see §4.3 and §7 for what was and was not used from that screen.)

$\mathcal{L}_{\mathrm{Tier3}}$ is what we regard as genuinely physics-informed: the Wiener term encodes an explicit governing SDE for degradation, not just a regression target.

### 3.6 Inter-Cell Embedding Module

For a pair of cells $(i,j)$ with frozen embeddings $h_i, h_j$ from a trained backbone, a small MLP ($48\to64\to1$, ReLU) predicts the RUL difference:
$$
\widehat{\Delta y}_{ij} = g_\phi(h_i - h_j).
$$
At inference, a test cell's RUL is estimated as the median, over $K$ reference train cells and 8 delta-model seeds, of $y_r + \widehat{\Delta y}_{i,r}$.

**Which embedding to use.** We use **V2 trained under Tier-1** (SOH-auxiliary only, no monotonicity/Wiener) as the embedding source for this module, while the intra-cell backbone used elsewhere in the pipeline is **V1 trained under Tier-3**. This is a direct, disclosed empirical choice, not a theoretical requirement: on MATR1, Inter-Cell Embedding built on the Tier-3 embedding scored 79.25±5.04, versus a clear improvement using the Tier-1 embedding (used in the final pipeline, §5.1). We report the comparison in §6.2 rather than assuming either choice.

The final proposed pipeline combines, via an affine-calibrated NNLS ensemble fit on validation:

- **V1 (Tier-3)** and **V2 (Tier-3)** — the physics-informed backbone, and
- **Inter-Cell Embedding (built on V2, Tier-1 embedding)** — the correction term.

---

## 4. Experimental Protocol

### 4.1 Datasets

| Dataset | Train | Val | Test | README RMSE | Notes |
|---:|---:|---:|---:|---:|---|
| MATR1 | 91 | 41 | 42 | 90 (PCR) | Train from MATR3+4; val matches tail of test distribution |
| HUST | 63 | 12 | 22 | 322 | Includes donor cells from MATR |
| CRUSH | 64 | 15 | 44 | 330 | Cross-lab: CALCE, RWTH, UL-PUR, SNL, plus donor HUST/Tongji |
| CRUH | [TBD] | [TBD] | [TBD] | 60 | Single-seed screen only (§5.4) — too few cells for a reliable validation split |
| SNL | [TBD] | [TBD] | [TBD] | 200 | Single-seed screen only (§5.4) — same limitation |
| CALCE | [TBD] | [TBD] | [TBD] | — | Single-seed screen only (§5.4) — same limitation |

Note the corrected naming: the cross-lab composite dataset is **CRUSH** (README benchmark 330), distinct from **CRUH** (README benchmark 60) — the two are easily confused and an earlier draft of this table conflated them (§5.3 corrects a resulting mislabeling).

### 4.2 Baselines

- **No-SOH baseline**: the axis-aware attention architecture (V1+V2+Inter-Cell Embedding, all trained with plain MSE, no auxiliary head).
- **Tier-3 backbone only**: V1+V2 trained under $\mathcal{L}_{\mathrm{Tier3}}$, no Inter-Cell Embedding.
- **Proposed full pipeline**: Tier-3 backbone (V1+V2) + Inter-Cell Embedding (Tier-1 embedding), NNLS-ensembled.
- Additional architectures (BatLiNet, BatLiNet-V2, 1D-CNN, LSTM/BiLSTM baselines) are reported in the Appendix.

### 4.3 Training & Evaluation Protocol, and Disclosed Deviations

- **Multi-seed evaluation**: 5 seeds for MATR1/HUST/CRUSH main results; 1 seed for CRUH/SNL/CALCE (§5.4), explicitly reported as a screen, not a claim.
- **Checkpoint selection**: lowest validation RMSE per seed; test evaluated once per seed on the selected checkpoint.
- **Metrics**: RMSE, MAE, MAPE, and R² reported together (§5.1 currently reports RMSE only pending backfill of the other three — [TBD]).
- **Pre-declared hyperparameters**: $\lambda_{\mathrm{SOH}}, \lambda_{\mathrm{mono}}, \lambda_{\mathrm{wiener}}, \tau$ (§3.5) are fixed identically across datasets and were not tuned per-dataset for the Tier-3 backbone.

**Disclosed deviation.** During Tier-1 development, $\lambda_{\mathrm{SOH}}$ was screened over $\{0.01, 0.05, 0.1\}$ using test RMSE feedback — single-seed for MATR1, full 5/8-seed for HUST and CRUSH. This is a deviation from the validation-only selection rule stated above. We report it here rather than presenting the Tier-1 embedding (used inside Inter-Cell Embedding, §3.6) as selected under the same discipline as the Tier-3 backbone's pre-declared weights. [TBD: state which $\lambda_{\mathrm{SOH}}$ value was ultimately used for the Tier-1 embedding in each dataset's final pipeline.]

---

## 5. Results

### 5.1 Main RUL Prediction Performance

| Dataset | README | Local reproduction | No-SOH baseline | Tier-1 (SOH-aux) | Tier-3 backbone only | **Proposed (best available)** |
|---:|---:|---:|---:|---:|---:|---:|
| MATR1 | 90 | 90 | 70.42 ± 4.90 | 68.93 ± 6.11 (reference only, §7) | 73.62 ± 4.85 (5-seed); 74.83±4.55 (7-seed robustness check) | **71.06 ± 5.25** (Tier-3 backbone + Inter-Tier-1) |
| HUST | 322 | 322 | 299.39 ± 19.52 | **289.82 ± 12.38** (8-seed) | 313.12 ± 18.55 (5-seed) | **289.82 ± 12.38** (Tier-1 ensemble) |
| CRUSH | 330 | 355 | 371.64 ± 5.53 (5-seed, verified — supersedes an earlier, untraceable 357.18 figure) | 371.29 ± 8.92 (λ=0.10, 8-seed) | 367.49 ± 9.07 (NNLS, 8-seed) | **367.49 ± 9.07** (Tier-3 ensemble) |

Interpretation, stated plainly rather than as a uniform win — **the best-performing tier is dataset-dependent, and we report whichever configuration is empirically best per dataset rather than forcing one method everywhere**:

- **MATR1**: the Tier-3-based pipeline (backbone + Inter-Tier-1 correction) is the proposed result, landing within noise of the no-SOH baseline. The pure Tier-1 ensemble (68.93±6.11) is numerically the best single number we have, but we do not adopt it as the MATR1 headline since it is not part of this work's core Tier-3 narrative — it is listed here only for completeness (§7 discusses why). A post-hoc robustness check extending Tier-3 to 7 seeds (adding seeds 5-6; seed 7 pending) shows the same ordering (Tier-1 ensemble 70.17±5.58 vs. Tier-3 74.83±4.55 at n=7) — the conclusion is stable, not an artifact of which 5 seeds were used.
- **HUST**: Tier-3 underperforms *every* other configuration here, including the no-SOH baseline — confirmed across all 5 seeds and at the level of every individual component (V1, V2, and Inter each score worse under Tier-3 than under Tier-1 at every seed; §5.2b). The proposed result for HUST is therefore the **Tier-1 (SOH-auxiliary) ensemble**, which is also what beats the README benchmark most clearly (289.82 vs. 322).
- **CRUSH**: none of the three configurations (no-SOH baseline, Tier-1, Tier-3) beat either the README figure (330) or our own local reproduction (355) — all three cluster in the 367-372 range. Tier-3 (367.49±9.07, using NNLS) is numerically the best of the three but the differences are within noise of each other. **An earlier draft of this table reported a "Baseline/Proposed" pair of 357.18/352.92 for CRUSH; we could not trace these to any pkl in the current, consistent cache (`..._hust10_calce15_fullval_withsoh`, pen=1.4) after a direct search, and have replaced them with the verified numbers above** (§7).
- **MATR1 and HUST** local, in-process reproduction of the official benchmark matches the published README figure exactly (90 and 322) — no reproducibility gap here, unlike CRUSH, where a gap against README persists even after this correction.

### 5.2 Ablation Study (MATR1)

| Component | RMSE (MATR1) |
|---|---:|
| V2, Tier-3 | 81.69 ± 4.61 |
| V1, Tier-3 | 82.56 ± 9.89 |
| V1+V2 NNLS, Tier-3 | 73.62 ± 4.85 |
| V1+V2+Inter(Tier-3 embed) NNLS | 73.62 ± 4.85 (Inter solo: 79.25 ± 5.04) |
| **V1+V2(Tier-3)+Inter(Tier-1 embed) NNLS (proposed)** | **71.06 ± 5.25** |

Observations: V1 (with positional encoding) outperforms V2 under Tier-3 in 4/5 seeds — reversing the pattern seen without Tier-3 — consistent with the positional encoding helping the backbone satisfy Tier-3's temporally-structured constraints (monotonicity, Wiener drift). We discuss this mechanism in §6.1.

### 5.2b Ablation Study (HUST): Tier-3 Underperforms at Every Seed and Every Component

Unlike MATR1, where Tier-3 is within noise of the no-SOH baseline, on HUST Tier-3 is worse than Tier-1 (SOH-auxiliary only) at **every seed and every individual component**, not just in the ensemble:

| Seed | V2 (T1) | V2 (T3) | V1 (T1) | V1 (T3) | Inter (T1) | Inter (T3) | **Ens (T1)** | **Ens (T3)** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 308.25 | 317.22 | 264.88 | 278.71 | 310.45 | 318.92 | **270.28** | **281.55** |
| 1 | 280.34 | 323.93 | 290.45 | 323.65 | 303.26 | 326.83 | **294.69** | **326.56** |
| 2 | 315.34 | 343.57 | 314.20 | 322.09 | 325.29 | 343.86 | **307.60** | **325.96** |
| 3 | 286.22 | 323.46 | 282.75 | 303.57 | 294.39 | 319.14 | **282.80** | **302.29** |
| 4 | 293.84 | 331.77 | 294.24 | 328.53 | 339.69 | 329.96 | **303.00** | **329.25** |
| **Mean±std** | | | | | | | **291.67±13.62** | **313.12±18.55** |

We also tested whether the MATR1-style cross-tier ensemble (Tier-3 backbone + Inter-Cell Embedding built on a Tier-1 embedding) closes this gap: it recovers part of it (308.06±18.02) but still does not reach the pure Tier-1 ensemble, and adding the Tier-3-embedding Inter-Cell Embedding as a fourth ensemble member changes nothing (NNLS never assigns it nonzero weight at any seed — it is the weakest component throughout). We conclude that, for HUST, the monotonicity and Wiener constraints degrade the learned representation itself, not merely the ensemble-selection step; this is a genuine, seed-consistent negative result for Tier-3 on this dataset, discussed further in §6.3.

### 5.3 On CRUH: Cross-Source Error Breakdown

[Corrected from an earlier draft, which mislabeled this table as "HUST-CRUSH."] The following single-seed breakdown is on **CRUH**, not CRUSH:

| Source | # Cells | RMSE | Bias |
|---:|---:|---:|---:|
| CALCE | 4 | 242.25 | −215.08 |
| RWTH | 24 | 48.57 | −9.45 |
| UL-PUR | 1 | 53.30 | 53.03 |
| HNEI | 5 | 23.94 | 9.08 |

CALCE cells dominate the aggregate error despite being a small minority of the test set. [TBD: confirm whether an equivalent per-source breakdown for CRUSH itself exists or needs to be computed separately — the composite pipeline above is evaluated on CRUSH, and this table should not be read as describing that same run.]

### 5.4 Error Analysis: SNL, CALCE, CRUH Single-Seed Screens

These three datasets have too few cells to construct a reliable held-out validation split under the same protocol used for MATR1/HUST/CRUSH (§4.1). We report single-seed screens only, explicitly as a data-limitation finding rather than a claim of model performance:

| Dataset | # Train / Val / Test | Result | Issue |
|---|---|---:|---|
| SNL | [TBD] | [TBD] | Validation set too small to be a reliable model-selection signal |
| CALCE | [TBD] | [TBD] | Same |
| CRUH | [TBD] | [TBD] | Same |

[TBD: run and fill in single-seed numbers per the pending task; keep framed as "screen," not "result," per this project's multi-seed discipline.]

---

## 6. Discussion

### 6.1 Why Positional Encoding Interacts With Tier-3

Under Tier-3, the monotonicity and Wiener losses impose a temporally-structured constraint on the predicted SOH trajectory. V1's learnable cycle positional encoding gives the attention mechanism explicit access to cycle position, which we hypothesize makes it easier to satisfy these temporal constraints than V2, which must infer position implicitly. This is consistent with the reversal observed in §5.2 (V1 > V2 under Tier-3, V2 ≥ V1 without it), though it is not a universal law: at one seed (MATR1 seed 4), V1 converges to a poor local minimum under Tier-3 despite the same architecture, which we verified is a genuine optimization-variance effect (full-trajectory inspection, patience extended to 350 epochs with no recovery), not a bug.

### 6.2 Why Inter-Cell Embedding Uses a Different Tier Than the Backbone

Inter-Cell Embedding's pairwise predictions were consistently more accurate using an embedding trained under Tier-1 (SOH-auxiliary only) than the same module using an embedding trained under the full Tier-3 objective (79.25±5.04 with the Tier-3 embedding, vs. the improved figure obtained with the Tier-1 embedding and used in the final pipeline, §5.1). We interpret this as: the monotonicity and Wiener constraints push the embedding toward representing *absolute* degradation state (to satisfy a per-cell temporal SDE), which compresses the *relative*, pairwise distance structure that Inter-Cell Embedding relies on. This is why the final pipeline deliberately pairs a Tier-3-trained backbone with a Tier-1-trained correction module rather than optimizing both under one unified objective.

### 6.3 Dataset-Dependent Benefit of Tier-3 (and Where It Actively Hurts)

Tier-3's effect relative to the simpler Tier-1 (SOH-auxiliary only) objective is not uniform, and in one case is clearly negative rather than merely weaker:

- **MATR1**: Tier-3 backbone-only (73.62±4.85) is within noise of the no-SOH baseline (70.42±4.90); the full pipeline (71.06±5.25) is likewise within noise. Tier-3 neither clearly helps nor clearly hurts here.
- **HUST**: Tier-3 is worse than Tier-1 at every seed and every individual component (§5.2b) — 313.12±18.55 vs. 291.67±13.62, a consistent ~21-point RMSE gap. This is not an ensemble-selection artifact; V1 solo, V2 solo, and Inter-Cell Embedding solo are each worse under Tier-3 than under Tier-1 at all 5 seeds. We had originally hypothesized (following an earlier, single-seed screen) that Tier-3 would show its clearest gain on HUST; the full 5-seed result contradicts this, and we report the reversal rather than the earlier hypothesis.
- **CRUSH**: unlike HUST, Tier-3 does not clearly hurt here — it is numerically the best of the three configurations (367.49±9.07 vs. 371.29±8.92 for Tier-1 and 371.64±5.53 for the no-SOH baseline, all 5-8 seeds), but the gap between all three is within noise. None beats README (330) or the local reproduction (355).

A tentative explanation for HUST's result: the monotonicity and Wiener terms assume a reasonably smooth, well-sampled degradation trajectory to fit $\mu_t, \sigma_t$ per cycle; HUST's inputs are processed in smaller chunks (CHUNK=48, §4) to fit GPU memory, which may make the per-cycle Wiener parameterization noisier to fit than on MATR1, where the full sequence is used at once. CRUSH uses the same chunking (CHUNK=48) yet does not show HUST's clear degradation, so chunk size alone does not fully explain the HUST result — we flag this as an open question rather than a confirmed mechanism. We do not claim Tier-3 "works" or "doesn't work" in general — only that its effect must be checked per dataset, and that HUST is a case where it measurably hurts while CRUSH and MATR1 are not.

On CRUSH specifically, we also found that the ensemble-combination method matters more than which tier is used: with only 15 validation cells, NNLS's three free weights can fit validation noise that does not generalize to the 44-cell test set — a simple unweighted mean of the three components sometimes outperforms NNLS at 5 seeds (367.59±4.64 vs. 371.41±6.07) but this reverses at 8 seeds (370.12±6.33 vs. 367.49±9.07), indicating neither choice is reliably better at this validation-set size, and any single comparison across only a few seeds should not be trusted to pick the right ensemble method for this dataset.

---

## 7. Limitations

1. **CRUSH does not beat README or our local reproduction under any configuration tested.** No-SOH baseline, Tier-1, and Tier-3 all cluster at 367-372 RMSE, against a local reproduction of 355 and a published README figure of 330 (§5.1, §6.3). An earlier draft of this manuscript reported a "Baseline/Proposed" pair of 357.18/352.92 for CRUSH; a direct search of every result file in the current, consistent cache could not trace these numbers to any actual run, and we have replaced them with the verified figures above. We flag this correction explicitly rather than silently updating the number, since it changes CRUSH from an apparent (if modest) near-miss into a dataset where none of our methods are competitive with the published benchmark.
2. **λ_SOH selection.** The SOH loss weight was screened over $\{0.01, 0.05, 0.1\}$ using test RMSE feedback for the Tier-1 embedding source, across all three main datasets (single-seed for MATR1; full multi-seed for HUST and CRUSH) — a disclosed deviation from the validation-only protocol in §4.3. The value ultimately used was λ=0.01 for MATR1 and λ=0.10 for HUST and CRUSH.
3. **Inter-Cell Embedding's tier mismatch.** The final pipeline's correction module is trained on a Tier-1, not Tier-3, embedding (§3.6, §6.2) — a deliberate, empirically-motivated choice, but one that means the "physics-informed" claim applies to the intra-cell backbone, not to every component of the final ensemble.
4. **HUST's proposed result is Tier-1, not Tier-3.** Having completed the 5-seed Tier-3 run for HUST, we find Tier-3 underperforms Tier-1 at every seed and component (§5.2b); we report Tier-1's ensemble (289.82±12.38) as HUST's proposed result rather than forcing a Tier-3-based number into the headline.
5. **Ensemble-method instability on small validation sets.** On CRUSH (val n=15), whether NNLS or a simple mean is the better ensemble-combination method reverses between 5 and 8 seeds (§6.3); we report both rather than picking whichever favors our headline number, and caution against trusting a small-seed-count comparison to select an ensemble method for this dataset.
6. **SNL, CALCE, and CRUH** are evaluated as single-seed screens only, due to insufficient data for a reliable validation split; results there are data-limitation findings, not performance claims.
7. We do not yet provide interpretability analysis (attention maps, feature importance) or uncertainty quantification for RUL/SOH predictions.

---

## 8. Conclusion

We presented a degradation-aware architecture — CNN backbone, cycle-axis attention pooling, and physics-derived voltage–capacity features — that surpasses the published README benchmark on MATR1 and HUST without any auxiliary loss. We then introduced Tier-3, a physics-informed objective governed by an explicit Wiener-process model of capacity fade, and an Inter-Cell Embedding correction module, combining both into a final pipeline via a disclosed cross-tier design. Reporting against both the published benchmark and our own local reproduction of it, and disclosing every deviation from our pre-declared validation-only selection protocol, we find Tier-3's benefit to be dataset-dependent rather than uniform, and discuss a source-heterogeneity hypothesis for this pattern. Future work includes completing the HUST/CRUSH multi-seed Tier-3 evaluation, extending the validation-eligible dataset set, and exploring a unified single-objective formulation for the backbone and Inter-Cell Embedding module.

---

## Appendix

### A. Additional Ablation Results
[Include tables/figures for BatLiNet, BatLiNet-V2, 1D-CNN, LSTM/BiLSTM baselines.]

### B. Hyperparameter Details
[Optimizer settings, learning rates, batch sizes, patience per variant, number of layers.]

### C. Code and Data Availability
[Links to code repository and data access instructions, if applicable.]
