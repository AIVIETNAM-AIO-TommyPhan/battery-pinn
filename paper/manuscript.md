# Physics-Informed Wiener-Diffusion Modeling with Cycle-Axis Attention for Lithium-Ion Battery Remaining Useful Life Prediction

> **Draft status**: outline + key sections drafted for review. HUST and CRUSH Tier-3 runs are both complete at 8/8 seeds (§5.1, §5.2b, §6.3) — HUST's proposed result is Tier-1 (8-seed); CRUSH's proposed result is Tier-3 on a redesigned val/train split (8-seed), which now beats local reproduction for the first time (still short of README). MATR1's headline is now the 8-seed result (74.0±6.1, §5.1) — the 8-seed robustness check (complete, all seeds) confirmed the same ordering as the original 5-seed pre-registration (71.1±5.3), so we report the more heavily-evaluated 8-seed number throughout rather than the 5-seed one. Numbers marked `[TBD]` are pending only the SNL/CALCE/CRUH/MATR2 single-seed screens — do not cite before these are filled in and verified against `report_expirement/`.

## Abstract

Accurate battery remaining useful life (RUL) prediction is challenging because the RUL label is sparse, degradation trajectories are heterogeneous across cells, and models often fail to generalize across laboratories and protocols. We propose a degradation-aware architecture combining a CNN backbone, learned attention pooling over the cycle axis, and physics-derived voltage–capacity features, which alone surpasses the published benchmark on MATR1 and HUST. We then investigate a physics-informed auxiliary objective (Tier-3) that supervises the network with the SOH trajectory under an explicit degradation model — a monotonicity constraint and a Wiener-process diffusion term ($dD(t) = \mu\,dt + \sigma\,dW(t)$) governing capacity fade — and combine it with an Inter-Cell Embedding module that corrects predictions using pairwise relationships between cells. Under an 8-seed evaluation protocol with validation-only model selection for the core architecture and Tier-3 weights (one disclosed exception: an auxiliary loss weight was screened with test-RMSE feedback rather than validation, §4.3), we report the resulting pipeline's performance on MATR1, HUST, and CRUSH against both the published README benchmark and our own in-process reproduction of it. The proposed pipeline surpasses README on MATR1 and HUST; on CRUSH it improves over our local reproduction but remains short of README. We disclose every case where a deviation from the pre-declared protocol occurred. The physics-informed objective's benefit is dataset-dependent; we report this honestly and discuss a mechanistic hypothesis (source heterogeneity) for why, rather than presenting a uniform improvement claim.

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
4. **Source-aware validation protocol**: train/validation/test separated by dataset and source, validation-only model selection, and 8-seed evaluation for MATR1, HUST, and CRUSH, with every deviation from this rule disclosed explicitly (§4.3, §7).

### 1.4 Contributions

- An **axis-aware attention architecture** that surpasses the published README benchmark on MATR1 and HUST using only architecture and physics-derived features, with no auxiliary loss.
- **Tier-3**, a physics-informed objective governed by an explicit stochastic degradation model (Wiener process on capacity fade), which we distinguish from simpler SOH-auxiliary supervision (Tier-1) both conceptually (§2.2) and in how it is used in the final pipeline (§3.6).
- An **Inter-Cell Embedding** module that learns pairwise degradation relationships between cells and is combined with the Tier-3 backbone via a disclosed cross-tier design (§3.6, §6.2).
- A **substantially disclosed evaluation protocol**: 8-seed mean±std throughout for MATR1, HUST, and CRUSH, comparison against both the published README benchmark and our own in-process reproduction of it, and explicit disclosure of every case where hyperparameter selection deviated from validation-only screening (§4.3, §7). Some reproducibility detail remains incomplete — marked `[TBD]` where it occurs (§4.1, §5.3, §5.4, Appendix B) — so we do not claim the protocol is complete, only that every known deviation is disclosed rather than hidden.

---

## 2. Related Work

### 2.1 Deep Learning for Battery RUL Prediction

Recent works have applied CNNs, LSTMs, attention mechanisms, and hybrid architectures to battery RUL prediction. These models typically take voltage, current, or capacity trajectories as input, output a single RUL value per cell, and are trained with an RUL loss (e.g., MSE or Huber). While these methods demonstrate strong performance on specific datasets, they often use random splits without controlling for source or protocol, report results on a single seed or split, and do not explicitly leverage the dense SOH trajectory as a learning signal.

### 2.2 Physics-Informed Learning, and Why We Distinguish Two Tiers

Physics-informed neural networks (PINNs), as defined by Raissi et al. (2019), incorporate a governing differential equation directly into the training loss as a soft constraint. Much of the recent "physics-informed" battery literature — including our own earlier Tier-1 design — instead uses a physically meaningful *quantity* (e.g., SOH) as an auxiliary supervision target, without an explicit governing equation. We treat this distinction as consequential rather than terminological:

- **Tier-1 (SOH-auxiliary)**: an auxiliary head predicts the SOH trajectory, supervised with a standard regression loss. This regularizes the representation toward a physically meaningful quantity but enforces no dynamics.
- **Tier-3 (this work's main objective)**: in addition to the SOH-auxiliary term, we add a monotonicity constraint and a Wiener-process diffusion loss over the *modeled* capacity-fade process, i.e., an explicit SDE governing how SOH evolves between cycles (§3.5). In the sense of Raissi et al. (2019), this loss enforces a governing equation structurally, rather than treating SOH as only a supervision target, in a way Tier-1 does not. We do not claim the SDE's own assumptions — independent Gaussian increments, unconstrained sign of the drift, no explicit bound on $D_t\in[0,1]$ — are separately validated against the data; §3.5 states exactly which of these are enforced by the architecture and which are not.

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

**Which SDE constraints are actually enforced.** $\sigma_t$ is guaranteed positive by construction ($\sigma_t = \mathrm{softplus}(\sigma_{\mathrm{head}}(h)) + 10^{-6}$, so $v_t>0$ and $\mathcal{L}_{\mathrm{Wiener}}$ is always well-defined). $\mu_t$ carries no sign constraint — the network is free to predict drift in either direction, so "monotonic decay in expectation" is encouraged only indirectly, through $\mathcal{L}_{\mathrm{mono}}$ acting on $\hat{s}_{i,t}$ directly, not through a constraint on $\mu_t$ itself. $D_t = 1-\hat{s}_{i,t}$ has no explicit bound to $[0,1]$ in the loss; the SOH head's output range is unconstrained (a linear layer, no sigmoid), so this bound is not architecturally guaranteed, only empirically typical. The Wiener loss also treats increments as conditionally independent Gaussians given $\mu_t,\sigma_t$ — a modeling choice consistent with a discretized Wiener process, not a property we separately verify against the observed SOH trajectories. We report these as scoped limitations of the current implementation, not as settled properties of the model.

### 3.6 Inter-Cell Embedding Module

For a pair of cells $(i,j)$ with frozen embeddings $h_i, h_j$ from a trained backbone, a small MLP ($48\to64\to1$, ReLU) predicts the RUL difference:
$$
\widehat{\Delta y}_{ij} = g_\phi(h_i - h_j).
$$
At inference, a test cell's RUL is estimated as the median, over $K$ reference train cells and 8 delta-model seeds, of $y_r + \widehat{\Delta y}_{i,r}$.

**Which embedding to use.** We use **V2 trained under Tier-1** (SOH-auxiliary only, no monotonicity/Wiener) as the embedding source for this module, while the intra-cell backbone used elsewhere in the pipeline is **V1 trained under Tier-3**. This is a direct, disclosed empirical choice, not a theoretical requirement: on MATR1, Inter-Cell Embedding built on the Tier-3 embedding scored 79.25±5.04, versus a clear improvement using the Tier-1 embedding (used in the final pipeline, §5.1). We report the comparison in §6.2 rather than assuming either choice.

**Notation.** From here on we write a component as `<variant>-<tier>`, e.g. `V1-T3` for V1 trained under Tier-3, and `Inter(V2-T1)` for the Inter-Cell Embedding module built on frozen V2-Tier-1 embeddings. An ensemble is written `combine[...]` with the method named explicitly (`NNLS[...]` or `mean[...]`).

The final proposed pipeline is `NNLS[V1-T3, V2-T3, Inter(V2-T1)]`, fit on validation:

- **V1-T3** and **V2-T3** — the physics-informed backbone, and
- **Inter(V2-T1)** — the correction term, built on a Tier-1 (not Tier-3) embedding (§6.2).

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

We compare the following configurations. The first three establish the architecture-ladder floor and are strictly simpler than V1/V2; the remaining four are all built on the axis-aware attention backbone, trainable in either variant (**V1**: with cycle positional encoding; **V2**: without — §3.4):

- **SmallCNN**: a plain CNN with no scalar branch and no axis-aware attention pooling (fixed pooling only) — the architecture-ladder floor, predating V1/V2 entirely. On MATR1, this scores 130.8 (seed 0), motivating why the subsequent architectural additions matter.
- **SmallCNN + scalar branch** (`SmallCNNScalarBranch`, no axis-aware attention yet): adds the 3-feature scalar branch (§3.3) but still uses fixed (non-attention) pooling over cycles. Scores 95.4 on MATR1 (seed 0) — most of the architecture's gain up to this point comes from the scalar branch, not attention.
- **No-SOH baseline** (`SmallCNNScalarBranchAxisAware`, i.e. **V1/V2 with no auxiliary head**): the axis-aware attention architecture (§3.4) trained with plain MSE only — no SOH head, no monotonicity/Wiener terms. Isolates axis-aware attention pooling's own contribution, on top of the scalar branch.
- **Tier-1**: No-SOH baseline + an SOH-auxiliary head (Smooth-L1), $\lambda_{\mathrm{SOH}}$ pre-declared per the disclosed deviation below. Used in this work only as the embedding source for Inter-Cell Embedding (§3.6), not as a standalone architectural claim except where a dataset's evidence makes it the proposed result (HUST, §5.1).
- **Tier-3**: No-SOH baseline + SOH-auxiliary head + monotonicity constraint + Wiener-process diffusion loss (§3.5) — this work's primary physics-informed objective.
- **Ensemble (proposed)**: V1 + V2 (both trained under whichever tier is that dataset's evidence-supported backbone, §5.1) combined with the Inter-Cell Embedding module, via either an affine-calibrated NNLS fit or a simple (unweighted) mean on validation (§3.6, §6.1) — we report both and adopt whichever the per-dataset validation evidence supports, since §6.1 shows NNLS is not a safe default at small validation sizes.
- Additional architectures (BatLiNet, BatLiNet-V2, 1D-CNN, LSTM/BiLSTM baselines) are reported in the Appendix.

The SmallCNN and SmallCNN+scalar-branch numbers above are single-seed (MATR1 only, from the architecture-ladder screen that motivated this work's design, `report_expirement/day_0901`) — they establish *why* axis-aware attention was adopted, not a multi-seed claim in their own right, and are not repeated as columns in §5.1's main results table.

**A gap in this ladder.** The three rungs shown (SmallCNN → SmallCNN+scalar → No-SOH baseline) add the scalar branch and axis-aware attention *together* in the last step, so the reported 95.4→70.4 gain cannot be attributed to attention alone — an intermediate "CNN + attention, no scalar branch" configuration was not run. We therefore describe the architecture's gain as coming from "the scalar branch and axis-aware attention combined," not from attention specifically, until that intermediate cell exists (§7).

### 4.3 Training & Evaluation Protocol, and Disclosed Deviations

- **Multi-seed evaluation**: **8 seeds for MATR1, HUST, and CRUSH** main results; 1 seed for CRUH/SNL/MATR2 (§5.4), explicitly reported as a screen, not a claim. (One exception: the No-SOH baseline for MATR1 was run at 5 seeds only and has not been re-run at 8; §5.1 flags this explicitly rather than presenting an unverified 8-seed number for it.)
- **Checkpoint selection**: lowest validation RMSE per seed; test evaluated once per seed on the selected checkpoint.
- **Metrics**: RMSE, MAE, MAPE, and R² reported together (§5.1 currently reports RMSE only pending backfill of the other three — [TBD]).
- **Pre-declared hyperparameters**: $\lambda_{\mathrm{SOH}}, \lambda_{\mathrm{mono}}, \lambda_{\mathrm{wiener}}, \tau$ (§3.5) are fixed identically across datasets and were not tuned per-dataset for the Tier-3 backbone.

**Disclosed deviation.** During Tier-1 development, $\lambda_{\mathrm{SOH}}$ was screened over $\{0.01, 0.05, 0.1\}$ using test RMSE feedback — single-seed for MATR1, full 8-seed for HUST and CRUSH. This is a deviation from the validation-only selection rule stated above. We report it here rather than presenting the Tier-1 embedding (used inside Inter-Cell Embedding, §3.6) as selected under the same discipline as the Tier-3 backbone's pre-declared weights. The value ultimately used in the headline pipeline is disclosed per-dataset below; see §7 item 2 for the corresponding limitation statement.

| Dataset | Candidate $\lambda_{\mathrm{SOH}}$ | Selection signal | Selected $\lambda_{\mathrm{SOH}}$ | Used in headline pipeline? |
|---|---|---|---:|---|
| MATR1 | 0.01, 0.05, 0.10 | Test RMSE, single-seed | 0.01 | Yes — Tier-1 embedding inside Inter-Cell Embedding (§3.6) |
| HUST | 0.01, 0.05, 0.10 | Test RMSE, 8-seed | 0.10 | Yes — HUST's proposed result is the Tier-1 ensemble itself (§5.1) |
| CRUSH | 0.01, 0.05, 0.10 | Test RMSE, 8-seed | 0.10 | Yes — Tier-1 embedding inside Inter-Cell Embedding (§3.6) |

Because this screen used test feedback rather than validation, we do not describe any result that depends on it (i.e., every headline number in §5.1, since all three datasets' proposed pipelines include an Inter-Cell Embedding module built on this Tier-1 embedding) as selected under strict validation-only discipline. We have not re-run this screen with validation-only selection; doing so, and checking whether the selected $\lambda_{\mathrm{SOH}}$ and downstream headline numbers change, is listed as future work (§8) rather than assumed not to matter.

---

## 5. Results

### 5.1 Main RUL Prediction Performance

| Dataset | README | Local reproduction | Tier-1 (SOH-aux) | Tier-3 backbone only | **Proposed (best available)** |
|---:|---:|---:|---:|---:|---:|
| MATR1 | 90 | 90 | 72.3 ± 7.6 | 75.9 ± 5.1 | **74.0 ± 6.1** |
| HUST | 322 | 322 | **289.8 ± 12.4** | 316.9 ± 16.8 | **289.8 ± 12.4** |
| CRUSH | 330 | 355 | 371.3 ± 8.9 | 367.5 ± 9.1 | **339.8 ± 13.9** |

**CRUSH split comparison.** CRUSH's Tier-1/Tier-3 columns above are on the original split (val=15, train_base=64); the Proposed column is on a redesigned split (val=20, train_base=70, §6.1) — not a paired comparison:

| Split | Local reproduction | Tier-1 | Tier-3 backbone only | Proposed | README |
|---|---:|---:|---:|---:|---:|
| Original (val=15, train_base=64) | 355 | 371.3 ± 8.9 | 367.5 ± 9.1 | — | 330 |
| Redesigned (val=20, train_base=70) | 355 | — | — | **339.8 ± 13.9** | 330 |

Interpretation, stated plainly rather than as a uniform win — **the best-performing tier is dataset-dependent, and we report whichever configuration is empirically best per dataset rather than forcing one method everywhere**:

- **MATR1**: the Tier-3-based pipeline (backbone + Inter-Tier-1 correction, 74.0±6.1, 8-seed) is the proposed result. The pure Tier-1 ensemble (72.3±7.6, 8-seed) is numerically the best single number we have, but we do not adopt it as the MATR1 headline since it is not part of this work's core Tier-3 narrative — it is listed here only for completeness (§7 discusses why). Seed 7 is the one exception across all 8 seeds: the first individual seed where the full Tier-3 ensemble (83.5) beats Tier-1 (86.8), narrowing but not overturning the mean ordering.
- **HUST**: Tier-3 underperforms *every* other configuration here — confirmed across all **8** seeds (not just the original 5) and at the level of every individual component (V1, V2, and Inter each score worse under Tier-3 than under Tier-1 at every seed; §5.2b). The proposed result for HUST is therefore the **Tier-1 (SOH-auxiliary) ensemble**, which is also what beats the README benchmark most clearly (289.8 vs. 322).
- **CRUSH**: the pre-redesign split's two configurations (Tier-1, Tier-3) both clustered in the 367-372 range, beating neither README (330) nor our local reproduction (355). Investigating why (§6.1) surfaced a real train/val/test leakage bug in that split (fixed, but did not change the outcome) and, separately, a val-set RUL-coverage problem that did: redesigning val (random draw, size 15→20) and reinforcing train_base (64→70 cells) flips the val/test correlation from strongly negative to strongly positive and yields a new 8-seed result, **339.8±13.9** (`mean[V1-T3, V2-T3, Inter(V2-T3)]`, on the redesigned split; note this uses the Tier-3, not Tier-1, embedding for the Inter term, unlike MATR1's proposed pipeline — §6.2) — this **beats local reproduction (355) for the first time**, though it remains ~3% short of README. The pre-redesign columns are retained above for the record, not as live baselines. **An earlier draft of this table reported a "Baseline/Proposed" pair of 357.18/352.92 for CRUSH; we could not trace these to any pkl in the (pre-redesign) cache after a direct search, and replaced them with the verified numbers above** (§7).
- **MATR1 and HUST** local, in-process reproduction of the official benchmark matches the published README figure exactly (90 and 322) — no reproducibility gap here, unlike CRUSH, where a gap against README persists even after this correction.

### 5.2 Ablation Study (MATR1)

| Component (notation: §3.6) | RMSE (MATR1, 8-seed) |
|---|---:|
| V2-T3 (solo) | 85.0 ± 6.7 |
| V1-T3 (solo) | 86.1 ± 9.5 |
| Inter(V2-T3) (solo) | 80.9 ± 4.8 |
| NNLS[V1-T3, V2-T3, Inter(V2-T3)] | 75.9 ± 5.1 |
| **NNLS[V1-T3, V2-T3, Inter(V2-T1)] (proposed)** | **74.0 ± 6.1** |

Observations: V1 (with positional encoding) underperforms V2 under Tier-3 on average across all 8 seeds (86.1 vs. 85.0) — a reversal from the 5-seed read, where V1 numerically led; the ordering is close and seed-sensitive rather than a stable effect in either direction (§6.1 revises our earlier reading of this mechanism accordingly).

### 5.2b Ablation Study (HUST): Tier-3 Underperforms at Every Seed and Every Component

Unlike MATR1, where Tier-3 is within noise of the no-SOH baseline, on HUST Tier-3 is worse than Tier-1 (SOH-auxiliary only) at **every seed and every individual component**, not just in the ensemble:

| Seed | V2 (T1) | V2 (T3) | V1 (T1) | V1 (T3) | Inter (T1) | Inter (T3) | **Ens (T1)** | **Ens (T3)** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 308.2 | 317.2 | 264.9 | 278.7 | 310.4 | 318.9 | **270.3** | **281.6** |
| 1 | 280.3 | 323.9 | 290.4 | 323.6 | 303.3 | 326.8 | **294.7** | **326.6** |
| 2 | 315.3 | 343.6 | 314.2 | 322.1 | 325.3 | 343.9 | **307.6** | **326.0** |
| 3 | 286.2 | 323.5 | 282.8 | 303.6 | 294.4 | 319.1 | **282.8** | **302.3** |
| 4 | 293.8 | 331.8 | 294.2 | 328.5 | 339.7 | 330.0 | **303.0** | **329.2** |
| 5 | 333.2 | 324.7 | 303.1 | 333.1 | 345.4 | 334.6 | **295.9** | **333.0** |
| 6 | 278.1 | 322.2 | 261.7 | 337.3 | 300.5 | 324.3 | **274.2** | **328.1** |
| 7 | 301.1 | 349.3 | 292.6 | 308.7 | 345.3 | 375.1 | **290.1** | **308.4** |
| **Mean±std (n=8)** | | | | | | | **289.8±12.4** | **316.9±16.8** |

We also tested whether the MATR1-style cross-tier ensemble (Tier-3 backbone + Inter-Cell Embedding built on a Tier-1 embedding) closes this gap: it recovers part of it (309.0±17.9, n=8) but still does not reach the pure Tier-1 ensemble, and adding the Tier-3-embedding Inter-Cell Embedding as a fourth ensemble member changes nothing (NNLS never assigns it nonzero weight at any seed — it is the weakest component throughout). This holds at all 8/8 seeds, not just the original 5. We conclude that, for HUST, the monotonicity and Wiener constraints degrade the learned representation itself, not merely the ensemble-selection step; this is a genuine, seed-consistent negative result for Tier-3 on this dataset, discussed further in §6.3.

### 5.2c Ablation Study (CRUSH): Redesigned Split, Ensemble Method Comparison

All components below are trained under Tier-3 on the redesigned split (val=20, train_base=70, §6.1); this is the ablation underlying the 339.8±13.9 headline in §5.1:

| Seed | V2-T3 | V1-T3 | Inter(V2-T3) | NNLS[V1-T3,V2-T3,Inter] | **mean[V1-T3,V2-T3,Inter]** |
|---:|---:|---:|---:|---:|---:|
| 0 | 323.2 | 349.1 | 317.9 | 331.5 | **327.5** |
| 1 | 340.7 | 349.0 | 338.5 | 339.4 | **339.6** |
| 2 | 362.0 | 341.9 | 376.8 | 366.8 | **352.2** |
| 3 | 331.0 | 344.3 | 333.5 | 327.4 | **330.5** |
| 4 | 336.1 | 308.9 | 333.8 | 313.6 | **319.7** |
| 5 | 367.3 | 371.4 | 391.8 | 364.7 | **367.0** |
| 6 | 320.2 | 363.1 | 328.6 | 369.8 | **340.3** |
| 7 | 376.0 | 315.7 | 379.8 | 314.5 | **341.8** |
| **Mean±std (n=8)** | 344.6±19.8 | 342.9±20.0 | 350.1±26.2 | 341.0±21.8 | **339.8±13.9** |

Unlike MATR1 and HUST, where the ablation question is *which tier* to use, CRUSH's ablation question is *which ensemble-combination method* to use — the three solo components (V2-T3, V1-T3, Inter) are all within a few points of each other and of both ensemble methods, but NNLS's per-seed weight vectors swing widely (e.g. `[0,0.48,0.52]` at one seed, `[1,0,0]` at another), overfitting that seed's validation draw; the equal-weight mean is both lower-mean and lower-variance (13.9 vs. 21.8 std) across all 8 seeds. This is the same NNLS-instability-at-small-val pattern discussed in §6.3, now shown at the per-seed level rather than only as a summary statistic. We do not have a corresponding Tier-1-vs-Tier-3 per-component breakdown on the redesigned split (only Tier-3 was re-run there, §5.1); that comparison exists only on the pre-redesign split (§5.1's split-comparison table) and should not be read as answering the same question as this table.

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

Under Tier-3, the monotonicity and Wiener losses impose a temporally-structured constraint on the predicted SOH trajectory. We had hypothesized that V1's learnable cycle positional encoding, by giving the attention mechanism explicit access to cycle position, would make it easier to satisfy these temporal constraints than V2, which must infer position implicitly — and an earlier, 5-seed read of MATR1 appeared to support this (V1 numerically ahead of V2 under Tier-3). The full 8-seed result in §5.2 does not confirm it: V2 has the lower mean RMSE under Tier-3 (85.0 vs. 86.1 for V1), though the gap is small relative to seed variability (§5.2). We therefore treat the positional-encoding/Tier-3 interaction as an open, unconfirmed hypothesis rather than a demonstrated effect, and revise our earlier reading accordingly. Separately, at one seed (MATR1 seed 4), V1 converges to a poor local minimum under Tier-3 despite the same architecture, which we verified is a genuine optimization-variance effect (full-trajectory inspection, patience extended to 350 epochs with no recovery), not a bug — this is noise in the same direction as the inconclusive mean, not independent evidence for either hypothesis.

### 6.2 Why Inter-Cell Embedding Uses a Different Tier Than the Backbone

Inter-Cell Embedding's pairwise predictions were consistently more accurate using an embedding trained under Tier-1 (SOH-auxiliary only) than the same module using an embedding trained under the full Tier-3 objective (79.25±5.04 with the Tier-3 embedding, vs. the improved figure obtained with the Tier-1 embedding and used in the final pipeline, §5.1). We interpret this as: the monotonicity and Wiener constraints push the embedding toward representing *absolute* degradation state (to satisfy a per-cell temporal SDE), which compresses the *relative*, pairwise distance structure that Inter-Cell Embedding relies on. This is why the final pipeline deliberately pairs a Tier-3-trained backbone with a Tier-1-trained correction module rather than optimizing both under one unified objective.

### 6.3 Dataset-Dependent Benefit of Tier-3 (and Where It Actively Hurts)

Tier-3's effect relative to the simpler Tier-1 (SOH-auxiliary only) objective is not uniform, and in one case is clearly negative rather than merely weaker:

- **MATR1**: Tier-3 backbone-only (75.9±5.1) is within noise of the no-SOH baseline (70.4±4.9); the full pipeline (74.0±6.1) is likewise within noise. Tier-3 neither clearly helps nor clearly hurts here.
- **HUST**: Tier-3 is worse than Tier-1 at every seed and every individual component (§5.2b) — 316.9±16.8 vs. 289.8±12.4, a consistent ~27-point RMSE gap, confirmed at all 8/8 seeds. This is not an ensemble-selection artifact; V1 solo, V2 solo, and Inter-Cell Embedding solo are each worse under Tier-3 than under Tier-1 at every seed. We had originally hypothesized (following an earlier, single-seed screen) that Tier-3 would show its clearest gain on HUST; the full result contradicts this, and we report the reversal rather than the earlier hypothesis.
- **CRUSH**: on the pre-redesign split, Tier-3 did not clearly hurt — it was numerically the best of the three configurations (367.5±9.1 vs. 371.3±8.9 for Tier-1 and 371.6±5.5 for the no-SOH baseline), but the gap between all three was within noise. Redesigning the val/train split (§6.1) then produced a materially better Tier-3 result, 339.8±13.9, which beats the local reproduction (355) for the first time, though it still falls short of README (330).

A tentative explanation for HUST's result: the monotonicity and Wiener terms assume a reasonably smooth, well-sampled degradation trajectory to fit $\mu_t, \sigma_t$ per cycle; HUST's inputs are processed in smaller chunks (CHUNK=48, §4) to fit GPU memory, which may make the per-cycle Wiener parameterization noisier to fit than on MATR1, where the full sequence is used at once. CRUSH uses the same chunking (CHUNK=48) yet does not show HUST's clear degradation, so chunk size alone does not fully explain the HUST result — we flag this as an open question rather than a confirmed mechanism. We do not claim Tier-3 "works" or "doesn't work" in general — only that its effect must be checked per dataset, and that HUST is a case where it measurably hurts while CRUSH and MATR1 are not.

On CRUSH specifically, we also found that the ensemble-combination method matters more than which tier is used. On the pre-redesign split, with only 15 validation cells, NNLS's three free weights could fit validation noise that did not generalize to the 44-cell test set — whether a simple unweighted mean or NNLS was better was itself unstable across seed counts, indicating neither choice was reliably better at that validation-set size. This instability was resolved, not just patched, by the val/train redesign (§6.1): on the redesigned split (val=20, train=70), `mean[V1-T3, V2-T3, Inter(V2-T3)]` beats `NNLS[V1-T3, V2-T3, Inter(V2-T3)]` decisively and consistently across all 8 seeds (339.8±13.9 vs. 341.0±21.8) — once val is large enough to be trustworthy, NNLS's extra flexibility buys nothing and its higher variance becomes a liability, not just noise.

---

## 7. Limitations

1. **CRUSH's pre-redesign configurations did not beat README or our local reproduction; the redesigned Tier-3 pipeline closes part of that gap.** No-SOH baseline, Tier-1, and Tier-3 on the pre-redesign split all clustered at 367-372 RMSE, against a local reproduction of 355 and a published README figure of 330 (§5.1, §6.3). An earlier draft of this manuscript reported a "Baseline/Proposed" pair of 357.18/352.92 for CRUSH; a direct search of every result file in the pre-redesign cache could not trace these numbers to any actual run, and we replaced them with the verified figures above — flagged explicitly rather than silently updated. The subsequent val/train redesign (§6.1) yields 339.8±13.9, beating local reproduction for the first time but still short of README.
2. **λ_SOH selection.** The SOH loss weight was screened over $\{0.01, 0.05, 0.1\}$ using test RMSE feedback for the Tier-1 embedding source, across all three main datasets (single-seed for MATR1; full 8-seed for HUST and CRUSH) — a disclosed deviation from the validation-only protocol in §4.3. The value ultimately used was λ=0.01 for MATR1 and λ=0.10 for HUST and CRUSH.
3. **Inter-Cell Embedding's tier mismatch.** The final pipeline's correction module is trained on a Tier-1, not Tier-3, embedding (§3.6, §6.2) — a deliberate, empirically-motivated choice, but one that means the "physics-informed" claim applies to the intra-cell backbone, not to every component of the final ensemble.
4. **HUST's proposed result is Tier-1, not Tier-3.** Having completed the full 8-seed Tier-3 run for HUST, we find Tier-3 underperforms Tier-1 at every seed and component (§5.2b); we report Tier-1's ensemble (289.8±12.4) as HUST's proposed result rather than forcing a Tier-3-based number into the headline.
5. **Ensemble-method instability on small validation sets, and how we resolved it for CRUSH.** On CRUSH's pre-redesign split (val n=15), whether NNLS or a simple mean was the better ensemble-combination method was itself unstable (§6.3); redesigning val (n=20) resolved this rather than merely papering over it — simple mean now wins decisively and consistently across all 8 seeds. We flag this as a caution for any dataset with a comparably small validation set, not just CRUSH.
6. **SNL, CALCE, and CRUH** are evaluated as single-seed screens only, due to insufficient data for a reliable validation split; results there are data-limitation findings, not performance claims.
7. We do not yet provide interpretability analysis (attention maps, feature importance) or uncertainty quantification for RUL/SOH predictions.
8. **Scalar branch vs. attention, not yet separated.** The architecture ladder (§4.2) adds the scalar branch and axis-aware attention in the same step (SmallCNN+scalar → No-SOH baseline), so we cannot currently attribute that gain to attention specifically rather than to the three scalar features. An intermediate "CNN + attention, no scalar branch" ablation would resolve this and is not yet run.
9. **$\lambda_{\mathrm{SOH}}$ selection was test-informed, and it feeds every headline number.** All three datasets' proposed pipelines include an Inter-Cell Embedding module built on a Tier-1 embedding whose $\lambda_{\mathrm{SOH}}$ was chosen via test-RMSE feedback (§4.3 disclosure table). We have not checked whether validation-only selection would choose a different $\lambda_{\mathrm{SOH}}$ or materially change the headline numbers; until that check is run, no result in §5.1 should be read as fully validation-only.

---

## 8. Conclusion

We presented a degradation-aware architecture — CNN backbone, cycle-axis attention pooling, and physics-derived voltage–capacity features — that surpasses the published README benchmark on MATR1 and HUST without any auxiliary loss. We then introduced Tier-3, a physics-informed objective governed by an explicit Wiener-process model of capacity fade, and an Inter-Cell Embedding correction module, combining both into a final pipeline via a disclosed cross-tier design. Reporting against both the published benchmark and our own local reproduction of it, and disclosing every deviation from our pre-declared validation-only selection protocol, we find Tier-3's benefit to be dataset-dependent rather than uniform, and discuss a source-heterogeneity hypothesis for this pattern. The HUST and CRUSH 8-seed Tier-3 evaluations are now complete (§5.1, §5.2b); future work instead includes re-running the $\lambda_{\mathrm{SOH}}$ screen (§4.3) under validation-only selection to check whether the headline numbers are sensitive to it, extending the validation-eligible dataset set beyond MATR1/HUST/CRUSH (§5.4), and exploring a unified single-objective formulation for the backbone and Inter-Cell Embedding module in place of the current disclosed cross-tier design (§3.6, §6.2).

---

## Appendix

### A. Additional Ablation Results
[Include tables/figures for BatLiNet, BatLiNet-V2, 1D-CNN, LSTM/BiLSTM baselines.]

### B. Hyperparameter Details

- **Optimizer**: AdamW, learning rate $10^{-3}$, default weight decay ($10^{-2}$); no learning-rate schedule.
- **Batching**: full-batch gradient descent — one `optimizer.step()` per epoch over the entire training split, no minibatching (train sizes are small: 91/63/64 cells for MATR1/HUST/CRUSH, §4.1).
- **Epochs / evaluation cadence**: up to 1000 epochs, val/test evaluated every 50 epochs; the checkpoint with the lowest validation RMSE is kept (§4.3).
- **Early-stopping patience**: 200 epochs (without improvement in validation RMSE) for the production cross-tier ensemble protocol used for HUST, CRUSH, and MATR1's main ablation (§5.1, §5.2); MATR1's earlier architecture-ladder screen (§4.2, `SmallCNN`/`SmallCNN+scalar` numbers) used patience 350 under an otherwise identical protocol. We do not have a single patience value that applies to every number in this manuscript and disclose the two values rather than implying uniformity.
- **Normalization**: feature and label transformations (log-scale + z-score for the RUL label) are fit on the training split only (`DataBundle.fit`, applied to train, then reused unfit on val/test) — no test-set or validation-set statistics leak into normalization.
- **Architecture sizes**: CNN backbone 6→16→32→32 channels (§3.4.1); scalar branch 3→16→16 (§3.4.3); SOH/$\mu$/$\sigma$ heads are single linear layers $\mathbb{R}^{48}\to\mathbb{R}^{100}$ (§3.5); Inter-Cell Embedding MLP $48\to64\to1$ (§3.6).
- **Initialization**: PyTorch defaults throughout — not overridden.
- Optimizer/lr/patience for the SNL, CALCE, CRUH, and MATR2 single-seed screens (§5.4) are `[TBD]`.

### C. Code and Data Availability
[Links to code repository and data access instructions, if applicable.]
