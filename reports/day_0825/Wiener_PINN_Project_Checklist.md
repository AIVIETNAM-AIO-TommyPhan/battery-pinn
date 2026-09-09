# Wiener/PINN Battery RUL Project Checklist

## How to use this checklist

Use this file to track the project step by step.

Mark an item as complete only when:

- The code runs successfully.
- The result is saved.
- The experiment uses the correct train/validation/test split.
- The relevant metrics are recorded.
- The decision is documented.

Status symbols:

```text
[ ] Not started
[~] In progress
[x] Complete
[-] Skipped or rejected
```

---

## 0. Current project state

- [x] Confirmed the current task is offline full-window prediction.
- [x] Confirmed the input uses approximately cycles 0–99.
- [x] Confirmed the target is total cycle life until EOL.
- [x] Confirmed the target is not remaining cycles after cycle 99.
- [x] Confirmed the cell-level split.
- [x] Confirmed the validated top-3 features.
- [x] Identified the short-life positive RUL bias.
- [x] Identified the full-RMSE versus short-life-safety trade-off.
- [x] Corrected the PINN citation mapping.
- [ ] Record the exact current benchmark values from the saved result files.

Current validated features:

```text
qdlin_diff_std
voltage_slope_50_90
voltage_soc_90
```

Current target definition:

```text
Total battery life until EOL
```

Current data split:

```text
Training:   MATR34 cells
Validation: 41 non-test MATR1 cells
Test:       42 official MATR1 cells
```

---

## 1. Source and citation checklist

- [x] Read the original PINN source.
- [x] Read the battery PINN4SOH paper.
- [x] Read the PINN4SOH repository.
- [x] Read the Zhou SOH paper.
- [x] Separate PINN theory from battery-specific PINN work.
- [x] Separate uncertainty/feature-selection work from PINN work.
- [ ] Add paper title, authors, year, DOI, and URL to the final report.
- [ ] Verify every empirical claim against the original paper.
- [ ] Do not cite search-tool artifacts or file-upload URLs.

### Correct sources

- General PINN framework: [Raissi PINNs repository](https://github.com/maziarraissi/PINNs)
- Battery PINN paper: [Nature Communications](https://www.nature.com/articles/s41467-024-48779-z)
- Battery PINN implementation: [PINN4SOH GitHub](https://github.com/wang-fujin/PINN4SOH)
- PINN4SOH code archive: [Zenodo code](https://zenodo.org/records/11046967)
- PINN4SOH data archive: [Zenodo data](https://zenodo.org/records/10963339)
- Cell-to-cell variation and uncertainty: [arXiv:2312.03097](https://arxiv.org/abs/2312.03097)

---

## 2. Data and label verification

- [x] Confirmed the RUL label counts from the beginning of battery life.
- [x] Confirmed the label corresponds to total life/EOL cycle.
- [x] Confirmed not to subtract 99 from predicted EOL cycle.
- [ ] Save the relevant `rul.py` code excerpt in the experiment report.
- [ ] Record the exact EOL threshold used by BatteryML.
- [ ] Verify the SOH target definition before Tier 1.
- [ ] Verify the first-cycle capacity used for SOH normalization.
- [ ] Verify SOH target alignment with cycles 0–99.
- [ ] Verify missing-cycle handling for SOH labels.
- [ ] Confirm all labels are finite.

Correct current convention:

```python
predicted_total_life = predicted_eol_cycle
```

Do not use:

```python
predicted_total_life = predicted_eol_cycle - 99
```

---

## 3. Evaluation discipline

- [x] Split by complete cell.
- [x] Keep training, validation, and test cells fixed.
- [x] Shuffle only training samples/batches.
- [x] Keep validation and test loaders unshuffled.
- [x] Fit normalization using training cells only.
- [x] Fit calibration using validation predictions only.
- [x] Do not use test labels for model selection.
- [x] Report raw and calibrated results separately.
- [ ] Save the exact split IDs with every experiment.
- [ ] Save the random seed with every experiment.
- [ ] Save the model checkpoint policy with every experiment.
- [ ] Save the feature names and order with every experiment.

Required metrics:

```text
full RMSE
full MAE
full MAPE
full R²
short-life RMSE
short-life MAE
short-life bias
short-life overprediction rate
maximum short-life overprediction
```

Bias definition:

```python
bias = prediction - true_value
```

Positive bias means RUL over-prediction.

---

## 4. Tier 0 — safety loss only

### Objective

Test whether asymmetric RUL loss reduces dangerous short-life over-prediction without causing an unacceptable full-RMSE increase.

### Configuration

- [ ] Use only the validated top-3 features.
- [ ] Keep the existing model architecture unchanged.
- [ ] Do not add a SOH head.
- [ ] Do not add monotonicity loss.
- [ ] Do not add smoothness loss.
- [ ] Do not add a Wiener head.
- [ ] Use seed 0 for screening.

### Loss

```python
error = pred_rul - true_rul

weight = torch.where(
    error > 0,
    safety_weight,
    1.0,
)

loss = (weight * error.square()).mean()
```

### Experiments

- [ ] Safety weight 1.0 — control.
- [ ] Safety weight 1.5.
- [ ] Safety weight 2.0.

### Record results

| Safety weight | Full RMSE | Short RMSE | Short bias | Optimistic rate | Decision |
|---:|---:|---:|---:|---:|---|
| 1.0 | — | — | — | — | — |
| 1.5 | — | — | — | — | — |
| 2.0 | — | — | — | — | — |

### Tier 0 decision

- [ ] Short-life positive bias decreases.
- [ ] Full RMSE remains acceptable.
- [ ] The model does not become excessively conservative.
- [ ] Validation results support the improvement.
- [ ] Select one configuration for five-seed evaluation.
- [ ] Stop the PINN direction if no configuration is useful.

Suggested project gate:

```text
Short-life positive bias decreases by approximately 25–30 cycles or more,
and full RMSE increases by no more than approximately 10–15%.
```

These are project decision rules, not universal scientific standards.

---

## 4.5 Tier 0.5 — cross-cell feature-ordering / ranking loss

Start only after Tier 0 finishes. Not the same as Tier 2 (that is a
*within-cell* temporal monotonicity constraint on a predicted SOH
trajectory; this is a *cross-cell* ordering constraint on the direct
RUL head, no SOH head required — cheaper than Tier 1, but still gated
on its own diagnostic passing first).

### Objective

Check whether a feature's ordering across cells matches true total-life
ordering across cells. If the ordering is reliable, test a small
pairwise ranking loss that penalizes the RUL head for reversing it.

### Step 1 — ranking diagnostics (train cells only, no new training)

Before writing any loss code, compute on the training split only:

- [ ] Spearman correlation (feature vs. true total life).
- [ ] Kendall correlation (feature vs. true total life).
- [ ] Pairwise ordering accuracy (fraction of cell pairs where the
      feature's order agrees with the true total-life order).

Report all three broken out by:

- [ ] all cells
- [ ] short-life cells
- [ ] mid-life cells
- [ ] long-life cells

`qdlin_diff_l2` is the first candidate feature to check (chosen for its
Tier-0 under-predict-penalty result, task #18).

**Step 1 results (train cells, n=91), best-direction pairwise accuracy:**

| Feature | all | short | mid | long |
|---|---:|---:|---:|---:|
| `qdlin_diff_std` | 76.9% | 76.7% | 57.9% (n.s.) | 78.2% |
| `voltage_slope_50_90` | 74.8% | 68.9% | 53.5% (n.s.) | 67.0% |
| `voltage_soc_90` | 74.7% | 67.3% | 55.8% (n.s.) | 66.1% |
| `qdlin_diff_min` | 70.1% | 78.1% | 60.7% (n.s.) | 66.5% |
| `qdlin_diff_l2` | 69.8% | 76.0% | 58.6% (n.s.) | 68.5% |

Also checked: the already-trained models' own TEST predictions (weight=1.5)
already order much better than any raw feature — `top3`=88.4%, `qdlin_diff_l2`
=86.4%, `qdlin_diff_min`=82.2% overall, and the mid-life weak spot shrinks a
lot at the prediction level (58-61% raw → 70-77% predicted). **Verdict:
promising but not strong enough for a blind ranking loss** — moderate,
real signal (>random 50%, esp. short/long), but the model already captures
most of it, so expect only a small incremental gain, not a large one. Every
feature shares the same weak point: mid-life pairs are consistently the
least reliable (never statistically significant) — don't force the model to
obey those.

### Step 2 — small ranking-loss test (only if Step 1 looks stable)

Per reviewer guidance: run this as a **small diagnostic on the existing
`top3` model, no architecture change** — not a main upgrade attempt. Exclude
or down-weight the noisy mid-life pairs; use only short-life and long-life
pairs (or a confidence weight per pair) so the loss isn't fighting near-random
mid-life signal.

```python
# pairwise margin loss: for cell pairs i, j with true_rul_i > true_rul_j,
# penalize the model for predicting pred_rul_i < pred_rul_j.
# pair_weight down-weights/excludes low-confidence (mid-life) pairs --
# e.g. 1.0 for a short-short or long-long pair, 0.0 (or a small constant)
# for any pair touching the mid-life tercile.
margin = 0.0
diff_true = true_rul.unsqueeze(1) - true_rul.unsqueeze(0)
diff_pred = pred_rul.unsqueeze(1) - pred_rul.unsqueeze(0)
sign_true = torch.sign(diff_true)
loss_rank = (pair_weight * torch.relu(margin - sign_true * diff_pred)).mean()

loss_total = loss_rul + lambda_rank * loss_rank
```

Test, one seed first:

- [ ] `lambda_rank = 0` (control).
- [ ] `lambda_rank = 0.001`.
- [ ] `lambda_rank = 0.005`.
- [ ] `lambda_rank = 0.01`.

### Tier 0.5 decision

- [ ] Ranking diagnostics (Step 1) show a reliable ordering — if not,
      stop here, do not write the ranking-loss code.
- [ ] Full RMSE remains acceptable at the tested `lambda_rank` values.
- [ ] Short-life RMSE/bias does not worsen (short/long confident pairs are
      the ones being trusted — don't damage them to chase mid-life).
- [ ] Long-life RMSE or bias improves — this is the actual bar for keeping
      the ranking loss, not just "ordering accuracy went up."
- [ ] Stop immediately if full RMSE or short-life metrics worsen.
- [ ] Promote to five-seed evaluation only if a `lambda_rank` shows
      improvement on the agreed validation criteria — same discipline
      as every other tier. Do not automatically run five seeds just because
      a screen looks positive.

---

## 5. Tier 1 — auxiliary SOH prediction

Start only if Tier 0 passes.

### Objective

Test whether predicting the SOH trajectory helps total-life prediction.

### Architecture

```text
top-3 features
→ existing representation
→ direct total-life head
→ auxiliary SOH trajectory head
```

### Before coding

- [ ] Confirm SOH target formula.
- [ ] Confirm SOH normalization.
- [ ] Confirm cycle indexing.
- [ ] Confirm target shape `[batch, 100]`.
- [ ] Confirm target/input alignment.
- [ ] Confirm BatteryML total-life label remains the direct RUL target.

### Loss

```python
loss_rul = F.smooth_l1_loss(rul_pred, rul_true)
loss_soh = F.smooth_l1_loss(soh_pred, soh_true)
loss_total = loss_rul + lambda_soh * loss_soh
```

Test:

- [ ] `lambda_soh = 0.01`.
- [ ] `lambda_soh = 0.05`.
- [ ] `lambda_soh = 0.10`.

Do not add monotonicity at this tier.

### Tier 1 decision

- [ ] SOH head improves or preserves full RMSE.
- [ ] SOH head improves or preserves short-life RMSE.
- [ ] SOH head improves or preserves short-life bias.
- [ ] SOH trajectories are numerically valid.
- [ ] Continue to Tier 2 only if evidence is positive.

---

## 6. Tier 2 — monotonicity constraint

Start only if Tier 1 passes.

### Objective

Prevent predicted SOH from increasing unrealistically over cycle number.

### Loss

```python
tolerance = 0.002

loss_mono = torch.relu(
    soh_pred[:, 1:]
    - soh_pred[:, :-1]
    - tolerance
).mean()
```

Test:

- [ ] `lambda_mono = 0.001`.
- [ ] `lambda_mono = 0.01`.
- [ ] `lambda_mono = 0.05`.

### Tier 2 decision

- [ ] Unreasonable SOH increases decrease.
- [ ] Real knee behavior is not removed.
- [ ] Full RMSE remains acceptable.
- [ ] Short-life positive bias does not worsen.
- [ ] Continue to Wiener modeling only if useful.

---

## 7. Optional smoothness loss

Do not add this automatically.

Start only if trajectory plots show artificial oscillations.

```python
second_diff = (
    soh_pred[:, 2:]
    - 2.0 * soh_pred[:, 1:-1]
    + soh_pred[:, :-2]
)

loss_smooth = second_diff.square().mean()
```

- [ ] Inspect SOH trajectory plots.
- [ ] Confirm oscillation is a model artifact.
- [ ] Test a small smoothness weight.
- [ ] Check that knee behavior remains visible.
- [ ] Keep or reject based on validation evidence.

---

## 8. Tier 3 — Wiener-informed neural network

Start only after the earlier tiers provide evidence.

### Objective

Model degradation with a mean drift and stochastic diffusion:

\[
dD(t)=\mu(t,x)dt+\sigma(t,x)dW(t)
\]

where:

```text
D     = degradation state
mu    = expected degradation rate
sigma = uncertainty/diffusion
W     = Wiener process
```

Use:

```python
D = 1.0 - SOH
```

### Architecture

```text
top-3 features
→ shared representation
→ RUL head
→ drift head μ
→ diffusion head σ
```

Ensure positive diffusion:

```python
sigma = F.softplus(raw_sigma) + 1e-6
```

### Discrete residual

```python
delta_D = D[:, 1:] - D[:, :-1]
residual = delta_D - mu[:, :-1] * delta_t
```

### Gaussian Wiener loss

```python
variance = sigma[:, :-1].square() * delta_t + 1e-6

loss_wiener = (
    0.5 * residual.square() / variance
    + 0.5 * torch.log(variance)
).mean()
```

### Total loss

```python
loss_total = (
    loss_rul
    + lambda_wiener * loss_wiener
    + lambda_safety * loss_safety
)
```

Test:

- [ ] `lambda_wiener = 0.001`.
- [ ] `lambda_wiener = 0.01`.
- [ ] `lambda_wiener = 0.05`.

### Wiener output evaluation

- [ ] Report mean predicted total life.
- [ ] Report median predicted total life.
- [ ] Report 5th percentile.
- [ ] Report 25th percentile.
- [ ] Report 75th percentile.
- [ ] Report 95th percentile.
- [ ] Check interval coverage.
- [ ] Check interval width.
- [ ] Check short-life optimistic error.

A conservative prediction may use the 25th percentile, but this must be validated:

```python
safe_prediction = np.percentile(rul_samples, 25)
```

---

## 9. Ensemble integration

Current roles:

```text
top3             → best global accuracy
qdlin_diff_min   → strongest short-life behavior
qdlin_diff_l2    → balanced behavior
```

- [ ] Generate validation predictions for all candidate models.
- [ ] Fit nonnegative ensemble weights using validation data only.
- [ ] Enforce weights sum to one.
- [ ] Freeze the weights before test evaluation.
- [ ] Compare equal averaging versus weighted averaging.
- [ ] Check whether the ensemble reduces short-life positive bias.

Constraint option:

```text
Minimize validation full RMSE
subject to short-life bias <= +20 cycles
```

The +20-cycle limit is a project-defined constraint.

---

## 10. Calibration

- [ ] Generate raw validation predictions.
- [ ] Fit isotonic or another calibration method on validation only.
- [ ] Freeze calibration.
- [ ] Apply it to test predictions.
- [ ] Report raw results.
- [ ] Report calibrated results separately.
- [ ] Check calibration by short/middle/long-life regime.
- [ ] Check whether calibration worsens short-life bias.

Correct order:

```text
train
→ validation prediction
→ fit calibration
→ freeze calibration
→ test evaluation
```

---

## 11. Multi-seed commitment

Use seed 0 for screening.

Run five seeds only for the best surviving configuration:

```python
SEEDS = [0, 1, 2, 3, 4]
```

- [ ] Save seed 0 result.
- [ ] Select candidate using validation criteria.
- [ ] Run seeds 0–4.
- [ ] Report mean ± standard deviation.
- [ ] Report per-cell seed standard deviation.
- [ ] Identify high-instability cells.

Multi-seed measures training variability. It is not the same as data shuffling.

---

## 12. Final comparison table

| Model | Full RMSE | Short RMSE | Short bias | Optimistic rate | Seed std | Status |
|---|---:|---:|---:|---:|---:|---|
| Current top-3 | — | — | — | — | — | Baseline |
| Top-3 + safety loss | — | — | — | — | — | Tier 0 |
| + cross-cell ranking loss | — | — | — | — | — | Tier 0.5 |
| Top-3 + SOH head | — | — | — | — | — | Tier 1 |
| + monotonicity | — | — | — | — | — | Tier 2 |
| + Wiener model | — | — | — | — | — | Tier 3 |

## 13. Final paper claims checklist

Before writing manuscript claims:

- [ ] Every benchmark value comes from a saved result file.
- [ ] Every empirical claim has a matching citation or project result.
- [ ] Source links match the source titles.
- [ ] Raissi is cited for general PINN theory.
- [ ] Wang et al. is cited for PINN4SOH.
- [ ] Zhou et al. is cited for cell variation/uncertainty, not PINN theory.
- [ ] Wiener-process sources are cited for stochastic degradation modeling.
- [ ] The total-life label convention is clearly stated.
- [ ] Short-life safety metrics are reported.
- [ ] Raw and calibrated results are separated.
- [ ] Test data is not used for selection or calibration.
- [ ] Limitations are reported.

## 14. Current recommended action

The immediate action is:

```text
Finish Tier 0.
```

Run:

```text
Existing top-3 model
+ safety weight 1.0
+ safety weight 1.5
+ safety weight 2.0
+ seed 0
```

Compare full RMSE and short-life safety. Do not implement the SOH trajectory or Wiener model until Tier 0 has been evaluated.

## 15. Final research direction

The mature model may become:

```text
validated top-3 representation
→ safety-aware RUL prediction
→ optional SOH auxiliary task
→ optional monotonicity constraint
→ Wiener drift/diffusion
→ uncertainty-aware total-life prediction
```

The main success criterion is:

```text
Reduce dangerous short-life over-prediction
while preserving acceptable full-set accuracy.
```
