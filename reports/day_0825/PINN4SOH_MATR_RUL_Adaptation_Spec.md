# PINN4SOH-Style MATR RUL Adaptation Specification

## 1. Purpose

This document specifies a physics-informed extension of the current BatteryML/MATR RUL pipeline, inspired by PINN4SOH.

The objective is to keep the validated 29-feature and BatLiNet inputs unchanged while adding:

- SOH trajectory prediction.
- Monotonic degradation constraints.
- Safety-aware RUL loss.
- RUL estimation from the predicted EOL crossing.
- Comparison with the existing direct-RUL model.

The reference PINN4SOH work focuses primarily on stable SOH estimation and degradation prognosis rather than direct RUL regression. Therefore, this document adapts the idea to the current MATR full-window setting instead of copying the original implementation directly.

## 2. Reference sources

- Wang et al., "Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis," Nature Communications, 2024. [https://www.nature.com/articles/s41467-024-48779-z](https://www.nature.com/articles/s41467-024-48779-z)
- PINN4SOH source code: [https://github.com/wang-fujin/PINN4SOH](https://github.com/wang-fujin/PINN4SOH)
- PINN4SOH data archive: [https://zenodo.org/records/10963339](https://zenodo.org/records/10963339)
- PINN4SOH code archive: [https://zenodo.org/records/11046967](https://zenodo.org/records/11046967)
- PINN thermal-management review: [https://www.preprints.org/manuscript/202505.0088](https://www.preprints.org/manuscript/202505.0088)
- BatteryML: [https://github.com/microsoft/BatteryML](https://github.com/microsoft/BatteryML)

The PINN4SOH paper models SOH degradation with a data-fitting term and physics-informed degradation constraints. The repository provides the implementation and associated data. [web:159][web:160][web:161][web:163][web:164]

## 3. Current project framing

### 3.1 Dataset split

Keep the existing cell-level split fixed:

```text
Training:   MATR34 cells
Validation: 41 non-test MATR1 cells
Test:       42 official MATR1 cells
```

The test set must not be used for:

- Loss-weight selection.
- Model selection.
- Ensemble-weight fitting.
- Calibration fitting.
- Threshold selection.

### 3.2 Prediction framing

The current task is an offline full-window task:

```text
Input: first 100 cycles
Output: battery RUL
```

This is not an online cycle-by-cycle prediction task. Features aggregated over cycles 0–99 are allowed under this framing.

### 3.3 Fixed inputs

Do not modify the validated feature definitions in the first PINN experiment.

Inputs remain:

```text
BatLiNet feature tensor:
    [batch, 6, 100, 1000]

Scalar features:
    [batch, 29]
```

The six BatLiNet channels are:

```text
charge voltage
 discharge voltage
charge current
discharge current
delta voltage
R(Q) proxy
```

The scalar branch contains the validated 29-feature set.

## 4. Model objectives

Compare two main approaches.

### 4.1 Direct-RUL baseline

```text
BatLiNet tensor + scalar features
→ CNN branch + scalar branch
→ fused representation
→ direct RUL prediction
```

### 4.2 PINN-inspired SOH trajectory model

```text
BatLiNet tensor + scalar features
→ CNN branch + scalar branch
→ fused representation
→ SOH trajectory for cycles 0–99
→ EOL crossing
→ RUL estimate
```

The second model is physics-informed through its loss functions. It should not be called a full electrochemical PINN unless an explicit electrochemical governing equation is included.

## 5. Targets

### 5.1 SOH target

Use normalized discharge capacity:

```python
soh_k = Q_discharge_k / Q_initial
```

or, if nominal capacity is the project convention:

```python
soh_k = Q_discharge_k / Q_nominal
```

Use one definition consistently across training, validation, and test.

Recommended initial definition:

```python
soh_k = Q_discharge_k / Q_initial
```

with optional clipping:

```python
soh_k = np.clip(soh_k, 0.0, 1.2)
```

Do not silently change the target definition between experiments.

### 5.2 RUL target

Use:

```python
rul = eol_cycle - observation_end_cycle
```

For the first-100-cycle full-window experiment:

```python
observation_end_cycle = 99
```

If the label in the existing BatteryML pipeline represents total life rather than post-observation RUL, retain the existing convention and document it explicitly.

### 5.3 EOL threshold

Use:

```python
SOH_EOL = 0.80
```

For noisy capacity trajectories, define EOL as the first cycle for which SOH remains below the threshold for several consecutive cycles.

Example:

```python
EOL_CONSECUTIVE_CYCLES = 3
```

Keep the current BatteryML label convention as the primary benchmark target if it differs from this definition. Use trajectory-derived RUL as a secondary experiment until the definitions are confirmed equivalent.

## 6. Architecture specification

### 6.1 CNN branch

Reuse the current small CNN branch to preserve comparability:

```python
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        return self.features(x).flatten(1)
```

### 6.2 Scalar branch

Use the validated scalar branch:

```python
class ScalarBranch(nn.Module):
    def __init__(self, n_scalar=29):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_scalar, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
```

### 6.3 SOH trajectory head

The first implementation may use a direct 100-cycle output head:

```python
class PINNSOHModel(nn.Module):
    def __init__(self, n_scalar=29, n_cycles=100):
        super().__init__()
        self.cnn = SmallCNN()
        self.scalar = ScalarBranch(n_scalar)
        self.fusion = nn.Sequential(
            nn.Linear(32 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.soh_head = nn.Linear(64, n_cycles)
        self.rul_head = nn.Linear(64, 1)

    def forward(self, x_curve, x_scalar):
        z_curve = self.cnn(x_curve)
        z_scalar = self.scalar(x_scalar)
        z = self.fusion(torch.cat([z_curve, z_scalar], dim=1))
        soh = self.soh_head(z)
        rul = self.rul_head(z).squeeze(1)
        return soh, rul
```

The direct 100-value SOH head is a practical baseline. A later version can use a temporal decoder or BiLSTM if the direct head cannot represent smooth trajectories.

## 7. Output constraints

### 7.1 SOH range

Use either output clipping:

```python
soh_pred = torch.clamp(soh_raw, 0.0, 1.2)
```

or a bounded activation:

```python
soh_pred = 1.2 * torch.sigmoid(soh_raw)
```

A sigmoid can make optimization difficult near the boundaries. Start with an unconstrained output plus a soft range penalty.

### 7.2 RUL range

Use:

```python
rul_pred = torch.relu(rul_raw)
```

or leave the output unconstrained and clip only during evaluation. Do not use test statistics to set the maximum RUL.

## 8. Loss specification

### 8.1 SOH data loss

```python
loss_soh = F.smooth_l1_loss(soh_pred, soh_true)
```

Smooth L1 is recommended initially because measured capacity can contain noise and occasional anomalies.

### 8.2 RUL loss

```python
loss_rul = F.smooth_l1_loss(rul_pred, rul_true)
```

### 8.3 Monotonicity loss

SOH should generally not increase as the battery ages:

```python
loss_mono_soh = torch.relu(
    soh_pred[:, 1:] - soh_pred[:, :-1] - tolerance
).mean()
```

Use:

```python
tolerance = 0.002
```

The tolerance prevents the model from being punished for tiny measurement fluctuations.

### 8.4 Smoothness loss

Use a second-difference penalty to discourage unrealistic oscillations:

```python
second_diff = (
    soh_pred[:, 2:]
    - 2.0 * soh_pred[:, 1:-1]
    + soh_pred[:, :-2]
)

loss_smooth = second_diff.square().mean()
```

Do not make this loss too strong because real battery degradation can contain knee points and changing slopes.

### 8.5 Range loss

```python
loss_range = (
    torch.relu(-soh_pred).square().mean()
    + torch.relu(soh_pred - 1.2).square().mean()
)
```

### 8.6 Safety loss

Positive RUL error means over-predicting remaining life:

```python
rul_error = rul_pred - rul_true
loss_safety = torch.relu(rul_error).square().mean()
```

This term should be tested carefully because it applies to all samples. A stronger short-life-aware version requires a short-life mask.

```python
short_mask = rul_true <= 499
positive_error = torch.relu(rul_pred - rul_true)

if short_mask.any():
    loss_safety_short = (
        positive_error[short_mask].square().mean()
    )
else:
    loss_safety_short = torch.zeros(
        (), device=rul_pred.device
    )
```

### 8.7 Total loss

Initial formulation:

```python
loss_total = (
    loss_soh
    + lambda_rul * loss_rul
    + lambda_mono * loss_mono_soh
    + lambda_smooth * loss_smooth
    + lambda_range * loss_range
    + lambda_safety * loss_safety_short
)
```

Initial values:

```text
lambda_rul    = 0.1
lambda_mono   = 0.01
lambda_smooth = 0.001
lambda_range  = 0.001
lambda_safety = 0.1
```

These are starting values only. Select them using validation data.

## 9. RUL from predicted SOH

If the model predicts SOH through cycle 99 only, it may not reach the EOL threshold inside the observed window. Therefore, use one of two approaches.

### 9.1 Direct RUL head

Use the direct RUL output for the main benchmark:

```python
rul_pred = model(...)[1]
```

The SOH trajectory acts as an auxiliary physics-informed task.

### 9.2 SOH trajectory extrapolation

For a true trajectory-derived RUL estimate:

1. Predict SOH for observed cycles.
2. Fit a degradation trend to the final part of the predicted trajectory.
3. Extrapolate until SOH reaches 0.80.
4. Calculate EOL cycle.
5. Subtract the observation endpoint.

Example conceptual function:

```python
def estimate_eol_from_soh(soh_pred, start_cycle=0, threshold=0.80):
    below = np.where(soh_pred <= threshold)[0]
    if len(below):
        return int(start_cycle + below[0])

    # Extrapolation is required if threshold is not observed.
    # Fit only the final degradation segment.
    return extrapolated_eol
```

Do not use trajectory-derived RUL as the only output until its label convention has been verified against BatteryML.

## 10. Training protocol

### 10.1 Fixed split

```text
Train: MATR34
Validation: MATR1 non-test
Test: official MATR1 test
```

### 10.2 Batch shuffling

Shuffle only training samples:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
)
```

Do not shuffle cycles within a battery and do not mix battery cells between train, validation, and test. Group-aware cell splits are required to prevent battery-level leakage. [web:85][web:86]

### 10.3 Multi-seed evaluation

Use:

```python
SEEDS = [0, 1, 2, 3, 4]
```

Multi-seed evaluation measures sensitivity to initialization and training batch order. It is not the same as shuffling, although `shuffle=True` can make the batch order seed-dependent.

### 10.4 Checkpoint selection

Because pooled validation RMSE may not match the test objective, record both:

```text
best validation checkpoint
fixed-epoch checkpoint
```

Use the same checkpoint policy for every model comparison. Do not select the policy after viewing test results.

## 11. Evaluation metrics

Report overall and short-life metrics:

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
seed standard deviation
```

Define bias as:

```python
bias = prediction - true_value
```

Therefore:

```text
positive bias = RUL over-prediction
negative bias = RUL under-prediction
```

Safety metric:

```python
optimistic_error = np.maximum(pred - true, 0)
```

Report:

```python
mean_optimistic_error
max_optimistic_error
optimistic_error_rate
short_life_optimistic_error
```

## 12. Required ablation experiments

Run one seed first.

### A. Existing baseline

```text
Current direct-RUL model
```

### B. Auxiliary SOH task

```text
Direct RUL + SOH trajectory prediction
```

### C. Add monotonicity

```text
B + SOH monotonicity loss
```

### D. Add smoothness

```text
C + SOH smoothness loss
```

### E. Add safety loss

```text
D + short-life asymmetric RUL loss
```

### F. Five-seed final comparison

Run only for the best candidates.

Table format:

| Model | Full RMSE | Short RMSE | Short bias | Optimistic rate | Seed std |
|---|---:|---:|---:|---:|---:|
| Direct-RUL baseline | — | — | — | — | — |
| Auxiliary SOH | — | — | — | — | — |
| + monotonicity | — | — | — | — | — |
| + smoothness | — | — | — | — | — |
| + safety loss | — | — | — | — | — |

## 13. Trade-off analysis

Do not select a model using full RMSE alone.

Use a constrained rule:

```text
Select the lowest full-RMSE model subject to:
short-life bias <= +20 cycles
```

Also plot:

```text
x-axis: full RMSE
y-axis: short-life RMSE
```

and:

```text
x-axis: full RMSE
y-axis: short-life bias
```

The current interpretation is:

```text
top3             → best global accuracy, unsafe short-life optimism
qdlin_diff_min   → strongest short-life behavior
qdlin_diff_l2   → balanced behavior
```

The PINN-inspired model should be considered successful if it reduces short-life positive bias without a large increase in full RMSE.

## 14. Implementation checks

Before training:

```python
assert train_base_scalar.shape[1] == 29
assert val_scalar.shape[1] == 29
assert test_scalar.shape[1] == 29

assert torch.isfinite(train_base_scalar).all()
assert torch.isfinite(val_scalar).all()
assert torch.isfinite(test_scalar).all()

assert train_base_feature.ndim == 4
assert val_feature.ndim == 4
assert test_feature.ndim == 4

assert train_base_feature.shape[1:] == val_feature.shape[1:]
assert train_base_feature.shape[1:] == test_feature.shape[1:]
```

After model creation:

```python
soh_pred, rul_pred = model(
    train_base_feature[:2].to(device),
    train_base_scalar[:2].to(device),
)

assert soh_pred.shape == (2, 100)
assert rul_pred.shape == (2,)
```

Check that the SOH target has the same shape:

```python
assert soh_true.shape == soh_pred.shape
```

## 15. Expected risks

### 15.1 Incorrect target alignment

The input contains the first 100 cycles, but the SOH trajectory must be aligned exactly with those same cycles.

### 15.2 Overly strong monotonicity

A high monotonicity weight can make the trajectory artificially smooth and hide real knee behavior.

### 15.3 Safety-loss bias

A strong asymmetric loss can reduce optimistic errors but increase conservative under-prediction.

### 15.4 RUL and SOH inconsistency

If the direct RUL head and SOH trajectory imply different EOL times, report both and investigate the inconsistency.

### 15.5 Domain shift

PINN constraints may improve physical plausibility but cannot create training examples for unseen chemistries, protocols, or extreme life ranges.

### 15.6 Dataset quality

Corrupted cells should be evaluated both with and without the repair strategy. Median-fill is an established project decision, but it should be documented as a data-repair method rather than a physical model component.

## 16. Recommended development order

1. Keep the validated 29 features unchanged.
2. Implement SOH target construction.
3. Implement the auxiliary SOH head.
4. Verify tensor shapes and target alignment.
5. Run the direct-RUL baseline with one seed.
6. Run the auxiliary SOH model with data loss only.
7. Add monotonicity loss.
8. Add smoothness loss if needed.
9. Add short-life safety loss.
10. Compare full RMSE and short-life bias.
11. Run five seeds only for promising configurations.
12. Apply calibration after model selection, using validation data only.

## 17. Success criteria

The PINN-inspired approach is considered useful if it satisfies all of the following:

```text
1. It runs without target or shape errors.
2. It does not use test data during training or calibration.
3. It improves or preserves full RMSE.
4. It reduces short-life positive bias.
5. It does not increase optimistic-error rate substantially.
6. It produces a physically reasonable SOH trajectory.
7. Its results are stable across multiple seeds.
```

The primary success criterion should be:

```text
Lower short-life optimistic bias with an acceptable full-RMSE trade-off.
```

## 18. Final recommended experiment

Use the following first:

```text
Input:
    existing BatLiNet tensor
    existing 29 scalar features

Model:
    SmallCNN + scalar branch + fusion
    direct RUL head
    auxiliary 100-cycle SOH head

Loss:
    SmoothL1 RUL loss
    SmoothL1 SOH loss
    small SOH monotonicity penalty
    small short-life asymmetric penalty

Training:
    MATR34 cells
    batch shuffle=True
    fixed validation/test cells
    one seed first

Evaluation:
    full RMSE
    short-life RMSE
    short-life bias
    optimistic-error rate
```

The recommended first model is therefore not a full electrochemical PINN. It is a **PINN-inspired, multi-task, safety-aware RUL model** based on the PINN4SOH concept and adapted to the existing BatteryML/MATR pipeline.
