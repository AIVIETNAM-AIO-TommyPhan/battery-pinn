"""Tier-3 loss terms (SOH auxiliary + monotonicity + Wiener drift/diffusion),
extracted verbatim from scripts/crush/ensemble/run_crush_v1v2inter_tier3_ensemble.py's
training loop. The asymmetric RUL term's under_penalty defaults to 1.0 (plain
symmetric MSE) -- CRUSH is the only dataset that overrides this (1.4); pass
1.0 explicitly for datasets whose scripts use a symmetric RUL loss to match
their behavior exactly.
"""
import torch
import torch.nn.functional as F


def asymmetric_rul_loss(pred, label, under_penalty=1.0):
    diff = pred - label
    weight = torch.where(diff < 0, under_penalty, 1.0)
    return (weight * diff ** 2).mean()


def soh_auxiliary_loss(soh_pred, soh_true):
    return F.smooth_l1_loss(soh_pred, soh_true)


def monotonicity_loss(soh_pred, tolerance=0.002):
    return F.relu(soh_pred[:, 1:] - soh_pred[:, :-1] - tolerance).mean()


def wiener_loss(soh_pred, mu, sigma):
    """Negative log-likelihood of the capacity-fade increment under a Wiener
    process: D(t) = 1 - SOH(t), dD ~ N(mu, sigma^2) per step."""
    D = 1.0 - soh_pred
    delta_D = D[:, 1:] - D[:, :-1]
    residual = delta_D - mu[:, :-1]
    variance = sigma[:, :-1] ** 2 + 1e-6
    return (0.5 * residual ** 2 / variance + 0.5 * torch.log(variance)).mean()


def tier1_total_loss(pred, label, soh_pred, soh_true, lambda_soh=0.01, under_penalty=1.0):
    """Tier-1: SOH-auxiliary-only supervision -- a physically-named target,
    but NOT physics-informed in the Raissi et al. (2019) sense (no governing
    equation as a soft constraint). Extracted verbatim from
    scripts/crush/ensemble/run_crush_v1v2inter_soh_ensemble.py's training
    loop: `chunk_loss = loss_rul_c + LAMBDA_SOH * loss_soh_c`. This is
    exactly Tier-3's loss with lambda_mono=lambda_wiener=0 -- kept as its
    own function (rather than a Tier-3 call with zeroed weights) because
    that's what the actual scripts do: Tier-1 scripts never compute mu/sigma
    at all (no `soh_head`'s sibling heads in the model), so there is no
    forward-pass cost paid for the terms Tier-1 doesn't use.

    Role in this project: Tier-1 is NOT the primary physics-informed claim
    (Tier-3 is) -- it exists only as (a) the embedding source for
    Inter-Cell Embedding (empirically better there than a Tier-3 embedding,
    see models/inter_embedding.py's docstring) and (b) HUST's actual
    proposed headline, since Tier-3 measurably underperforms Tier-1 there
    at all 8/8 seeds (reports/day_0906/report.md, HUST section)."""
    loss_rul = asymmetric_rul_loss(pred, label, under_penalty)
    loss_soh = soh_auxiliary_loss(soh_pred, soh_true)
    total = loss_rul + lambda_soh * loss_soh
    return total, dict(rul=loss_rul, soh=loss_soh)


def tier3_total_loss(pred, label, soh_pred, soh_true, mu, sigma,
                      lambda_soh=0.01, lambda_mono=0.05, lambda_wiener=0.05,
                      mono_tolerance=0.002, under_penalty=1.0):
    """Tier-3: SOH-auxiliary + monotonicity + Wiener drift/diffusion -- this
    project's primary physics-informed claim, in the Raissi et al. (2019)
    sense (a governing equation, the Wiener SDE `dD(t)=mu*dt+sigma*dW(t)`
    on the capacity-fade increment, enforced as a soft training constraint
    via the Gaussian-increment NLL in wiener_loss()). Pre-declared defaults
    match every dataset's Tier-3 run except under_penalty (CRUSH=1.4,
    everything else=1.0 i.e. symmetric)."""
    loss_rul = asymmetric_rul_loss(pred, label, under_penalty)
    loss_soh = soh_auxiliary_loss(soh_pred, soh_true)
    loss_mono = monotonicity_loss(soh_pred, mono_tolerance)
    loss_wiener = wiener_loss(soh_pred, mu, sigma)
    total = loss_rul + lambda_soh * loss_soh + lambda_mono * loss_mono + lambda_wiener * loss_wiener
    return total, dict(rul=loss_rul, soh=loss_soh, mono=loss_mono, wiener=loss_wiener)
