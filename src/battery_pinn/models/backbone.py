"""Axis-aware CNN backbone (V1=use_pe True / V2=use_pe False) with the Tier-3
heads (SOH auxiliary, monotonicity-drift mu, Wiener-diffusion sigma). Extracted
from scripts/crush/ensemble/run_crush_v1v2inter_tier3_ensemble.py -- the
architecture is identical across CRUSH/HUST/MATR1's Tier-3 scripts; only the
loss WEIGHTS applied outside forward() (lambda_soh, lambda_mono,
lambda_wiener, and CRUSH's asymmetric under_penalty on the RUL term) differ
per dataset, and those stay in the calling script, not here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableCyclePositionalEncoding(nn.Module):
    def __init__(self, num_cycles=50, d_model=32):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, num_cycles, d_model))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    def forward(self, x):
        assert x.ndim == 3 and x.shape[1:] == self.pos.shape[1:]
        return x + self.pos


class CycleAttentionPooling(nn.Module):
    def __init__(self, d_model=32, hidden_dim=16):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        logits = self.score(x)
        weights = torch.softmax(logits, dim=1)
        pooled = (weights * x).sum(dim=1)
        return pooled, weights.squeeze(-1)


class SmallCNNScalarBranchAxisAwareTier3(nn.Module):
    """use_pe=True -> "V1" (learnable cycle positional encoding),
    use_pe=False -> "V2" (no positional encoding)."""

    def __init__(self, n_scalar, use_pe: bool, n_cycles_soh=100):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.pe = LearnableCyclePositionalEncoding(num_cycles=50, d_model=32) if use_pe else None
        self.pool = CycleAttentionPooling(d_model=32, hidden_dim=16)
        self.scalar = nn.Sequential(nn.Linear(n_scalar, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
        self.head = nn.Linear(32 + 16, 1)
        self.soh_head = nn.Linear(32 + 16, n_cycles_soh)
        self.mu_head = nn.Linear(32 + 16, n_cycles_soh)
        self.sigma_head = nn.Linear(32 + 16, n_cycles_soh)

    def forward(self, x, scalar, return_embedding=False, return_tier3=False):
        conv3_out = self.backbone(x)
        assert conv3_out.shape[1] == 32 and conv3_out.shape[2] == 50
        cycle_tokens = conv3_out.mean(dim=-1).transpose(1, 2)
        if self.pe is not None:
            cycle_tokens = self.pe(cycle_tokens)
        signal_embedding, attn = self.pool(cycle_tokens)
        scalar_embedding = self.scalar(scalar)
        h = torch.cat([signal_embedding, scalar_embedding], dim=1)
        out = self.head(h).squeeze(1)
        if return_tier3:
            soh = self.soh_head(h)
            mu = self.mu_head(h)
            sigma = F.softplus(self.sigma_head(h)) + 1e-6
            if return_embedding:
                return out, h, soh, mu, sigma
            return out, soh, mu, sigma
        if return_embedding:
            return out, h
        return out
