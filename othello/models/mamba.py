"""Mamba encoder for the 8x8 board: 64 square tokens through Mamba-style
selective-S6 blocks — causal depthwise conv, input-dependent per-token
dt/B/C, per-channel decay A, gate branch, skip connection. Pure torch (the
box is cu130/sm_120 where mamba-ssm wheels aren't available); the 64-step
scan is a short Python loop over the sequence dim. Sized to match the
ConvNeXt A/B arm (~13M params): d=896 x 2 blocks x d_state=16. (Block
params scale ~8d^2 — in_proj is d->4d — so 896, not 1280, is the ~13.5M
width for two blocks.)

The scan's exp() materialization dominates forward time in eager mode; on
the training box pass --compile-blocks so inductor fuses the scan into one
kernel.

Contract (same as OthelloRL): forward(x, mask) -> masked-softmax policy
(B,1,8,8), plus (policy, value) when return_val=True; features(x) ->
(trunk map (B,128,8,8), pre-softmax logits) for the MAE decoder.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from othello.models.base import TokenEncoderBase


class MambaBlock(nn.Module):
    """One selective state-space layer: in_proj (u | gate), causal
    depthwise conv on u, S6 scan with per-token dt/B/C, output gate."""

    def __init__(self, d, d_state=16, expand=2, dt_scale=0.5):
        super().__init__()
        inner = d * expand
        self.d, self.d_state, self.inner = d, d_state, inner
        self.norm = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, inner * 2)          # [scan input u, gate]
        self.conv1d = nn.Conv1d(inner, inner, kernel_size=4, groups=inner,
                                padding=3, bias=True)   # causal (trim to L after)
        self.dt_proj = nn.Linear(d, inner)
        # dt_bias initialized so that softplus(dt_bias) = dt_scale
        self.dt_bias = nn.Parameter(torch.full((inner,), math.log(math.exp(dt_scale) - 1.0)))
        self.A_log = nn.Parameter(torch.log(torch.rand(inner, d_state) * 15.0 + 1.0))
        self.B_proj = nn.Linear(d, d_state, bias=False)  # shared across inner channels
        self.C_proj = nn.Linear(d, d_state, bias=False)
        self.D = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, d)

    def forward(self, x):
        """x: (B, L, d)."""
        B, L, _ = x.shape
        h = self.norm(x)
        u, gate = self.in_proj(h).chunk(2, dim=-1)      # (B, L, inner) each
        u = F.silu(self.conv1d(u.transpose(1, 2)).transpose(1, 2)[:, :L])
        dt = F.softplus(self.dt_proj(h) + self.dt_bias, threshold=20)
        Bm = self.B_proj(h)                              # (B, L, d_state)
        Cm = self.C_proj(h)
        A = torch.exp(self.A_log)                        # (inner, d_state) > 0
        a = torch.exp(-dt.unsqueeze(-1) * A)             # (B, L, inner, d_state)
        b = dt.unsqueeze(-1) * Bm.unsqueeze(2)           # (B, L, inner, d_state)
        state = u.new_zeros(B, self.inner, self.d_state)
        outs = []
        for t in range(L):
            state = a[:, t] * state + b[:, t] * u[:, t].unsqueeze(-1)
            outs.append((Cm[:, t].unsqueeze(1) * state).sum(-1))  # (B, inner)
        y = torch.stack(outs, dim=1) + self.D * u        # (B, L, inner)
        return x + self.out_proj(y * F.silu(gate))


class MambaEncoder(TokenEncoderBase):
    def __init__(self, model_dim=896, n_layers=2, d_state=16, expand=2,
                 board_size=8, map_channels=128, name="mamba", return_val=False):
        super().__init__(model_dim, board_size=board_size, map_channels=map_channels,
                         name=name, return_val=return_val)
        self.n_layers, self.d_state, self.expand = n_layers, d_state, expand
        self.blocks = nn.ModuleList([MambaBlock(model_dim, d_state, expand)
                                     for _ in range(n_layers)])
        self.arch_spec = ("mamba", model_dim, n_layers, d_state, expand)

    def features(self, x):
        toks = self.tokenize(x)
        for blk in self.blocks:
            toks = blk(toks)
        return self.heads_from(toks)
