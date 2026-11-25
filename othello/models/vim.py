"""Vision-Mamba (Vim) encoder for the 8x8 board: 64 square tokens through
bidirectional S6 selective-scan layers.

Pure torch on purpose — the training box runs cu130/sm_120 (Blackwell)
where mamba-ssm wheels aren't available, and at 64 tokens a short Python
loop over the sequence dim with batched elementwise ops is fast enough.

Contract (same as OthelloRL): forward(x, mask) -> masked-softmax policy
(B,1,8,8), plus (policy, value) when return_val=True; features(x) ->
(trunk map (B,128,8,8), pre-softmax logits) for the MAE decoder — the map
projection is fixed at 128 channels so the MAE decoder is identical across
architectures in the A/B.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from othello.models.base import TokenEncoderBase, scan_orders
from othello.models.blocks import ResidualBlock


class VimBlock(nn.Module):
    """One S6 selective-scan layer: input-dependent dt/B/C, per-channel
    decay A (init in (1, 16]), a gate branch, and a skip connection."""

    def __init__(self, d, d_state=16, expand=2, dt_scale=0.5):
        super().__init__()
        inner = d * expand
        self.d, self.d_state, self.inner = d, d_state, inner
        self.norm = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, inner * 2)          # [scan input u, gate]
        self.dt_proj = nn.Linear(d, inner)
        # dt_bias initialized so that softplus(dt_bias) = dt_scale
        self.dt_bias = nn.Parameter(torch.full((inner,), math.log(math.exp(dt_scale) - 1.0)))
        self.A_log = nn.Parameter(torch.log(torch.rand(inner, d_state) * 15.0 + 1.0))
        self.B_proj = nn.Linear(d, d_state, bias=False)
        self.C_proj = nn.Linear(d, d_state, bias=False)
        self.D = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, d)

    def forward(self, x, order):
        """x: (B, L, d); order: (L,) LongTensor scan permutation."""
        B, L, _ = x.shape
        h = self.norm(x)
        u, gate = self.in_proj(h).chunk(2, dim=-1)      # (B, L, inner) each
        dt = F.softplus(self.dt_proj(h) + self.dt_bias, threshold=20)
        Bm = self.B_proj(h)                              # (B, L, d_state)
        Cm = self.C_proj(h)
        u, gate, dt, Bm, Cm = (t[:, order] for t in (u, gate, dt, Bm, Cm))
        A = torch.exp(self.A_log)                        # (inner, d_state) > 0
        a = torch.exp(-dt.unsqueeze(-1) * A)             # (B, L, inner, d_state)
        b = dt.unsqueeze(-1) * Bm.unsqueeze(2)           # (B, L, inner, d_state)
        state = u.new_zeros(B, self.inner, self.d_state)
        outs = []
        for t in range(L):
            state = a[:, t] * state + b[:, t] * u[:, t].unsqueeze(-1)
            outs.append((Cm[:, t].unsqueeze(1) * state).sum(-1))  # (B, inner)
        y = torch.stack(outs, dim=1) + self.D * u        # skip connection
        y = y * F.silu(gate)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(L, device=order.device)
        return x + self.out_proj(y[:, inv])


class VimEncoder(TokenEncoderBase):
    def __init__(self, model_dim=192, n_layers=8, d_state=16, expand=2,
                 board_size=8, map_channels=128, name="vim", return_val=False):
        super().__init__(name=name, return_val=return_val)
        d = model_dim
        self.model_dim, self.n_layers = d, n_layers
        self.board_size = board_size
        self.trunk_channels = map_channels
        self.embed = nn.Linear(2, d)
        self.pos_embed = nn.Parameter(torch.randn(board_size * board_size, d) * 0.02)
        self.blocks = nn.ModuleList([VimBlock(d, d_state, expand) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d)
        self.policy_head = nn.Linear(d, 1)
        # MAE trunk map: tokens -> (B, map_channels, 8, 8); registered here so
        # the saved state_dict stays strict-loadable
        self.map_proj = nn.Conv1d(d, map_channels, kernel_size=1)
        # shared-trunk value head, same structure as OthelloRL.value_head
        self.value_head = nn.Sequential(
            nn.Conv2d(map_channels + 1, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=64, affine=True),
            nn.LeakyReLU(),
            ResidualBlock(channels=64),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=1),
            nn.Tanh(),
        )
        self.arch_spec = ("vim", model_dim, n_layers, d_state, expand)

    def features(self, x):
        """Trunk map (B, trunk_channels, 8, 8) + pre-softmax logits (B, 1, 8, 8)."""
        B = x.shape[0]
        n = self.board_size ** 2
        toks = self.embed(x.permute(0, 2, 3, 1).reshape(B, n, 2)) + self.pos_embed
        for i, blk in enumerate(self.blocks):
            order = scan_orders(self.board_size, self.board_size, i).to(x.device)
            toks = blk(toks, order)
        toks = self.norm_out(toks)
        logits = self.policy_head(toks).view(B, 1, self.board_size, self.board_size)
        trunk = self.map_proj(toks.transpose(1, 2)).view(
            B, self.trunk_channels, self.board_size, self.board_size)
        return trunk, logits

    def forward(self, x, mask):
        trunk, logits = self.features(x)
        masked = logits * mask + (1 - mask) * -1e32
        policy = masked.view(x.size(0), -1).softmax(dim=1).view(masked.size())
        if self.return_val:
            value = self.value_head(torch.cat((trunk, policy), dim=1))
            return policy, value
        return policy
