"""Vision-LSTM (xLSTM / ViL) encoder for the 8x8 board: 64 square tokens
through mLSTM blocks (matrix memory, exponential gates), alternating
row/column forward/backward scan orders per ViL.

Pure torch, no external deps. Numerical pins from the xLSTM paper's stable
form: gates in log space, exp inputs clamped to [-20, 20], forget gate via
exp(-softplus(f)) with init bias 2.0, and the memory normalized by
max(|n^T q|, 1).

Contract (same as OthelloRL): forward(x, mask) -> masked-softmax policy
(B,1,8,8), plus (policy, value) when return_val=True; features(x) ->
(trunk map (B,128,8,8), pre-softmax logits) for the MAE decoder.
"""
import torch
import torch.nn.functional as F
from torch import nn

from othello.models.base import TokenEncoderBase, scan_orders
from othello.models.blocks import ResidualBlock


class MLSTMBlock(nn.Module):
    """One mLSTM layer: per-head scalar input/forget gates, matrix memory
    C (H, dh, dh), output gate, skip connection."""

    def __init__(self, d, heads=4, forget_bias=2.0):
        super().__init__()
        assert d % heads == 0
        self.d, self.heads, self.dh = d, heads, d // heads
        self.norm = nn.LayerNorm(d)
        self.qkv_proj = nn.Linear(d, 3 * d)
        self.ig_proj = nn.Linear(d, heads)              # input gate, log space
        self.fg_proj = nn.Linear(d, heads)              # forget gate, pre-softplus
        self.og_proj = nn.Linear(d, d)                  # output gate (sigmoid)
        self.out_proj = nn.Linear(d, d)
        self.forget_bias = nn.Parameter(torch.full((heads,), float(forget_bias)))

    def forward(self, x, order):
        """x: (B, L, d); order: (L,) LongTensor scan permutation."""
        B, L, _ = x.shape
        h = self.norm(x)
        q, k, v = self.qkv_proj(h).view(B, L, 3, self.heads, self.dh).unbind(2)
        i_t = torch.exp(self.ig_proj(h).view(B, L, self.heads, 1).clamp(-20.0, 20.0))
        f_t = torch.exp(-F.softplus(
            (self.fg_proj(h) + self.forget_bias).view(B, L, self.heads, 1)
            .clamp(-20.0, 20.0), threshold=20.0))
        o = torch.sigmoid(self.og_proj(h).view(B, L, self.heads, self.dh))
        q, k, v, i_t, f_t, o = (t[:, order] for t in (q, k, v, i_t, f_t, o))
        C = torch.zeros(B, self.heads, self.dh, self.dh, device=x.device)
        n = torch.zeros(B, self.heads, self.dh, device=x.device)
        outs = []
        for t in range(L):
            # gate scalars need explicit (B,H,1,1) to broadcast against C
            C = (f_t[:, t].unsqueeze(-1) * C
                 + i_t[:, t].unsqueeze(-1) * torch.einsum('bhd,bhe->bhde', v[:, t], k[:, t]))
            n = f_t[:, t] * n + i_t[:, t] * k[:, t]
            qn = (C.transpose(-1, -2) @ q[:, t].unsqueeze(-1)).squeeze(-1)   # (B,H,dh)
            denom = (n * q[:, t]).sum(-1, keepdim=True).abs().clamp(min=1.0)
            outs.append(qn / denom * o[:, t])
        y = torch.stack(outs, dim=1)                    # (B, L, H, dh)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(L, device=order.device)
        return x + self.out_proj(y[:, inv].reshape(B, L, -1))


class XLSTMEncoder(TokenEncoderBase):
    def __init__(self, model_dim=192, n_layers=8, heads=4, board_size=8,
                 map_channels=128, name="xlstm", return_val=False):
        super().__init__(name=name, return_val=return_val)
        d = model_dim
        self.model_dim, self.n_layers, self.heads = d, n_layers, heads
        self.board_size = board_size
        self.trunk_channels = map_channels
        self.embed = nn.Linear(2, d)
        self.pos_embed = nn.Parameter(torch.randn(board_size * board_size, d) * 0.02)
        self.blocks = nn.ModuleList([MLSTMBlock(d, heads) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d)
        self.policy_head = nn.Linear(d, 1)
        self.map_proj = nn.Conv1d(d, map_channels, kernel_size=1)
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
        self.arch_spec = ("xlstm", model_dim, n_layers, heads)

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
