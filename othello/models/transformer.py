"""Transformer encoder for the 8x8 board: 64 square tokens through pre-norm
self-attention blocks. At 64 tokens attention is tiny, so the model is sized
to match the ConvNeXt A/B arm (~13M params): d=256 x 16 blocks x 8 heads.

Contract (same as OthelloRL): forward(x, mask) -> masked-softmax policy
(B,1,8,8), plus (policy, value) when return_val=True; features(x) ->
(trunk map (B,128,8,8), pre-softmax logits) for the MAE decoder.
"""
import torch
from torch import nn

from othello.models.base import TokenEncoderBase


class SelfAttn(nn.MultiheadAttention):
    """MHA that returns only the output tensor — train.py's torchsummary()
    hooks every leaf module and chokes on the (output, weights) tuple."""

    def forward(self, x):
        return super().forward(x, x, x, need_weights=False)[0]


class TransformerBlock(nn.Module):
    """Pre-norm MHA + MLP block with residual connections (ViT recipe)."""

    def __init__(self, d, heads=8, mlp_ratio=4):
        super().__init__()
        assert d % heads == 0
        self.norm1 = nn.LayerNorm(d)
        self.attn = SelfAttn(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, d * mlp_ratio),
            nn.GELU(),
            nn.Linear(d * mlp_ratio, d),
        )

    def forward(self, x):
        h = self.attn(self.norm1(x))
        x = x + h
        return x + self.mlp(self.norm2(x))


class TransformerEncoder(TokenEncoderBase):
    def __init__(self, model_dim=256, n_layers=16, heads=8, board_size=8,
                 map_channels=128, name="transformer", return_val=False):
        super().__init__(model_dim, board_size=board_size, map_channels=map_channels,
                         name=name, return_val=return_val)
        self.n_layers, self.heads = n_layers, heads
        self.blocks = nn.ModuleList([TransformerBlock(model_dim, heads)
                                     for _ in range(n_layers)])
        self.arch_spec = ("transformer", model_dim, n_layers, heads)

    def features(self, x):
        toks = self.tokenize(x)
        for blk in self.blocks:
            toks = blk(toks)
        return self.heads_from(toks)
