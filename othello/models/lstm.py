"""Stacked-LSTM encoder for the 8x8 board: 64 square tokens through
vanilla LSTM layers (cuDNN, one kernel per layer — no Python scan loop),
with the Vision-LSTM trick of cycling row/column forward/backward scan
orders so the recurrence mixes both spatial axes. Sized to match the
ConvNeXt A/B arm (~13M params): d=512 x 6 layers.

Contract (same as OthelloRL): forward(x, mask) -> masked-softmax policy
(B,1,8,8), plus (policy, value) when return_val=True; features(x) ->
(trunk map (B,128,8,8), pre-softmax logits) for the MAE decoder.
"""
import torch
from torch import nn

from othello.models.base import TokenEncoderBase, scan_orders


class ScanLSTM(nn.LSTM):
    """nn.LSTM that returns only the output sequence — train.py's
    torchsummary() hooks every leaf module and chokes on the
    (output, (h_n, c_n)) tuple."""

    def forward(self, x):
        return super().forward(x)[0]


class LSTMBlock(nn.Module):
    """LayerNorm -> LSTM over the permuted token order -> residual. The
    permutation is undone afterwards so position embeddings stay aligned."""

    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.lstm = ScanLSTM(d, d, batch_first=True)

    def forward(self, x, order):
        h = self.norm(x)
        h = h[:, order]
        h = self.lstm(h)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(len(order), device=order.device)
        return x + h[:, inv]


class LSTMEncoder(TokenEncoderBase):
    def __init__(self, model_dim=512, n_layers=6, board_size=8,
                 map_channels=128, name="lstm", return_val=False):
        super().__init__(model_dim, board_size=board_size, map_channels=map_channels,
                         name=name, return_val=return_val)
        self.n_layers = n_layers
        self.blocks = nn.ModuleList([LSTMBlock(model_dim) for _ in range(n_layers)])
        self.arch_spec = ("lstm", model_dim, n_layers)

    def features(self, x):
        toks = self.tokenize(x)
        for i, blk in enumerate(self.blocks):
            order = scan_orders(self.board_size, self.board_size, i).to(x.device)
            toks = blk(toks, order)
        return self.heads_from(toks)
