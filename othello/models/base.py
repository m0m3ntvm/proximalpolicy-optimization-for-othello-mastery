"""Shared plumbing for the non-ConvNeXt encoder architectures.

train.py / selfplay / eval code is duck-typed against the OthelloRL
attribute surface, so new encoders inherit this small base to get the same
save/load semantics (full-object pickles + __dict__.update, the
--load-checkpoint resume flow) without touching OthelloRL itself — its
pickles stay byte-compatible.

The base also owns everything the token-based encoders (Transformer / LSTM
/ LSTM) share: 2ch-per-square embedding + learned positional encoding,
layer norm, the policy head, the fixed-128-channel trunk map projection
(the MAE 3D-conv decoder is identical across architectures), the
shared-trunk value head, and the masked-softmax forward(). Subclasses only
provide their block stack and features().
"""
import torch
from torch import nn

from othello.models.blocks import ResidualBlock


def scan_orders(height, width, idx):
    """Flattened token orderings for an (height, width) grid, cycling over
    row-major / column-major, forward / backward — the Vision-LSTM scan
    recipe. Position embeddings stay square-aligned: blocks permute tokens
    before the recurrence and undo the permutation afterwards.

    Returns a LongTensor index of length height * width."""
    n = height * width
    kind = idx % 4
    if kind == 0:
        order = torch.arange(n)
    elif kind == 1:
        order = torch.arange(n).view(height, width).t().reshape(-1)
    elif kind == 2:
        order = torch.flip(torch.arange(n), dims=(0,))
    else:
        order = torch.flip(torch.arange(n).view(height, width).t().reshape(-1), dims=(0,))
    return order


class TokenEncoderBase(nn.Module):
    """OthelloRL-compatible attribute surface: name / return_val / device
    attributes, full-object-pickle save/load, the shared token-encoder head
    stack, and the masked-softmax policy/value contract in forward()."""

    def __init__(self, d, board_size=8, map_channels=128, name="A", return_val=False):
        super().__init__()
        self.name = name
        self.return_val = return_val
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dim = d
        self.board_size = board_size
        self.trunk_channels = map_channels
        n = board_size * board_size
        self.embed = nn.Linear(2, d)
        self.pos_embed = nn.Parameter(torch.randn(n, d) * 0.02)
        self.norm_out = nn.LayerNorm(d)
        self.policy_head = nn.Linear(d, 1)
        # MAE trunk map: tokens -> (B, map_channels, 8, 8); fixed 128 channels
        # so the 3D-conv decoder is identical across architectures
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

    def tokenize(self, x):
        """(B, 2, 8, 8) board planes -> (B, 64, d) position-embedded tokens."""
        B = x.shape[0]
        n = self.board_size ** 2
        toks = self.embed(x.permute(0, 2, 3, 1).reshape(B, n, 2)) + self.pos_embed
        return toks

    def heads_from(self, toks):
        """(B, 64, d) -> (trunk map (B, 128, 8, 8), pre-softmax logits (B, 1, 8, 8))."""
        B = toks.shape[0]
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

    def save(self, path):
        try:
            torch.save(self, path)
            return True
        except Exception as e:
            print("Model Saving Failed :", e)
            return False

    def load(self, path):
        try:
            loaded_model = torch.load(path, map_location=self.device, weights_only=False)
            self.__dict__.update(loaded_model.__dict__)
            return True
        except Exception as e:
            print(f"Model {self.name} wasn't found :", e)
            return False
