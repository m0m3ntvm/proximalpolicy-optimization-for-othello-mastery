"""Shared plumbing for the non-ConvNeXt encoder architectures.

train.py / selfplay / eval code is duck-typed against the OthelloRL
attribute surface, so new encoders inherit this small base to get the same
save/load semantics (full-object pickles + __dict__.update, the
--load-checkpoint resume flow) without touching OthelloRL itself — its
pickles stay byte-compatible.
"""
import torch
from torch import nn


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
    attributes, full-object-pickle save/load, and (in subclasses) the
    masked-softmax policy/value contract in forward()."""

    def __init__(self, name="A", return_val=False):
        super().__init__()
        self.name = name
        self.return_val = return_val
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
