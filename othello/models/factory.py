"""Encoder factory: one construction point for every architecture.

train.py, pretrain_mae.py and the eval worker pools all build agents
through make_agent() so a single --arch flag selects the encoder, and the
attached arch_spec tuple (a plain picklable attribute, NOT a registered
buffer) lets spawn-pool eval workers rebuild the same architecture from a
state_dict via make_from_spec(). The convnext entry is the untouched
OthelloRL with arch_spec attached.
"""
from othello.models.agent import OthelloRL
from othello.models.vim import VimEncoder
from othello.models.xlstm import XLSTMEncoder

ARCHS = ("convnext", "vim", "xlstm")


def make_agent(arch="convnext", internal_channels=64, res_depth=8,
               model_dim=192, n_layers=8, d_state=16, expand=2, heads=4,
               name="A", return_val=False):
    if arch == "convnext":
        m = OthelloRL(internal_channels=internal_channels, res_depth=res_depth,
                      name=name, return_val=return_val)
        m.arch_spec = ("convnext", internal_channels, res_depth)
    elif arch == "vim":
        m = VimEncoder(model_dim=model_dim, n_layers=n_layers, d_state=d_state,
                       expand=expand, name=name, return_val=return_val)
    elif arch == "xlstm":
        m = XLSTMEncoder(model_dim=model_dim, n_layers=n_layers, heads=heads,
                         name=name, return_val=return_val)
    else:
        raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHS}")
    return m


def make_from_spec(arch_spec, name="A", return_val=False):
    """Rebuild an architecture from an arch_spec tuple."""
    arch = arch_spec[0]
    if arch == "convnext":
        _, internal_channels, res_depth = arch_spec
        return make_agent(arch=arch, internal_channels=internal_channels,
                          res_depth=res_depth, name=name, return_val=return_val)
    if arch == "vim":
        _, model_dim, n_layers, d_state, expand = arch_spec
        return make_agent(arch=arch, model_dim=model_dim, n_layers=n_layers,
                          d_state=d_state, expand=expand, name=name,
                          return_val=return_val)
    if arch == "xlstm":
        _, model_dim, n_layers, heads = arch_spec
        return make_agent(arch=arch, model_dim=model_dim, n_layers=n_layers,
                          heads=heads, name=name, return_val=return_val)
    raise ValueError(f"unknown arch_spec {arch_spec}")
