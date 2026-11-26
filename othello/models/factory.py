"""Encoder factory: one construction point for every architecture.

train.py, pretrain_mae.py and the eval worker pools all build agents
through make_agent() so a single --arch flag selects the encoder, and the
attached arch_spec tuple (a plain picklable attribute, NOT a registered
buffer) lets spawn-pool eval workers rebuild the same architecture from a
state_dict via make_from_spec(). The convnext entry is the untouched
OthelloRL with arch_spec attached; the other arms are sized to match the
ConvNeXt A/B arm (~13M params) via _DEFAULTS.

Multi-GPU: parallelize() wraps an agent in AgentParallel (nn.DataParallel
with attribute delegation to the inner model) when more than one CUDA
device is available; unwrap() peels the wrapper back for state_dict /
load_state_dict / deepcopy sites. On a single GPU parallelize() is a no-op,
so the hot path is untouched on the RTX 3050 dev machine.
"""
import torch
from torch import nn

from othello.models.agent import OthelloRL
from othello.models.lstm import LSTMEncoder
from othello.models.mamba import MambaEncoder
from othello.models.transformer import TransformerEncoder

ARCHS = ("convnext", "transformer", "lstm", "mamba")

# per-arch defaults when explicit sizes are not given; the challenger arms
# all land within ~20% of the convnext 128x16 arm's 13.5M params
_DEFAULTS = {
    "convnext": dict(internal_channels=128, res_depth=16),
    "transformer": dict(model_dim=256, n_layers=16, heads=8),
    "lstm": dict(model_dim=512, n_layers=6),
    "mamba": dict(model_dim=896, n_layers=2, d_state=16, expand=2),
}


def make_agent(arch="convnext", internal_channels=None, res_depth=None,
               model_dim=None, n_layers=None, d_state=16, expand=2, heads=8,
               name="A", return_val=False):
    d = _DEFAULTS[arch]
    if arch == "convnext":
        ic, rd = internal_channels or d["internal_channels"], res_depth or d["res_depth"]
        m = OthelloRL(internal_channels=ic, res_depth=rd, name=name,
                      return_val=return_val)
        m.arch_spec = ("convnext", ic, rd)
    elif arch == "transformer":
        m = TransformerEncoder(model_dim=model_dim or d["model_dim"],
                               n_layers=n_layers or d["n_layers"], heads=heads,
                               name=name, return_val=return_val)
    elif arch == "lstm":
        m = LSTMEncoder(model_dim=model_dim or d["model_dim"],
                        n_layers=n_layers or d["n_layers"], name=name,
                        return_val=return_val)
    elif arch == "mamba":
        m = MambaEncoder(model_dim=model_dim or d["model_dim"],
                         n_layers=n_layers or d["n_layers"], d_state=d_state,
                         expand=expand, name=name, return_val=return_val)
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
    if arch == "transformer":
        _, model_dim, n_layers, heads = arch_spec
        return make_agent(arch=arch, model_dim=model_dim, n_layers=n_layers,
                          heads=heads, name=name, return_val=return_val)
    if arch == "lstm":
        _, model_dim, n_layers = arch_spec
        return make_agent(arch=arch, model_dim=model_dim, n_layers=n_layers,
                          name=name, return_val=return_val)
    if arch == "mamba":
        _, model_dim, n_layers, d_state, expand = arch_spec
        return make_agent(arch=arch, model_dim=model_dim, n_layers=n_layers,
                          d_state=d_state, expand=expand, name=name,
                          return_val=return_val)
    raise ValueError(f"unknown arch_spec {arch_spec}")


class AgentParallel(nn.DataParallel):
    """DataParallel that keeps the wrapped model's duck-type contract
    visible: reads (arch_spec, name, return_val, trunk_channels, features,
    internal_channels, ...) delegate to the inner module, and writes to the
    two attrs the pipeline mutates (name, return_val) are routed there too.
    state_dict() stays wrapper-prefixed on purpose — call unwrap() for the
    clean keys the checkpoints use."""

    def __getattr__(self, name):
        # reached only when normal lookup (incl. nn.Module's own __getattr__)
        # fails; delegate anything else to the inner model
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def __setattr__(self, name, value):
        # DataParallel.__init__ sets module/src_device/output_device through
        # this method; only the duck attrs the pipeline writes are routed
        # down (parameters/buffers/submodules fall through to nn.Module)
        if name in ("return_val", "name") and hasattr(self, "module"):
            setattr(self.module, name, value)
        else:
            super().__setattr__(name, value)


def unwrap(model):
    """Return the plain model behind a DataParallel wrapper (idempotent)."""
    while isinstance(model, nn.DataParallel):
        model = model.module
    return model


def parallelize(model):
    """Wrap in AgentParallel when more than one CUDA device is available,
    otherwise return the model unchanged (single-GPU hot path untouched)."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return AgentParallel(model)
    return model


def compile_blocks(model):
    """torch.compile each encoder block in place (SSM/LSTM scan fusion on
    the training box; fixed batch shapes keep the compile cache hot)."""
    m = unwrap(model)
    for i in range(len(m.blocks)):
        m.blocks[i] = torch.compile(m.blocks[i])
    return model
