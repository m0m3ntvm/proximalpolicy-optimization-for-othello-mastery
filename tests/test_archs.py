"""Encoder architecture tests: Transformer / LSTM must satisfy the same
duck-type contract as OthelloRL (masked-softmax policy, shared-trunk value
head, state_dict round-trip through the factory, numerical stability) so
the MAE and RL pipelines can swap architectures behind a single --arch
flag. The challenger arms are sized to match the convnext 128x16 arm
(~13M params)."""
import numpy as np
import pytest
import torch

from othello.models.base import scan_orders
from othello.models.factory import (AgentParallel, make_agent, make_from_spec,
                                     parallelize, unwrap)

# tiny sizes so the suite stays CPU-friendly
TINY = [
    ("convnext", dict(internal_channels=32, res_depth=2)),
    ("transformer", dict(model_dim=32, n_layers=2, heads=4)),
    ("lstm", dict(model_dim=32, n_layers=2)),
]


def _batch(arch, B=3):
    x = torch.randn(B, 2, 8, 8)
    # a random ~30%-density legal-move mask, as the board produces
    mask = (torch.rand(B, 1, 8, 8) < 0.3).float()
    if mask.sum() == 0:
        mask[0, 0, 0, 0] = 1.0
    return x, mask


@pytest.mark.parametrize("arch,kwargs", TINY)
def test_policy_is_masked_softmax(arch, kwargs):
    m = make_agent(arch=arch, name="q", **kwargs)
    x, mask = _batch(arch)
    policy = m(x, mask)
    assert policy.shape == (3, 1, 8, 8)
    assert torch.isfinite(policy).all()
    # zero probability on illegal squares, sums to 1 over legal ones
    assert (policy * (1 - mask) < 1e-6).all()
    assert torch.allclose(policy.sum(dim=(1, 2, 3)), torch.ones(3), atol=1e-6)


@pytest.mark.parametrize("arch,kwargs", TINY)
def test_value_head_shape_and_range(arch, kwargs):
    m = make_agent(arch=arch, name="q", return_val=True, **kwargs)
    x, mask = _batch(arch)
    policy, value = m(x, mask)
    assert value.shape == (3, 1)
    assert torch.isfinite(value).all()
    assert (value >= -1.0).all() and (value <= 1.0).all()  # tanh output


@pytest.mark.parametrize("arch,kwargs", TINY)
def test_features_contract_matches_forward(arch, kwargs):
    m = make_agent(arch=arch, name="q", **kwargs)
    x, mask = _batch(arch)
    trunk, logits = m.features(x)
    assert logits.shape == (3, 1, 8, 8)
    policy = m(x, mask)
    expected = (logits * mask + (1 - mask) * -1e32).view(3, -1).softmax(dim=1).view(3, 1, 8, 8)
    assert torch.allclose(policy, expected, atol=1e-6)
    # trunk map channels: 128 for the token encoders (fixed MAE decoder width)
    if arch == "convnext":
        assert trunk.shape == (3, kwargs["internal_channels"], 8, 8)
    else:
        assert trunk.shape == (3, 128, 8, 8)


@pytest.mark.parametrize("arch,kwargs", TINY)
def test_grad_flow(arch, kwargs):
    m = make_agent(arch=arch, name="q", return_val=True, **kwargs)
    x, mask = _batch(arch)
    policy, value = m(x, mask)
    (policy.sum() + value.sum()).backward()
    params = dict(m.named_parameters())
    if arch == "convnext":
        probe = "InConv.0.weight"
    else:
        probe = "embed.weight"
    assert params[probe].grad is not None
    assert torch.isfinite(params[probe].grad).all()


@pytest.mark.parametrize("arch,kwargs", TINY)
def test_state_dict_round_trip_via_factory(arch, kwargs):
    m = make_agent(arch=arch, name="q", return_val=True, **kwargs)
    x, mask = _batch(arch)
    p1, v1 = m(x, mask)
    rebuilt = make_from_spec(m.arch_spec, name="q2", return_val=True)
    rebuilt.load_state_dict(m.state_dict())  # strict by default
    p2, v2 = rebuilt(x, mask)
    assert torch.equal(p1, p2)
    assert torch.equal(v1, v2)


@pytest.mark.parametrize("arch,kwargs", [
    ("transformer", dict()),
    ("lstm", dict()),
])
def test_default_param_budget(arch, kwargs):
    """Default configs land within ~20% of the convnext 128x16 arm (13.5M)."""
    m = make_agent(arch=arch, **kwargs)
    n = sum(p.numel() for p in m.parameters())
    assert 8e6 < n < 20e6, f"{arch} has {n:,} params, expected ~13M"


def test_scan_orders_are_distinct_permutations():
    orders = [scan_orders(8, 8, i) for i in range(4)]
    for o in orders:
        assert sorted(o.tolist()) == list(range(64))
    seen = {tuple(o.tolist()) for o in orders}
    assert len(seen) == 4


@pytest.mark.parametrize("arch,kwargs", [
    ("transformer", dict(model_dim=32, n_layers=2, heads=4)),
    ("lstm", dict(model_dim=32, n_layers=2)),
])
def test_numerical_stability_on_large_inputs(arch, kwargs):
    m = make_agent(arch=arch, name="q", return_val=True, **kwargs)
    x, mask = _batch(arch)
    policy, value = m(x * 20.0, mask)  # large logits stress the exp gates
    assert torch.isfinite(policy).all()
    assert torch.isfinite(value).all()


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DataParallel requires a CUDA device")
def test_agentparallel_delegates_and_unwraps():
    m = make_agent(arch="transformer", model_dim=32, n_layers=2, heads=4,
                   name="q", return_val=True)
    wrapped = AgentParallel(m)
    # reads delegate to the inner model (duck-type contract)
    assert wrapped.arch_spec == m.arch_spec
    assert wrapped.name == "q"
    assert wrapped.return_val is True
    assert wrapped.trunk_channels == m.trunk_channels
    # writes to the two mutated duck attrs are routed to the inner model
    wrapped.return_val = False
    wrapped.name = "z"
    assert m.return_val is False
    assert m.name == "z"
    # state_dict: wrapper-prefixed on the DP object, clean after unwrap
    assert all(k.startswith("module.") for k in wrapped.state_dict())
    assert set(unwrap(wrapped).state_dict()) == set(m.state_dict())
    # unwrap is idempotent
    assert unwrap(wrapped) is m
    assert unwrap(m) is m


def test_parallelize_is_noop_on_single_gpu():
    m = make_agent(arch="lstm", model_dim=32, n_layers=2)
    out = parallelize(m)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        assert isinstance(out, AgentParallel)
        assert unwrap(out) is m
    else:
        assert out is m  # single-GPU hot path untouched
