"""PPO_Loss accuracy tests: clip direction, value loss, entropy, gradients."""
import numpy as np
import pytest
import torch

from othello.rl.ppo import PPO_Loss


def make_batch(n=7, seed=0):
    """A batch of valid-ish policies/actions for n states."""
    g = torch.Generator().manual_seed(seed)
    policy_old = torch.rand(n, 1, 8, 8, generator=g) + 1e-3
    policy_new = torch.rand(n, 1, 8, 8, generator=g) + 1e-3
    policy_old = policy_old / policy_old.sum(dim=(2, 3), keepdim=True)
    policy_new = policy_new / policy_new.sum(dim=(2, 3), keepdim=True)
    actions = torch.zeros_like(policy_new)
    idx = torch.randint(0, 64, (n,), generator=g)
    actions.view(n, 64)[torch.arange(n), idx] = 1.0
    values = torch.randn(n, 1, generator=g)
    ret = torch.randn(n, generator=g)
    return policy_old, policy_new, actions, values, ret


def manual_loss(policy_old, policy_new, actions, values, ret, clip, c_v, c_e):
    """Independent reference implementation of the PPO objective."""
    ret = ret.unsqueeze(-1).float()
    advantages = ret - values.detach()
    prob_old = (policy_old.detach() * actions).sum(dim=(2, 3), keepdim=True).squeeze(-1).squeeze(-1) + 1e-9
    prob_new = (policy_new * actions).sum(dim=(2, 3), keepdim=True).squeeze(-1).squeeze(-1)
    ratio = prob_new / prob_old
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip)
    policy_loss = -torch.minimum(advantages * ratio, advantages * clipped).mean()
    value_loss = torch.nn.functional.mse_loss(values, ret)
    pol = torch.clamp(policy_new, 1e-8, 1 - 1e-8).view(policy_new.size(0), -1)
    entropy_loss = -torch.mean(torch.sum(pol * torch.log(pol), dim=1))
    total = policy_loss + c_v * value_loss - c_e * entropy_loss
    return total, policy_loss, value_loss, entropy_loss


@pytest.fixture
def loss_fn():
    return PPO_Loss(clip_param=0.2, value_coefficient=0.5, entropy_coefficient=0.09, echo=True)


def test_loss_matches_manual_reference(loss_fn):
    batch = make_batch(seed=3)
    got = loss_fn(*batch)
    expected = manual_loss(*batch, 0.2, 0.5, 0.09)
    for g, e in zip(got[:4], expected):
        assert torch.allclose(g, e, atol=1e-6), f'{g.item()} vs {e.item()}'
    # the fifth echoed value is the mean advantage
    advantages = batch[4].unsqueeze(-1).float() - batch[3].detach()
    assert torch.allclose(got[4], advantages.mean())


def test_clip_limits_positive_advantage_updates():
    """ratio >> 1 + clip with positive advantage: min() must pick the clipped term."""
    n, clip = 4, 0.2
    policy_old = torch.full((n, 1, 8, 8), 1 / 64)
    policy_new = torch.full((n, 1, 8, 8), 1 / 64)
    policy_new[:, 0, 0, 0] = 0.5  # taken action becomes much more likely
    actions = torch.zeros_like(policy_new)
    actions[:, 0, 0, 0] = 1.0
    values = torch.zeros(n, 1)
    ret = torch.full((n,), 2.0)  # advantage = +2
    loss_fn = PPO_Loss(clip_param=clip, value_coefficient=0.5, entropy_coefficient=0.09, echo=True)
    _, policy_loss, _, _, _ = loss_fn(policy_old, policy_new, actions, values, ret)
    ratio = 0.5 / (1 / 64 + 1e-9)
    assert ratio > 1 + clip  # the clip must engage
    # min(A*r, A*clip(r)) = A*(1+clip) for positive A
    assert torch.allclose(policy_loss, torch.tensor(-2.0 * (1 + clip)), atol=1e-5)


def test_clip_does_not_limit_negative_advantage():
    """ratio >> 1 with negative advantage: min() must pick the unclipped term
    (clipping only limits *gains* on good actions, not penalties on bad ones)."""
    n, clip = 4, 0.2
    policy_old = torch.full((n, 1, 8, 8), 1 / 64)
    policy_new = torch.full((n, 1, 8, 8), 1 / 64)
    policy_new[:, 0, 0, 0] = 0.5
    actions = torch.zeros_like(policy_new)
    actions[:, 0, 0, 0] = 1.0
    values = torch.zeros(n, 1)
    ret = torch.full((n,), -2.0)  # advantage = -2
    loss_fn = PPO_Loss(clip_param=clip, value_coefficient=0.5, entropy_coefficient=0.09, echo=True)
    _, policy_loss, _, _, _ = loss_fn(policy_old, policy_new, actions, values, ret)
    ratio = 0.5 / (1 / 64 + 1e-9)
    # min(A*r, A*clip) = A*r for negative A and r > clip -> loss = -A*r
    assert torch.allclose(policy_loss, torch.tensor(2.0 * ratio), atol=0.1)
    assert policy_loss.item() > 2.0 * (1 + clip)  # the clip must NOT have engaged


def test_value_loss_is_mse(loss_fn):
    policy_old, policy_new, actions, values, ret = make_batch(seed=5)
    _, _, value_loss, _, _ = loss_fn(policy_old, policy_new, actions, values, ret)
    expected = torch.nn.functional.mse_loss(values, ret.unsqueeze(-1).float())
    assert torch.allclose(value_loss, expected, atol=1e-6)


def test_entropy_of_uniform_policy():
    n = 3
    policy_old = torch.full((n, 1, 8, 8), 1 / 64)
    policy_new = torch.full((n, 1, 8, 8), 1 / 64)
    actions = torch.zeros_like(policy_new)
    actions[:, 0, 0, 0] = 1.0
    values = torch.zeros(n, 1)
    ret = torch.zeros(n)
    loss_fn = PPO_Loss(clip_param=0.2, value_coefficient=0.5, entropy_coefficient=0.09, echo=True)
    _, _, _, entropy_loss, _ = loss_fn(policy_old, policy_new, actions, values, ret)
    # entropy of a uniform policy over 64 actions = ln(64)
    assert abs(entropy_loss.item() - np.log(64)) < 1e-4


def test_gradients_flow_through_policy(loss_fn):
    policy_old, _, actions, values, ret = make_batch(seed=11)
    policy_new = torch.rand(7, 1, 8, 8, requires_grad=True) + 1e-3
    policy_new = policy_new / policy_new.sum(dim=(2, 3), keepdim=True)
    policy_new.retain_grad()  # normalized tensor is not a leaf
    total, *_ = loss_fn(policy_old, policy_new, actions, values, ret)
    total.backward()
    assert policy_new.grad is not None
    assert policy_new.grad.abs().sum() > 0
