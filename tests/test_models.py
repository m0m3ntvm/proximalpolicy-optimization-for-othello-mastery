"""Architecture smoke tests: masked softmax, output shapes, value head, CUDA."""
import numpy as np
import pytest
import torch

from othello.game.board import OthelloBoard
from othello.models.agent import OthelloRL, value_Net


def initial_inputs():
    b = OthelloBoard()
    (x, mask), moves = b.getBoard()
    return torch.from_numpy(x), torch.from_numpy(mask), moves


def test_policy_masked_softmax():
    model = OthelloRL(return_val=False).eval()
    x, mask, moves = initial_inputs()
    with torch.no_grad():
        out = model(x, mask)
    assert out.shape == (1, 1, 8, 8)
    assert torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-4)
    # masked-out squares get exactly zero mass (softmax(-1e32) underflows)
    assert (out * (1 - mask)).sum().item() == 0.0
    argmax = np.unravel_index(out[0, 0].numpy().argmax(), (8, 8))
    assert argmax in moves


def test_policy_single_legal_square():
    model = OthelloRL(return_val=False).eval()
    x, mask, _ = initial_inputs()
    mask = torch.zeros_like(mask)
    mask[0, 0, 5, 3] = 1.0
    with torch.no_grad():
        out = model(x, mask)
    assert torch.isclose(out[0, 0, 5, 3], torch.tensor(1.0), atol=1e-4)


def test_value_head_shapes_and_range():
    model = OthelloRL(return_val=True).eval()
    x, mask, _ = initial_inputs()
    with torch.no_grad():
        policy, value = model(x, mask)
    assert policy.shape == (1, 1, 8, 8)
    assert value.shape == (1, 1)
    assert -1.0 <= value.item() <= 1.0


def test_value_net_standalone():
    net = value_Net().eval()
    with torch.no_grad():
        out = net(torch.randn(1, 3, 8, 8))
    assert out.shape == (1, 1)
    assert -1.0 <= out.item() <= 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_cuda_forward():
    model = OthelloRL(return_val=True).cuda().eval()
    x, mask, moves = initial_inputs()
    with torch.no_grad():
        policy, value = model(x.cuda(), mask.cuda())
    assert policy.shape == (1, 1, 8, 8)
    assert torch.isclose(policy.sum().cpu(), torch.tensor(1.0), atol=1e-4)
    assert -1.0 <= value.item() <= 1.0
    argmax = np.unravel_index(policy[0, 0].cpu().numpy().argmax(), (8, 8))
    assert argmax in moves
