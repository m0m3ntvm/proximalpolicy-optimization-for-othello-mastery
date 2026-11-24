"""Published checkpoint compatibility: all 8 must load and play legal games."""
import os
import random

import numpy as np
import pytest
import torch

from othello.checkpoint import load_agent
from othello.game.board import OthelloBoard

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints')

ALL = [1, 2, 3, 4, 5, 6, 7, 8]


@pytest.mark.parametrize('index', ALL)
def test_checkpoint_loads(index):
    path = os.path.join(CKPT_DIR, f'{index}.pth')
    assert os.path.exists(path)
    agent = load_agent(path, 'cpu')
    assert agent.internal_channels == 64
    assert agent.res_depth == 8
    assert agent.return_val is False
    assert not agent.training  # load_agent puts the model in eval mode


@pytest.mark.parametrize('index', ALL)
def test_checkpoint_plays_legal_game(index):
    """One full network-played game per checkpoint; every ply must be a legal move."""
    random.seed(index)
    torch.manual_seed(index)
    agent = load_agent(os.path.join(CKPT_DIR, f'{index}.pth'), 'cpu')
    b = OthelloBoard()
    nz = 4
    for _ in range(100):
        alive = b.modelMove(agent, epsilon=0.0, boltzmann=True, device=None)
        if not alive:
            break
        new_nz = np.count_nonzero(b.board)
        assert new_nz in (nz, nz + 1), f'ckpt-{index}: illegal board change {nz} -> {new_nz}'
        nz = new_nz
    else:
        pytest.fail(f'ckpt-{index}: game did not terminate in 100 plies')
