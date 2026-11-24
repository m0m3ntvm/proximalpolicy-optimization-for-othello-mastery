"""Minimax engine consistency tests on small boards (vs an exhaustive reference)."""
import random

import numba
import numpy as np
import pytest
import torch

from othello.game.board import OthelloBoard
from othello.game.rules import evaluate, get_valid_moves
from othello.search.minimax import miniMax, minimax


def make_tt():
    return numba.typed.Dict.empty(
        key_type=numba.types.int64,
        value_type=numba.types.Tuple((
            numba.types.float64,
            numba.types.Tuple((numba.types.int64, numba.types.int64)),
            numba.types.boolean)),
    )


def brute_force(board, depth, maximizing=True):
    """Exhaustive (no pruning) search with the same piece-count evaluation."""
    if depth == 0 or np.all(board != 0):
        return evaluate(board)
    legal, _, move_board = get_valid_moves(board if maximizing else -board)
    if not np.any(legal):
        return evaluate(board)
    sign = 1 if maximizing else -1
    best = -np.inf if maximizing else np.inf
    for move, flips in move_board.items():
        clone = board.copy()
        clone[move[0], move[1]] = sign
        clone = np.where(flips, sign, clone)
        val = brute_force(clone, depth - 1, not maximizing)
        best = max(best, val) if maximizing else min(best, val)
    return best


def random_position(seed, size=4, plies=3):
    """A midgame position from a random opening on a small board."""
    b = OthelloBoard(size=size)
    random.seed(seed)
    for _ in range(plies):
        b.modelMove(miniMax(depth=1), epsilon=1.0, boltzmann=False)  # model never consulted
    return b.board.astype(np.float64)


def assert_agrees(board, depth):
    value, move = minimax(board, make_tt(), depth)
    assert value == brute_force(board, depth)
    _, _, move_board = get_valid_moves(board)
    if move_board:
        assert move in move_board
    else:
        assert move == (0, 0)  # no legal move: engine reports a pass


@pytest.mark.parametrize('depth', [1, 2, 3, 4])
def test_minimax_initial_position(depth):
    assert_agrees(OthelloBoard(size=4).board.astype(np.float64), depth)


@pytest.mark.parametrize('seed', [0, 1, 2, 3])
@pytest.mark.parametrize('depth', [2, 3])
def test_minimax_random_positions(seed, depth):
    board = random_position(seed)
    assert not np.all(board != 0)
    assert_agrees(board, depth)


def test_wrapper_output_is_legal_move():
    b = OthelloBoard(size=4)
    (x, mask), moves = b.getBoard()
    eng = miniMax(depth=2)
    out = eng(torch.from_numpy(x), torch.from_numpy(mask))
    assert out.shape == (1, 1, 4, 4)
    assert out.sum().item() == 1.0
    argmax = np.unravel_index(out[0, 0].numpy().argmax(), (4, 4))
    assert argmax in moves
