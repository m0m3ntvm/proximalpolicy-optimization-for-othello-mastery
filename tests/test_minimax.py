"""Minimax engine consistency tests on small boards (vs an exhaustive reference)."""
import random

import numpy as np
import pytest
import torch

from othello.game.board import OthelloBoard
from othello.game.rules import evaluate, get_valid_moves
from othello.search.minimax import _gen_moves, _to_padded, make_transposition_table, miniMax, minimax


def make_tt():
    return make_transposition_table()


def brute_force(board, depth, maximizing=True):
    """Exhaustive (no pruning) search with the same piece-count evaluation."""
    if depth == 0 or np.all(board != 0):
        return evaluate(board)
    legal, _, move_board = get_valid_moves(board if maximizing else -board)
    if not np.any(legal):
        return brute_force(board, depth - 1, not maximizing)  # pass
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


def find_pass_position(rng, size=4, trials=100000):
    """A position where the player to move has no legal move but the opponent does."""
    for _ in range(trials):
        board = np.zeros((size, size))
        n = rng.integers(4, size * size - 1)
        idx = rng.choice(size * size, n, replace=False)
        board.flat[idx] = rng.choice(np.array([1.0, -1.0]), n)
        mask_plus, _, _ = get_valid_moves(board)
        mask_minus, _, _ = get_valid_moves(-board)
        if not np.any(mask_plus) and np.any(mask_minus):
            return board
    return None


@pytest.mark.parametrize('depth', [2, 3])
def test_minimax_handles_pass_positions(depth):
    """A player with no legal move passes: the engine must search the opponent's reply
    instead of treating the position as terminal."""
    board = find_pass_position(np.random.default_rng(7))
    assert board is not None
    value, move = minimax(board, make_tt(), depth)
    assert value == brute_force(board, depth)  # reference implements the same pass rule
    assert move == (0, 0)  # the player to move has no move


def test_minimax_pass_position_is_not_treated_as_terminal():
    """At depth 2 the pass lets the opponent flip pieces with no reply, so the value
    must move off the raw piece count (the legacy engine returned evaluate(board)
    as if the game had ended)."""
    board = find_pass_position(np.random.default_rng(7))
    assert board is not None
    value, _ = minimax(board, make_tt(), depth=2)
    assert value == brute_force(board, 2)
    assert value != evaluate(board)


def test_wrapper_transposition_table_persists():
    """The wrapper keeps one transposition table across calls instead of rebuilding it."""
    b = OthelloBoard(size=4)
    (x, mask), moves = b.getBoard()
    eng = miniMax(depth=2)
    out1 = eng(torch.from_numpy(x), torch.from_numpy(mask))
    assert len(eng.transposition_table) > 0
    out2 = eng(torch.from_numpy(x), torch.from_numpy(mask))
    assert torch.equal(out1, out2)
    # a deeper search on a later position reuses the same table and stays legal
    eng.depth = 3
    b2 = OthelloBoard(size=4)
    (x2, mask2), moves2 = b2.getBoard()
    out3 = eng(torch.from_numpy(x2), torch.from_numpy(mask2))
    argmax = np.unravel_index(out3[0, 0].numpy().argmax(), (4, 4))
    assert argmax in moves2


def test_tt_does_not_serve_shallow_entries_to_deeper_searches():
    """Persisted TT entries carry their search depth and must only be reused when they
    cover the requested depth, otherwise a shallow cached score would leak upward."""
    b = OthelloBoard(size=4)
    (x, mask), _ = b.getBoard()
    board = x[0][0].astype(np.float64)
    tt = make_tt()
    value_d2, _ = minimax(board, tt, depth=2)
    value_d3, _ = minimax(board, tt, depth=3)  # same TT: must re-search, not reuse depth-2
    assert value_d3 == brute_force(board, 3)  # sound value, not the depth-2 cache
    assert value_d2 == brute_force(board, 2)


def engine_move_set(board, own):
    """Legal moves of `own` (in the board's frame) as (x, y) coordinates."""
    b, P, n = _to_padded(np.asarray(board))
    moves, flips, cnt = _gen_moves(b, own, -own, P, n)
    return {(int(m) // P - 1, int(m) % P - 1) for m in moves[:cnt]}


@pytest.mark.parametrize('seed', range(8))
def test_engine_movegen_matches_rules(seed):
    """The standalone move generator must agree with othello.game.rules on random
    positions (this pins the rewrite of the engine's own move generation)."""
    rng = np.random.default_rng(seed)
    board = np.zeros((6, 6))
    n = rng.integers(1, 30)
    idx = rng.choice(36, n, replace=False)
    board.flat[idx] = rng.choice(np.array([1.0, -1.0]), n)
    for own in (1.0, -1.0):
        mask, _, _ = get_valid_moves(board if own == 1 else -board)
        reference = {(x, y) for x in range(6) for y in range(6) if mask[x, y]}
        assert engine_move_set(board, own) == reference


def test_parallel_root_matches_serial():
    """Root-level parallelism must return the exact same value and move as the serial
    search (first-max tie-break in the same move order)."""
    b = OthelloBoard(size=4)
    (x, mask), _ = b.getBoard()
    eng_serial = miniMax(depth=5, workers=1, return_val=True)
    eng_parallel = miniMax(depth=5, workers=4, parallel_depth=5, return_val=True)
    (out_s, val_s), (out_p, val_p) = (
        eng_serial(torch.from_numpy(x), torch.from_numpy(mask)),
        eng_parallel(torch.from_numpy(x), torch.from_numpy(mask)),
    )
    assert torch.equal(out_s, out_p)
    assert val_s == val_p
