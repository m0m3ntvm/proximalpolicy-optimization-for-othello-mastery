"""Board-state, pass-handling, match-scoring, and self-play return tests."""
import random

import numpy as np
import pytest
import torch

from othello.game.board import OthelloBoard
from othello.game.rules import evaluate
from othello.search.minimax import miniMax


def test_switch_player_canonicalizes():
    b = OthelloBoard()
    before = b.board.copy()
    b.switchPlayer()
    assert b.player is False
    assert np.array_equal(b.board, np.rot90(before, k=2) * -1)
    b.switchPlayer()
    assert b.player is True
    assert np.array_equal(b.board, before)


def test_make_move_flips_and_switches():
    b = OthelloBoard()
    (x, mask), moves = b.getBoard()
    assert len(moves) == 4
    target = (2, 4)  # e3
    assert target in moves
    assert b.makeMove(helpers=(x, mask), x=2, y=4, board_change=moves[target]) is True
    # white now holds 4, black 1; after switching, the board is the opponent's
    # perspective, so +1 pieces belong to black
    assert (b.board == 1).sum() == 1
    assert (b.board == -1).sum() == 4
    assert b.player is False


def test_make_move_rejects_out_of_bounds():
    b = OthelloBoard()
    (x, mask), moves = b.getBoard()
    with pytest.raises(ValueError):
        b.makeMove(helpers=(x, mask), x=8, y=0, board_change=moves[(2, 4)])


def test_make_move_rejects_occupied_square():
    b = OthelloBoard()
    (x, mask), moves = b.getBoard()
    b.makeMove(helpers=(x, mask), x=2, y=4, board_change=moves[(2, 4)])
    occupied = np.argwhere(b.board != 0)[0]  # some occupied square in the new perspective
    with pytest.raises(ValueError):
        b.makeMove(helpers=b.getBoard()[0], x=int(occupied[0]), y=int(occupied[1]),
                   board_change=np.zeros((8, 8), dtype=bool))


def test_single_pass_continues_game():
    """A player with no moves passes; the game continues from the opponent's perspective."""
    b = OthelloBoard()
    # construct: +1 has no moves but the game is not over
    b.board = np.ones((8, 8), dtype=np.short)
    b.board[0, 0] = 0
    b.board[7, 7] = -1
    b.last_player_moved = True
    (x, mask), moves = b.getBoard()
    assert len(moves) == 0
    assert b.makeMove(helpers=(x, mask), board_change=None) is True  # single pass
    assert b.player is False  # switched to the opponent
    assert b.last_player_moved is False


def test_double_pass_ends_game():
    b = OthelloBoard()
    b.board = np.ones((8, 8), dtype=np.short)
    b.board[0, 0] = 0
    b.last_player_moved = True
    (x, mask), moves = b.getBoard()
    assert len(moves) == 0
    assert b.makeMove(helpers=(x, mask), board_change=None) is True  # first pass
    assert b.makeMove(helpers=(x, mask), board_change=None) is False  # second pass ends game


def test_model_move_skips_model_when_no_moves():
    """modelMove must pass without consulting the model when no legal move exists."""
    b = OthelloBoard()
    b.board = np.ones((8, 8), dtype=np.short)
    b.board[0, 0] = 0
    b.last_player_moved = True

    class ExplodingModel:
        name = 'exploding'

        def __call__(self, *args, **kwargs):
            raise RuntimeError('model consulted on a pass position')

    assert b.modelMove(ExplodingModel(), switching=True, epsilon=0.0) is True  # single pass


def test_model_move_nan_output_falls_back_to_legal_move():
    """NaN/all-zero model outputs must not crash the game: a random legal move is played."""

    class NaNModel:
        name = 'nan'

        def __call__(self, x, mask):
            return torch.full((1, 1, 8, 8), float('nan'))

    for boltzmann in (False, True):
        random.seed(0)
        b = OthelloBoard()
        assert b.modelMove(NaNModel(), switching=True, epsilon=0.0, boltzmann=boltzmann) is True
        assert np.count_nonzero(b.board) == 5  # exactly one legal move was made


def reference_winner(board):
    """Perspective-correct winner: +1 pieces belong to board.player (True=white).

    Returns 1 for white, -1 for black, 0 for a draw.
    """
    delta = float(evaluate(board.board))
    white_score = delta if board.player else -delta
    if white_score == 0:
        return 0
    return 1 if white_score > 0 else -1


def test_model_vs_model_double_pass_scoring():
    """modelVsModel's winner attribution agrees with a perspective-correct reference
    on a constructed double-pass ending (white clearly ahead)."""
    b = OthelloBoard()
    b.board = np.ones((8, 8), dtype=np.short)
    b.board[0, 0] = 0
    b.last_player_moved = True
    dummy = miniMax(depth=1)  # never consulted: no legal moves anywhere
    result = b.modelVsModel(dummy, dummy, start_explore=(0, 0))
    assert reference_winner(b) == 1
    # white (first model) wins => [True, False] per the documented convention
    assert result.tolist() == [True, False]


def test_model_vs_model_model_one_always_plays_white():
    """model_1 always moves first (white): colours alternate strictly across plies,
    including through passes, so model_1's colour no longer depends on the opening."""
    calls = []
    dummy = miniMax(depth=1)
    orig = OthelloBoard.modelMove

    def tracked(self, OthelloRL, switching=True, epsilon=0.0, boltzmann=True, device=None):
        calls.append(self.player)
        return orig(self, OthelloRL, switching, epsilon, boltzmann, device)

    OthelloBoard.modelMove = tracked
    try:
        random.seed(3)
        b = OthelloBoard()
        b.modelVsModel(dummy, dummy, start_explore=(3, 40))
    finally:
        OthelloBoard.modelMove = orig

    assert calls[0] is True  # white moves first: that ply belongs to model_1
    assert all(p1 != p2 for p1, p2 in zip(calls, calls[1:]))  # colours alternate every ply


def test_model_vs_model_agrees_with_reference_on_random_games():
    """modelVsModel's result attribution agrees with the perspective-correct reference
    across varied openings and playouts (covers all four ending paths).

    model_1 always plays white, so the result maps directly onto the reference.
    """
    dummy = miniMax(depth=1)
    for g in range(150):
        random.seed(g)
        b = OthelloBoard()
        result = b.modelVsModel(dummy, dummy, start_explore=(0, 40))
        ref = reference_winner(b)
        expected = [True, False] if ref == 1 else [False, True] if ref == -1 else [False, False]
        assert result.tolist() == expected, f'game {g}: result {result.tolist()} vs reference {ref}'


def test_random_games_terminate_legally():
    """Random games always terminate with a full board or a double pass, never corrupt."""
    random.seed(1)
    dummy = miniMax(depth=1)
    for _ in range(50):
        b = OthelloBoard()
        nz = 4
        alive = True
        for _ in range(100):
            alive = b.modelMove(dummy, epsilon=1.0, boltzmann=False, device=None)
            if not alive:
                break
            new_nz = np.count_nonzero(b.board)
            assert new_nz in (nz, nz + 1), f'illegal change {nz} -> {new_nz}'
            nz = new_nz
        else:
            raise AssertionError('game did not terminate in 100 plies')


def test_self_play_returns_are_gamma_scaled_and_signed():
    """selfPlay targets: +gamma**empties for the winner's plies, -gamma**empties for the
    loser's plies, 0 for draws. Regresses the precedence bug where the black loser's
    return was -1 ** scale == -1.0 instead of -scale."""
    winners = set()
    for seed in range(12):
        random.seed(seed)
        b = OthelloBoard()
        winner, data = b.selfPlay(miniMax(depth=1), start_explore=(0, 4), epsilon=1.0,
                                  boltzmann=False)
        assert data is not None
        n_white = len(b.white_actions)
        assert len(data) == n_white + len(b.black_actions)
        for i, ((state, _mask), _action, ret) in enumerate(data):
            empties = np.count_nonzero(state[0] == 0)
            scale = float(b.gamma ** empties)
            is_white = i < n_white
            if winner == -1:
                expected = 0.0
            else:
                expected = scale if (winner == is_white) else -scale
            assert abs(float(ret) - expected) < 1e-6, (
                f'seed {seed} ply {i}: ret={ret} expected={expected}')
        winners.add(winner)
        if len(winners) == 2:
            break
    assert winners == {True, False}  # both winning and losing branches were exercised
