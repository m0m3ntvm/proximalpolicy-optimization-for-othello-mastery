"""Board-state, pass-handling, and match-scoring tests."""
import random

import numpy as np

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


def test_model_vs_model_agrees_with_reference_on_random_games():
    """modelVsModel's result attribution agrees with the perspective-correct reference
    across varied openings and playouts (covers all four ending paths).

    modelVsModel plays every opening ply with model_1, so model_1's colour depends
    on the parity of the random opening length; the reference must account for that.
    """
    dummy = miniMax(depth=1)
    for g in range(150):
        random.seed(g)
        b = OthelloBoard()
        m1_white = [None]
        orig = OthelloBoard.modelMove

        def tracked(self, OthelloRL, switching=True, epsilon=0.0, boltzmann=True, device=None):
            if epsilon < 1.0 and m1_white[0] is None:  # first non-opening ply
                m1_white[0] = self.player
            return orig(self, OthelloRL, switching, epsilon, boltzmann, device)

        OthelloBoard.modelMove = tracked
        try:
            result = b.modelVsModel(dummy, dummy, start_explore=(0, 40))
        finally:
            OthelloBoard.modelMove = orig

        ref = reference_winner(b)
        assert m1_white[0] is not None
        m1_won = (ref == 1) == m1_white[0]  # white's result, mapped through m1's colour
        expected = [True, False] if m1_won else [False, True] if ref != 0 else [False, False]
        assert result.tolist() == expected, (
            f'game {g}: result {result.tolist()} vs reference {ref} (m1_white={m1_white[0]})')


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
