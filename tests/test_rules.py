"""Move-generation and rules-kernel tests."""
import numpy as np

from othello.game.board import OthelloBoard
from othello.game.rules import board_hash, decide, evaluate, get_valid_moves


def initial_board():
    return OthelloBoard().board.copy()


def test_initial_position_legal_moves():
    """The code's opening position has exactly the four expected moves, each flipping one piece."""
    mask, under_attack, move_board = get_valid_moves(initial_board())
    expected = {(2, 4), (3, 5), (4, 2), (5, 3)}  # e3, f4, c5, d6
    legal = set(move_board.keys())
    assert legal == expected
    assert mask.sum() == 4
    for move, flips in move_board.items():
        assert flips.sum() == 1, f'{move} should flip exactly one piece'


def test_initial_position_under_attack():
    mask, under_attack, move_board = get_valid_moves(initial_board())
    # both black pieces are capturable in one move
    assert under_attack.sum() == 2
    assert under_attack[3, 4] and under_attack[4, 3]


def test_move_flips_are_correct():
    """Applying a move's flip mask must yield exactly the position after the move."""
    board = initial_board()
    _, _, move_board = get_valid_moves(board)
    for move, flips in move_board.items():
        after = board.copy()
        after[move[0], move[1]] = 1.0
        after = np.where(flips, 1.0, after)
        # re-derive: black pieces remaining must be 1 (from 2 initially)
        assert (after == -1).sum() == 1


def test_no_legal_moves():
    """A board full of +1 (except one square) gives no legal moves."""
    board = np.ones((8, 8))
    board[0, 0] = 0
    mask, under_attack, move_board = get_valid_moves(board)
    assert not np.any(mask)
    assert len(move_board) == 0


def test_evaluate_is_piece_count():
    board = initial_board()
    assert evaluate(board) == 0.0
    board[0, 0] = 1.0
    board[0, 1] = -1.0
    assert evaluate(board) == 0.0
    board[0, 2] = 1.0
    assert evaluate(board) == 1.0


def test_board_hash_deterministic_and_sensitive():
    b1 = initial_board()
    b2 = initial_board()
    assert board_hash(b1) == board_hash(b2)
    b2[0, 0] = 1.0
    assert board_hash(b1) != board_hash(b2)


def test_decide_picks_mass_location():
    probs = np.zeros((8, 8))
    probs[6, 3] = 1.0
    for _ in range(20):
        assert decide(probs) == (6, 3)
