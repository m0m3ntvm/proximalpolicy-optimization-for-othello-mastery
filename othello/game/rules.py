"""Core Othello rules as Numba-jitted kernels.

Board convention: +1 = player to move (canonicalized), -1 = opponent, 0 = empty.
"""
import numpy as np
from numba import njit


@njit
def board_hash(board) -> int:
    """Polynomial-roll hash of a board, used as a transposition-table key."""
    hash_value = 0
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            hash_value = (hash_value * 991 + int(board[i, j])) & 0xFFFFFFFFFFFFFFFF
    return hash_value


@njit
def evaluate(board) -> float:
    """Raw piece-count evaluation (positive favours the +1 player)."""
    return np.sum(board).item()


@njit
def get_valid_moves(board):
    """Legal moves for the +1 player.

    Returns (legal_moves, under_attack, move_board):
      legal_moves  : bool board, True where a move is legal
      under_attack : bool board, opponent pieces capturable in one move
      move_board   : {(row, col): bool board of pieces flipped by that move}
    """
    size = board.shape[0]
    legal_moves = np.zeros_like(board, dtype=np.bool_)
    possible_conv = np.zeros_like(board, dtype=np.bool_)

    move_board = {
        # save each move and the resulting board
    }
    for row_index in range(size):
        for col_index in range(size):
            if board[row_index, col_index] == 0.0:
                # Check in all eight directions
                row_min, row_max = max(row_index - 1, 0), min(row_index + 2, size)
                col_min, col_max = max(col_index - 1, 0), min(col_index + 2, size)
                dir = [(board[i, j] == -1.0, (i - row_index, j - col_index)) for i in range(row_min, row_max) for j in range(col_min, col_max)]

                # Check if any direction has a black piece
                shortlisted = False
                for indicator, _ in dir:
                    if indicator:
                        shortlisted = True
                        break

                if shortlisted:
                    # Use dir to check if a conversion will happen in this direction
                    conversion = np.zeros(board.shape, dtype=np.bool_)
                    sup_conv = np.zeros(board.shape, dtype=np.bool_)  # converts from a single move
                    for i in range(len(dir)):
                        indicator, (dy, dx) = dir[i]
                        if indicator:
                            ni, nj = row_index + dy, col_index + dx
                            temp = np.zeros(board.shape, dtype=np.bool_)
                            while 0 <= ni < size and 0 <= nj < size and board[ni, nj] == -1.0:
                                temp[ni, nj] = True
                                ni, nj = ni + dy, nj + dx

                            if 0 <= ni < size and 0 <= nj < size and board[ni, nj] == 1.0:
                                conversion = np.logical_or(conversion, temp)
                                sup_conv = np.logical_or(sup_conv, temp)
                      # save for use later in creating moves
                    # If conversion is possible, update legal moves and possible_conv
                    if np.any(conversion):
                        legal_moves[row_index, col_index] = True
                        possible_conv = np.logical_or(possible_conv, sup_conv)
                        move_board[(row_index, col_index)] = sup_conv

    return legal_moves, possible_conv, move_board


def decide(probabilities, rng=None):
    """Sample a move from a probability distribution over squares.

    rng: an np.random.RandomState / Generator to draw from (default: the
    global numpy stream).
    """
    probabilities = probabilities / probabilities.sum()
    indices = (rng if rng is not None else np.random).choice(
        probabilities.size, size=1, p=probabilities.ravel())
    i, j = indices[0] // probabilities.shape[1], indices[0] % probabilities.shape[1]
    return i, j
