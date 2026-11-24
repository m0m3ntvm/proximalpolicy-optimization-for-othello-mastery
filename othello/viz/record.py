"""Play an agent-vs-agent game while recording every ply as a display-oriented state.

The loop mirrors OthelloBoard.modelVsModel exactly (model_1 = white, opening plies
alternate with epsilon=1.0, double-pass ends the game), but snapshots the board
after each ply in a fixed orientation (+1 = white pieces) so a video can be
rendered without the per-ply canonicalization flip.
"""
import random

import numpy as np

from othello.game.board import OthelloBoard
from othello.game.rules import evaluate


def display_board(board):
    """Fixed-orientation snapshot of an OthelloBoard: +1 = white pieces."""
    if board.player:  # white to move: the board is already in white's frame
        return board.board.astype(np.float64)
    return (np.rot90(board.board, k=2) * -1).astype(np.float64)


def _move_delta(before, after):
    """Placed square (None for a pass) and flipped squares between two snapshots."""
    placed = np.argwhere((before == 0) & (after != 0))
    flipped = np.argwhere((before != 0) & (after == -before))
    last = tuple(int(x) for x in placed[0]) if len(placed) == 1 else None
    return last, [(int(r), int(c)) for r, c in flipped]


def play_game(agent_w, agent_b, start_explore=(0, 32), Epsilon_1=0.0, Epsilon_2=0.0,
              boltzmann=False, board=None):
    """Play agent_w (white, moves first) vs agent_b with modelVsModel's rules.

    Returns (result, states): result = [white_won, black_won]; states = list of
    dicts with keys board (fixed-orientation float64), white_to_move, ply,
    last_move, flipped, white_count, black_count, is_opening.
    """
    b = board if board is not None else OthelloBoard(size=8)
    start, finish = start_explore
    n_open = random.randint(start, min(finish, 62))
    turn_1 = True
    ply = 0
    states = []

    def snapshot(is_opening, last_move=None, flipped=None):
        disp = display_board(b)
        states.append(dict(
            board=disp,
            white_to_move=bool(b.player),
            ply=ply,
            last_move=last_move,
            flipped=flipped,
            white_count=int(np.count_nonzero(disp == 1)),
            black_count=int(np.count_nonzero(disp == -1)),
            is_opening=is_opening,
        ))
        return disp

    before = snapshot(False)  # initial position

    for _ in range(n_open):
        b.modelMove(agent_w if turn_1 else agent_b, switching=True, epsilon=1.0, boltzmann=False)
        turn_1 = not turn_1
        ply += 1
        disp = display_board(b)
        before = snapshot(True, *_move_delta(before, disp))

    while True:
        is_one = turn_1
        singal = b.modelMove(OthelloRL=agent_w if is_one else agent_b,
                             epsilon=Epsilon_1 if is_one else Epsilon_2, boltzmann=boltzmann)
        ply += 1
        disp = display_board(b)
        before = snapshot(False, *_move_delta(before, disp))
        if not singal:
            delta = evaluate(board=b.board)
            result = np.array([delta < 0, delta > 0]) if is_one else np.array([delta > 0, delta < 0])
            return result, states
        turn_1 = not turn_1
