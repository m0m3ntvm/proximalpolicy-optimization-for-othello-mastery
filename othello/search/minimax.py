"""Numba-jitted alpha-beta minimax engine with a transposition table."""
import numba
import numpy as np
import torch
from numba import njit
from torch import nn

from othello.game.rules import board_hash, evaluate, get_valid_moves

TT_VALUE_TYPE = numba.types.Tuple((
    numba.types.float64,                                        # score
    numba.types.Tuple((numba.types.int64, numba.types.int64)),  # best move
    numba.types.boolean,                                        # maximizing player
    numba.types.int64,                                          # depth searched
))


def make_transposition_table():
    """A typed transposition table suitable for the minimax kernel."""
    return numba.typed.Dict.empty(key_type=numba.types.int64, value_type=TT_VALUE_TYPE)


class miniMax(nn.Module):
    """Minimax engine wrapped as an nn.Module so it plugs into modelMove's interface.

    The transposition table persists across calls (and games): entries are keyed by
    (board, player) and only reused when the stored depth covers the requested one.
    """

    def __init__(self, depth: int = 5, name: str = "Engine", return_val: bool = False) -> None:
        super(miniMax, self).__init__()

        self.depth = depth
        self.name = name
        self.return_val = return_val
        self.transposition_table = make_transposition_table()

    def forward(self, input, mask):
        board = input[0][0].cpu().numpy()

        value, (x, y) = minimax(board, transposition_table=self.transposition_table, depth=self.depth)

        result = torch.zeros((1, 1, board.shape[0], board.shape[1]))
        result[0][0][x][y] = 1.0

        return (result, value / (board.shape[0] * board.shape[1])) if self.return_val else result


@njit
def minimax(board, transposition_table, depth: int = 3, maximizing_player: bool = True, alpha=-np.inf, beta=np.inf):
    if depth == 0 or np.all(board != 0):
        return evaluate(board), (0, 0)

    key = board_hash(board)

    if key in transposition_table:
        score, best_move, maximizing_player_prev, searched_depth = transposition_table.get(key)
        if maximizing_player_prev == maximizing_player and searched_depth >= depth:
            return score, best_move

    best_move = None

    if maximizing_player:
        max_eval = -np.inf
        legal_moves, _, move_board = get_valid_moves(board)
        if np.any(legal_moves):
            for move, conversion_board in move_board.items():
                board_clone = np.copy(board)
                board_clone[move[0]][move[1]] = 1
                board_clone = np.where(conversion_board, 1.0, board_clone)
                eval_score, _ = minimax(board_clone, transposition_table, depth - 1, maximizing_player=False, alpha=alpha, beta=beta)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
        else:
            # no legal move: pass (the opponent moves on the same board)
            max_eval, _ = minimax(board, transposition_table, depth - 1, maximizing_player=False, alpha=alpha, beta=beta)
            best_move = (0, 0)  # sentinel: this player has no move

        transposition_table.update({key: (max_eval, best_move, maximizing_player, depth)})

        return max_eval, best_move
    else:
        min_eval = np.inf
        legal_moves, _, move_board = get_valid_moves(-1.0 * board)
        if np.any(legal_moves):
            for move, conversion_board in move_board.items():
                board_clone = np.copy(board)
                board_clone[move[0]][move[1]] = -1.0
                board_clone = np.where(conversion_board, -1.0, board_clone)
                eval_score, _ = minimax(board_clone, transposition_table, depth - 1, maximizing_player=True, alpha=alpha, beta=beta)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
        else:
            # no legal move: pass (the opponent moves on the same board)
            min_eval, _ = minimax(board, transposition_table, depth - 1, maximizing_player=True, alpha=alpha, beta=beta)
            best_move = (0, 0)  # sentinel: this player has no move

        transposition_table.update({key: (min_eval, best_move, maximizing_player, depth)})

        return min_eval, best_move
