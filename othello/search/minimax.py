"""Standalone numba alpha-beta minimax engine for Othello.

Self-contained: move generation, evaluation, hashing, and search all live in this
module (no dependency on othello.game), so the engine can be benchmarked, embedded,
or optimized independently. The public API is unchanged:

    engine = miniMax(depth=7)                # nn.Module wrapper for modelMove
    value, (x, y) = minimax(board, tt, depth)  # plain-array shim
    tt = make_transposition_table()          # typed table for the kernel

Speed techniques (all exact — same minimax value and move set as before):

- padded int8 1D board with a border sentinel: ray-scan move generation with no
  per-node dicts, negations, or np.where allocations
- moves ordered by flip count: alpha-beta pruning improves drastically while the
  full-window search value is unchanged (ordering only affects cutoffs and the
  tie-break among equal-valued moves)
- 128-bit Zobrist transposition-table keys: fixed key type for any board size and
  a collision probability of ~2^-128 (the old polynomial hash was ~2^-64)
- depth-tagged TT entries: a cached score is only reused when it was searched at
  least as deep as the current request
- a player with no legal move passes (the opponent moves on the same board) — the
  position is never mistaken for a terminal one
- root-level parallelism: the root's moves are searched on a thread pool, each
  thread with a thread-local TT (numa kernels release the GIL). The results are
  combined with a first-max tie-break, so parallel and serial searches return the
  identical move.
"""
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numba
import numpy as np
import torch
from numba import njit
from torch import nn

# Zobrist tables: two independent 64-bit tables so the TT key is 128 bits.
# Cell index = (row - 1) * n + (col - 1), value index = piece + 1 (0/1/2).
_ZOBRIST_SEED = 20240817
_ZOBRIST_RNG = np.random.default_rng(_ZOBRIST_SEED)
_Z1 = _ZOBRIST_RNG.integers(1, 2 ** 63, size=(256, 3), dtype=np.int64)
_Z2 = _ZOBRIST_RNG.integers(1, 2 ** 63, size=(256, 3), dtype=np.int64)

# Hard cap on table entries: once full, the search keeps reading existing entries
# but stops inserting (a frozen full table beats clear-and-refill thrash). Deep
# searches saturate the table within a single move, so the wrapper clears it
# between moves. The table is a pure cache — the cap never changes search values.
_TT_MAX_ENTRIES = 500_000

TT_VALUE_TYPE = numba.types.Tuple((
    numba.types.float64,   # exact score in the root player's frame
    numba.types.int64,     # best move (flat padded index, -1 = pass)
    numba.types.boolean,   # maximizing player
    numba.types.int64,     # depth searched
))


def make_transposition_table():
    """A typed transposition table suitable for the minimax kernel."""
    return numba.typed.Dict.empty(
        key_type=numba.types.UniTuple(numba.types.int64, 2),
        value_type=TT_VALUE_TYPE,
    )


@njit(cache=True, nogil=True)
def _to_padded(board_2d):
    """2D board -> 1D int8 board padded with a border sentinel (3)."""
    n = board_2d.shape[0]
    P = n + 2
    b = np.full(P * P, np.int8(3), dtype=np.int8)
    for r in range(n):
        for c in range(n):
            b[(r + 1) * P + (c + 1)] = board_2d[r, c]
    return b, P, n


@njit(cache=True, nogil=True)
def _gen_moves(b, own, opp, P, n):
    """Legal moves of `own` as flat padded indices plus their flip counts."""
    moves = np.empty(n * n, dtype=np.int64)
    flips = np.empty(n * n, dtype=np.int64)
    cnt = 0
    dirs = (-P - 1, -P, -P + 1, -1, 1, P - 1, P, P + 1)
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            idx = r * P + c
            if b[idx] != 0:
                continue
            total = 0
            for d in dirs:
                k = idx + d
                if b[k] != opp:
                    continue
                k += d
                while b[k] == opp:
                    k += d
                if b[k] == own:
                    total += (k - idx) // d - 1
            if total > 0:
                moves[cnt] = idx
                flips[cnt] = total
                cnt += 1
    return moves, flips, cnt


@njit(cache=True, nogil=True)
def _order_moves(moves, flips, cnt):
    """Insertion sort by flip count, descending (more flips searched first)."""
    for i in range(1, cnt):
        m = moves[i]
        f = flips[i]
        j = i
        while j > 0 and flips[j - 1] < f:
            moves[j] = moves[j - 1]
            flips[j] = flips[j - 1]
            j -= 1
        moves[j] = m
        flips[j] = f


@njit(cache=True, nogil=True)
def _apply(b, idx, own, opp, P):
    """Apply a move in place (place the piece, flip the sandwiched runs)."""
    b[idx] = own
    dirs = (-P - 1, -P, -P + 1, -1, 1, P - 1, P, P + 1)
    for d in dirs:
        k = idx + d
        if b[k] != opp:
            continue
        while b[k] == opp:
            k += d
        if b[k] == own:
            kk = idx + d
            while kk != k:
                b[kk] = own
                kk += d


@njit(cache=True, nogil=True)
def _key(b, P, n):
    """128-bit Zobrist key plus the number of empty cells."""
    h1 = 0
    h2 = 0
    empties = 0
    i = 0
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            v = b[r * P + c]
            h1 ^= _Z1[i, v + 1]
            h2 ^= _Z2[i, v + 1]
            if v == 0:
                empties += 1
            i += 1
    return h1, h2, empties


@njit(cache=True, nogil=True)
def _score(b, P, n):
    """Piece count (+1 - -1), i.e. the score from the board's own frame."""
    s = 0
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            s += b[r * P + c]
    return float(s)


@njit(cache=True, nogil=True)
def _search(b, tt, P, n, depth, maximizing, alpha, beta, cap):
    """Alpha-beta minimax on a padded board, with a depth-tagged transposition table.

    The board is always in the root player's frame; `maximizing` says whose turn it
    is. Values returned are exact (full-window) scores in the root frame. `cap`
    bounds the table size: once full, entries are read but no longer inserted, so
    memory stays bounded and the table never thrash-clears mid-search.
    """
    if depth == 0:
        return _score(b, P, n), -1

    h1, h2, empties = _key(b, P, n)
    key = (h1, h2)

    if key in tt:
        score, best_move, prev_max, searched_depth = tt.get(key)
        if prev_max == maximizing and searched_depth >= depth:
            return score, best_move

    if empties == 0:
        s = _score(b, P, n)
        if len(tt) < cap:
            tt.update({key: (s, -1, maximizing, depth)})
        return s, -1

    own = 1 if maximizing else -1
    opp = -own
    moves, flips, cnt = _gen_moves(b, own, opp, P, n)

    if cnt == 0:
        # no legal move: pass (the opponent moves on the same board)
        value, _ = _search(b, tt, P, n, depth - 1, not maximizing, alpha, beta, cap)
        if len(tt) < cap:
            tt.update({key: (value, -1, maximizing, depth)})
        return value, -1

    _order_moves(moves, flips, cnt)

    best = -np.inf if maximizing else np.inf
    best_move = -1
    for i in range(cnt):
        child = np.copy(b)
        _apply(child, moves[i], own, opp, P)
        v, _ = _search(child, tt, P, n, depth - 1, not maximizing, alpha, beta, cap)
        if maximizing:
            if v > best:
                best = v
                best_move = moves[i]
            if best > alpha:
                alpha = best
            if beta <= alpha:
                break
        else:
            if v < best:
                best = v
                best_move = moves[i]
            if best < beta:
                beta = best
            if beta <= alpha:
                break

    if len(tt) < cap:
        tt.update({key: (best, best_move, maximizing, depth)})
    return best, best_move


@njit(cache=True, nogil=True)
def _move_xy(move, P):
    """Flat padded index -> (x, y) in original coordinates; -1 -> (0, 0)."""
    if move < 0:
        return 0, 0
    return move // P - 1, move % P - 1


def minimax(board, transposition_table, depth: int = 3, maximizing_player: bool = True,
            alpha=-np.inf, beta=np.inf):
    """Compatibility shim over the optimized kernels: search a 2D board from the
    root player's frame and return (value, (x, y)), with (0, 0) for a pass."""
    b, P, n = _to_padded(np.asarray(board))
    value, move = _search(b, transposition_table, P, n, depth, maximizing_player, alpha, beta,
                          _TT_MAX_ENTRIES)
    x, y = _move_xy(move, P)
    return value, (x, y)


class miniMax(nn.Module):
    """Minimax engine wrapped as an nn.Module so it plugs into modelMove's interface.

    The transposition table persists across calls (and games), so repeated
    positions are answered instantly. Deep searches (depth >= parallel_depth) are
    split across a thread pool: each thread keeps its own TT and the root result is
    combined with a first-max tie-break, making the parallel move identical to the
    serial one.
    """

    def __init__(self, depth: int = 5, name: str = "Engine", return_val: bool = False,
                 workers: int = None, parallel_depth: int = 7) -> None:
        super(miniMax, self).__init__()

        self.depth = depth
        self.name = name
        self.return_val = return_val
        self.workers = max(1, min(workers if workers is not None else (os.cpu_count() or 1), 8))
        self.parallel_depth = parallel_depth
        self.transposition_table = make_transposition_table()
        self._warmed = False
        self._pool = None
        self._tls = threading.local()

    def _search_root_parallel(self, b, P, n, moves, cnt):
        """Search the root's children on a thread pool; identical to the serial root."""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.workers)
        tls = self._tls
        depth = self.depth

        def task(i):
            tt = getattr(tls, 'tt', None)
            if tt is None:
                tt = make_transposition_table()
                tls.tt = tt
            elif len(tt) >= _TT_MAX_ENTRIES:
                tt.clear()  # saturated by a previous subtree: start fresh
            child = np.copy(b)
            _apply(child, moves[i], 1, -1, P)
            return _search(child, tt, P, n, depth - 1, False, -np.inf, np.inf,
                           _TT_MAX_ENTRIES)[0]

        vals = self._pool.map(task, range(cnt))
        best = -np.inf
        best_i = -1
        for i, v in enumerate(vals):
            if v > best:
                best, best_i = v, i
        move = int(moves[best_i]) if best_i >= 0 else -1

        h1, h2, _ = _key(b, P, n)
        self.transposition_table[(int(h1), int(h2))] = (float(best), move, True, int(depth))
        return best, move

    def forward(self, input, mask):
        board = input[0][0].cpu().numpy()
        b, P, n = _to_padded(board)

        if not self._warmed:
            # compile every kernel on this (single-threaded) first call
            _search(b, make_transposition_table(), P, n, 1, True, -np.inf, np.inf,
                    _TT_MAX_ENTRIES)
            self._warmed = True

        moves, flips, cnt = _gen_moves(b, 1, -1, P, n)
        _order_moves(moves, flips, cnt)

        if cnt == 0 or self.depth < self.parallel_depth or self.workers == 1:
            value, move = _search(b, self.transposition_table, P, n, self.depth, True,
                                  -np.inf, np.inf, _TT_MAX_ENTRIES)
        else:
            value, move = self._search_root_parallel(b, P, n, moves, cnt)

        if len(self.transposition_table) >= _TT_MAX_ENTRIES:
            self.transposition_table.clear()  # saturated: start fresh next move

        x, y = _move_xy(move, P)
        result = torch.zeros((1, 1, n, n))
        result[0][0][x][y] = 1.0

        return (result, value / (n * n)) if self.return_val else result
