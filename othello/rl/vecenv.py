"""Vectorized self-play: many games in lockstep with one batched model forward per ply.

Each game is the exact state machine of ``OthelloBoard.selfPlay``: canonicalized
boards (+1 = player to move), random openings (then forgotten), epsilon /
boltzmann move selection, passes, double-pass and full-board termination, and
the same gamma-scaled, winner-signed returns.

Every game has its own pair of RNG streams (python ``random.Random`` and a
numpy ``Generator``), derived from a per-game ``SeedSequence`` spawn. Games
therefore do not depend on each other or on the batch size: for the same seed,
``batchedSelfPlay(games=8, batch=2)`` and ``batch=8`` produce identical games
(pinned by tests). A lockstep loop over a shared global stream could not do
this — interleaved draws would change the bits each game sees.

The point is throughput: N games share one forward per ply, amortizing the
per-ply torch/GPU launch and transfer overhead ~N-fold. Measured ~35x per-game
on an idle RTX 3050 laptop at batch 32 (single 8x8 forwards never let the GPU
leave its idle power state; a batch does).
"""
import random

import numpy as np
import torch
from numba import njit

from othello.game.rules import decide, get_valid_moves


@njit(cache=True)
def _vec_movegen(boards, active):
    """Legal-move masks and under-attack boards for every active game.

    Same semantics as ``game.rules.get_valid_moves`` per board: mask[g, r, c]
    is True where game g's +1 player can move; under[g] is the union of all
    pieces capturable in one move.
    """
    n_games, n, _ = boards.shape
    mask = np.zeros((n_games, n, n), np.bool_)
    under = np.zeros((n_games, n, n), np.bool_)
    for g in range(n_games):
        if not active[g]:
            continue
        for r in range(n):
            for c in range(n):
                if boards[g, r, c] != 0:
                    continue
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if not (0 <= rr < n and 0 <= cc < n):
                            continue
                        if boards[g, rr, cc] != -1:
                            continue
                        rr, cc = rr + dr, cc + dc
                        while 0 <= rr < n and 0 <= cc < n and boards[g, rr, cc] == -1:
                            rr, cc = rr + dr, cc + dc
                        if 0 <= rr < n and 0 <= cc < n and boards[g, rr, cc] == 1:
                            mask[g, r, c] = True
                            fr, fc = r + dr, c + dc
                            while fr != rr or fc != cc:
                                under[g, fr, fc] = True
                                fr, fc = fr + dr, fc + dc
    return mask, under


@njit(cache=True)
def _vec_apply(boards, moves, active):
    """Play moves[g] (flat row-major index, or -1 for a pass) for each active game."""
    n_games, n, _ = boards.shape
    for g in range(n_games):
        if not active[g] or moves[g] < 0:
            continue
        r, c = moves[g] // n, moves[g] % n
        boards[g, r, c] = 1
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if not (0 <= rr < n and 0 <= cc < n):
                    continue
                if boards[g, rr, cc] != -1:
                    continue
                end_r, end_c = rr, cc
                while 0 <= end_r < n and 0 <= end_c < n and boards[g, end_r, end_c] == -1:
                    end_r, end_c = end_r + dr, end_c + dc
                if 0 <= end_r < n and 0 <= end_c < n and boards[g, end_r, end_c] == 1:
                    while rr != end_r or cc != end_c:
                        boards[g, rr, cc] = 1
                        rr, cc = rr + dr, cc + dc


class VectorSelfPlay:
    """Lockstep self-play over `n_games` boards; one batched forward per ply.

    Mirrors OthelloBoard.selfPlay per game and emits the same
    (state, mask, action, return) transition format.
    """

    def __init__(self, n_games, model, device, size=8, epsilon=0.0, boltzmann=True,
                 start_explore=(0, 32), game_seeds=None):
        self.n_games = n_games
        self.n = size
        self.gamma = np.exp(np.log(0.01) / (size ** 2 - 4))
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.boltzmann = boltzmann
        self.start_explore = start_explore

        self.boards = np.zeros((n_games, size, size), dtype=np.short)
        for g in range(n_games):
            self.boards[g, size // 2 - 1, size // 2 - 1] = 1
            self.boards[g, size // 2, size // 2] = 1
            self.boards[g, size // 2 - 1, size // 2] = -1
            self.boards[g, size // 2, size // 2 - 1] = -1
        self.players = np.ones(n_games, dtype=bool)  # True = white to move
        self.last_moved = np.ones(n_games, dtype=bool)
        self.active = np.ones(n_games, dtype=bool)

        # per-game independent RNG streams: games don't depend on each other
        # or on the batch size; game_seeds[g] is a SeedSequence (or None for entropy)
        if game_seeds is None:
            game_seeds = np.random.SeedSequence().spawn(n_games)
        else:
            game_seeds = [s if s is not None else np.random.SeedSequence()
                          for s in game_seeds]
        self.rng_py = [random.Random(int(s.generate_state(1, dtype=np.uint32)[0]))
                       for s in game_seeds]
        self.rng_np = [np.random.Generator(np.random.PCG64(s)) for s in game_seeds]

        # per-game transition buffers (white entries first at game end, as in selfPlay)
        self.states = [[] for _ in range(n_games)]
        self.masks = [[] for _ in range(n_games)]
        self.actions = [[] for _ in range(n_games)]
        self.scales = [[] for _ in range(n_games)]
        self.is_white = [[] for _ in range(n_games)]
        self.data = []
        self.data_by_game = [[] for _ in range(n_games)]

    def _canonicalize(self, g):
        b = self.boards[g]
        self.boards[g] = b[::-1, ::-1]  # rot180 view of the same buffer
        self.boards[g] *= -1            # negate in place

    def _pass(self, g):
        """Mirror OthelloBoard.makeMove's pass branch. Returns False when the
        game ends: the second consecutive pass, or any pass on a full board."""
        if not self.last_moved[g] or (self.boards[g] != 0).all():
            self.players[g] = not self.players[g]
            self._canonicalize(g)
            return False
        self.last_moved[g] = False
        self.players[g] = not self.players[g]
        self._canonicalize(g)
        return True

    def _openings(self):
        lo, hi = self.start_explore
        for g in range(self.n_games):
            depth = self.rng_py[g].randint(lo, min(hi, 32))
            for _ in range(depth):
                mask, _, moves = get_valid_moves(self.boards[g])
                if len(moves) == 0:
                    # sequential loop ignores the game-ending return here too
                    self._pass(g)
                else:
                    self.rng_np[g].random()  # the epsilon branch draws before choosing
                    (r, c) = self.rng_py[g].choice(list(moves.keys()))
                    self.boards[g][r, c] = 1
                    self.boards[g][moves[(r, c)]] = 1
                    self.last_moved[g] = True
                    self.players[g] = not self.players[g]
                    self._canonicalize(g)

    def _finish(self, g):
        """Score the game and emit transitions (white plies first, like selfPlay)."""
        delta = float(self.boards[g].sum())
        winner = bool(self.players[g]) if delta > 0 else (not bool(self.players[g]) if delta < 0 else -1)
        white, black = [], []
        for i in range(len(self.actions[g])):
            scale = np.float32(self.scales[g][i])
            if self.is_white[g][i]:
                ret = 0.0 if winner == -1 else scale if winner else -1 * scale
            else:
                ret = 0.0 if winner == -1 else -1 * scale if winner else scale
            entry = ((self.states[g][i], self.masks[g][i]), self.actions[g][i], ret)
            (white if self.is_white[g][i] else black).append(entry)
        self.data_by_game[g] = white + black
        self.data += white + black

    def run(self):
        self._openings()
        n = self.n
        while self.active.any():
            mask, under = _vec_movegen(self.boards, self.active)
            legal = [np.argwhere(mask[g]).tolist() for g in range(self.n_games)]
            chosen = np.full(self.n_games, -1, dtype=np.int64)
            model_games = []

            for g in range(self.n_games):
                if not self.active[g]:
                    continue
                if not legal[g]:
                    if not self._pass(g):
                        self.active[g] = False
                        self._finish(g)
                    continue
                if self.rng_np[g].random() >= self.epsilon:
                    model_games.append(g)
                else:
                    r, c = self.rng_py[g].choice(legal[g])
                    chosen[g] = r * n + c

            if model_games:
                m = len(model_games)
                x = np.empty((m, 2, n, n), dtype=np.float32)
                mk = np.empty((m, 1, n, n), dtype=np.float32)
                for i, g in enumerate(model_games):
                    x[i, 0] = self.boards[g]
                    x[i, 1] = under[g]
                    mk[i, 0] = mask[g]
                with torch.no_grad():
                    xt = torch.from_numpy(x)
                    mt = torch.from_numpy(mk)
                    if self.device:
                        xt, mt = xt.to(self.device), mt.to(self.device)
                    out = self.model(xt, mt)
                    probs = out.cpu().numpy() if self.device else out.numpy()
                for i, g in enumerate(model_games):
                    p = np.nan_to_num(probs[i, 0], nan=0.0)
                    if p.sum() > 0:
                        if self.boltzmann:
                            r, c = decide(p, self.rng_np[g])
                        else:
                            r, c = divmod(int(p.argmax()), n)
                    else:
                        r, c = None, None
                    if r is None or [r, c] not in legal[g]:
                        print(f"WARNING: model {getattr(self.model, 'name', self.model)} proposed "
                              f"illegal move {(r, c)}; playing a random legal move.")
                        r, c = self.rng_py[g].choice(legal[g])
                    chosen[g] = r * n + c

            # record transitions (pre-move state, as in saveBoard), then apply
            for g in range(self.n_games):
                if not self.active[g] or chosen[g] < 0:
                    continue
                state = np.empty((2, n, n), dtype=np.float32)
                state[0] = self.boards[g]
                state[1] = under[g]
                self.states[g].append(state)
                self.masks[g].append(mask[g].astype(np.float32)[None])
                action = np.zeros((1, n, n), dtype=np.float32)
                r, c = divmod(chosen[g], n)
                action[0, r, c] = 1.0
                self.actions[g].append(action)
                self.scales[g].append(self.gamma ** np.count_nonzero(self.boards[g] == 0))
                self.is_white[g].append(bool(self.players[g]))

            _vec_apply(self.boards, chosen, self.active)
            for g in range(self.n_games):
                if not self.active[g] or chosen[g] < 0:
                    continue
                self.last_moved[g] = True
                self.players[g] = not self.players[g]
                self._canonicalize(g)
                if (self.boards[g] != 0).all():
                    self.active[g] = False
                    self._finish(g)


def batchedSelfPlay(model, device, games, Epsilon=0.0, boltzmann=False, board_size=8,
                    batch=64, start_explore=(0, 32), seed=None):
    """Play `games` self-play games in lockstep batches of `batch` games.

    Returns the same ((state, mask), action, return) list format as
    parallelSelfPlay, so it drops into the training loop unchanged. With
    `seed` set, the run is deterministic and independent of the batch size.
    """
    getattr(model, 'eval', lambda: None)()
    if hasattr(model, 'return_val'):
        model.return_val = False
    seeds = np.random.SeedSequence(seed).spawn(games) if seed is not None else [None] * games
    data = []
    for first in range(0, games, batch):
        env = VectorSelfPlay(min(batch, games - first), model, device, size=board_size,
                             epsilon=Epsilon, boltzmann=boltzmann,
                             start_explore=start_explore,
                             game_seeds=seeds[first:first + batch])
        env.run()
        data += env.data
    return data
