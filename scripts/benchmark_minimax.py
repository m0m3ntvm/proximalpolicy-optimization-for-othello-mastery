"""Benchmark the standalone minimax engine (old engine baseline vs new engine).

Methodology: midgame positions from seeded random 20-ply openings (epsilon=1.0, so
the engine is never consulted while generating them). Per engine, the thread pool
is spawned with one untimed call and each position is then timed once (cold TT per
position, pool warm) — i.e. the per-move cost a match actually pays.

Baseline for the previous engine (single cold search, seed-0 midgame):

    depth 5  initial 0.005s   midgame 0.141s
    depth 7  initial 0.032s   midgame 3.507s
"""
import random
import time

import torch

from othello.game.board import OthelloBoard
from othello.search.minimax import miniMax

BASELINE = {
    5: (0.005, 0.141),
    7: (0.032, 3.507),
}


def midgame(seed, plies=20):
    random.seed(seed)
    b = OthelloBoard(size=8)
    dummy = miniMax(depth=1, workers=1)  # never consulted (epsilon=1.0)
    for _ in range(plies):
        b.modelMove(dummy, epsilon=1.0, boltzmann=False)
    (x, mask), _ = b.getBoard()
    return torch.from_numpy(x), torch.from_numpy(mask)


def main():
    # warm-up: compile every kernel once (cached on disk for later runs)
    b = OthelloBoard(size=8)
    (x, mask), _ = b.getBoard()
    warm = miniMax(depth=3, workers=1)
    warm(torch.from_numpy(x), torch.from_numpy(mask))

    positions = [midgame(s) for s in range(5)]

    print(f"{'depth':>5} {'serial mid':>12} {'auto mid':>12}  {'baseline mid':>13} {'speedup':>9}")
    for depth in range(3, 9):
        eng_s = miniMax(depth=depth, workers=1)
        eng_a = miniMax(depth=depth)  # default: parallel root only when depth >= parallel_depth

        # untimed first call: pool spawn (auto) + any lazy setup
        eng_s(*positions[0])
        eng_a(*positions[0])

        def avg(engine, pos):
            best = float('inf')
            total = 0.0
            for p in pos:
                t0 = time.perf_counter()
                out = engine(*p)
                dt = time.perf_counter() - t0
                best = min(best, dt)
                total += dt
                if engine is eng_a:
                    assert torch.equal(out, eng_s(*p)), f"serial != default at depth {depth}"
            return best, total / len(pos)

        _, a_s_mid = avg(eng_s, positions[1:])
        _, a_a_mid = avg(eng_a, positions[1:])

        base = BASELINE.get(depth)
        if base:
            speedup = base[1] / a_a_mid
            print(f"{depth:>5} {a_s_mid:>12.4f} {a_a_mid:>12.4f}  {base[1]:>13.3f} {speedup:>8.1f}x")
        else:
            print(f"{depth:>5} {a_s_mid:>12.4f} {a_a_mid:>12.4f}")


if __name__ == "__main__":
    main()
