"""Generate self-supervised pretraining data with ZERO engine input.

Positions come from two sources, both engine-free:
  - random play: two random movers play full games (the reachable-position
    distribution)
  - random boards: uniformly random stone patterns (out-of-distribution
    states that force the network to learn the raw legality rule)

For every position the shards store the canonical 2-channel model input
(board, under-attack), the legal-move mask, and the game's outcome from
the mover's perspective (from the recorded random-play game itself — the
cheapest honest value target there is). pretrain_ssl.py consumes them.

Labels come from the rules' move generator, which is just the definition
of the game — no minimax engine is ever consulted.

Usage:
  uv run --no-sync python scripts/generate_ssl_positions.py --positions 200000 --out data/ssl
"""
import argparse
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

import numpy as np

from othello import OthelloBoard


def _worker(args):
    """Play random games (+ random boards); return (x, mask, outcome) arrays."""
    n_positions, random_frac, seed_base, games = args
    rng = np.random.default_rng(seed_base)
    xs, masks, outs = [], [], []
    while len(xs) < n_positions:
        if random.random() < random_frac:
            # uniform random board (skip near-empty boards; the current player
            # is fixed to white and the network sees both colors via switching)
            b = rng.integers(-1, 2, size=(8, 8))
            if (b != 0).sum() < 4:
                continue
            board = OthelloBoard(size=8)
            board.board = b.astype(np.float64)
            (x, mask), all_moves = board.getBoard()
            xs.append(x[0].copy())
            masks.append(mask[0].copy())
            outs.append(0.0)  # no outcome for synthetic boards
            continue
        # random-play game: record every ply, then label each position with
        # the game outcome from the mover's perspective (draws are dropped)
        seed = seed_base + games
        np.random.seed(seed)
        random.seed(seed)
        games += 1
        board = OthelloBoard(size=8)
        g_xs, g_masks, g_whites = [], [], []
        while True:
            (x, mask), all_moves = board.getBoard()
            if len(all_moves) == 0:
                if not board.makeMove(helpers=(x, mask), x=None, y=None,
                                      board_change=None, switching=True):
                    break
                continue
            g_xs.append(x[0].copy())
            g_masks.append(mask[0].copy())
            g_whites.append(bool(board.player))
            I, J = random.choice(list(all_moves.keys()))
            if not board.makeMove(helpers=(x, mask), x=int(I), y=int(J),
                                  board_change=all_moves[(int(I), int(J))], switching=True):
                break
        sign = np.sign(board.board.sum())
        if sign == 0:
            continue  # skip draws: no honest outcome label
        white_won = sign > 0
        xs.extend(g_xs)
        masks.extend(g_masks)
        outs.extend(1.0 if white == white_won else -1.0 for white in g_whites)
    xs, masks, outs = xs[:n_positions], masks[:n_positions], outs[:n_positions]
    return (np.array(xs, dtype=np.float32), np.array(masks, dtype=np.float32),
            np.array(outs, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--positions', type=int, default=200000, help='total labeled positions')
    ap.add_argument('--random-boards', type=float, default=0.1,
                    help='fraction of positions from uniform random boards (rest: random play)')
    ap.add_argument('--workers', type=int, default=None,
                    help='parallel processes (default: min(8, cores//2))')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=str, default='data/ssl', help='output directory')
    args = ap.parse_args()

    workers = args.workers or min(8, max(1, (os.cpu_count() or 1) // 2))
    chunk, rem = divmod(args.positions, workers)
    tasks = []
    for w in range(workers):
        n = chunk + (1 if w < rem else 0)
        tasks.append((n, args.random_boards, args.seed + w * chunk, 0))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ctx = mp.get_context('spawn')
    with ctx.Pool(workers) as pool:
        results = pool.map(_worker, tasks)
    dt = time.time() - t0

    total = 0
    for w, (xs, masks, outs) in enumerate(results):
        np.savez_compressed(out_dir / f'positions_{w:02d}.npz',
                            x=xs, mask=masks, outcome=outs)
        total += len(outs)
    print(f'{total} positions in {dt:.1f} s ({total / dt:.0f} positions/s wall-clock, '
          f'{workers} workers).')
    print(f'saved {len(results)} shard(s) to {out_dir}')


if __name__ == '__main__':
    main()
