"""Generate trajectory chunks for masked-autoencoding pretraining (no engine).

Plays games with RANDOM movers (zero engine input) and slices each game
into windows of consecutive positions: a random start ply plus the next
few plies "in the future". Every chunk stores, per frame:

  x        (L, 2, 8, 8)  canonical board + under-attack channels
  mask     (L, 1, 8, 8)  legal-move mask
  outcome  (L,)          game outcome from the frame's mover's perspective

pretrain_mae.py then masks the chunks heavily (tube blocks + fully-masked
future frames) and trains an encoder to reconstruct the sequence.

Usage:
  uv run --no-sync python scripts/generate_ssl_chunks.py --chunks 20000 --out data/ssl_chunks
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
    """Play random games and slice chunks; return (x, mask, outcome) arrays."""
    n_chunks, chunk_len, seed_base = args
    rng = np.random.default_rng(seed_base)
    xs, masks, outs = [], [], []
    games = 0
    while len(xs) < n_chunks:
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
        outcomes = np.where(np.array(g_whites) == white_won, 1.0, -1.0)
        P = len(g_xs)
        if P < chunk_len + 1:
            continue  # need at least a few plies ahead of the start
        starts = rng.integers(0, P - chunk_len, size=2)  # 2 random windows per game
        for s in starts:
            xs.append(np.stack(g_xs[s:s + chunk_len]))
            masks.append(np.stack(g_masks[s:s + chunk_len]))
            outs.append(outcomes[s:s + chunk_len].astype(np.float32))
            if len(xs) >= n_chunks:
                break
    xs, masks, outs = xs[:n_chunks], masks[:n_chunks], outs[:n_chunks]
    return (np.array(xs, dtype=np.float32), np.array(masks, dtype=np.float32),
            np.array(outs, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--chunks', type=int, default=20000, help='total chunks to generate')
    ap.add_argument('--chunk-len', type=int, default=8,
                    help='frames per chunk (the "few steps into the future")')
    ap.add_argument('--workers', type=int, default=None,
                    help='parallel processes (default: min(8, cores//2))')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=str, default='data/ssl_chunks', help='output directory')
    args = ap.parse_args()

    workers = args.workers or min(8, max(1, (os.cpu_count() or 1) // 2))
    chunk, rem = divmod(args.chunks, workers)
    tasks = []
    for w in range(workers):
        n = chunk + (1 if w < rem else 0)
        tasks.append((n, args.chunk_len, args.seed + w * chunk))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ctx = mp.get_context('spawn')
    with ctx.Pool(workers) as pool:
        results = pool.map(_worker, tasks)
    dt = time.time() - t0

    total = 0
    for w, (xs, masks, outs) in enumerate(results):
        np.savez_compressed(out_dir / f'chunks_{w:02d}.npz',
                            x=xs, mask=masks, outcome=outs)
        total += len(xs)
    print(f'{total} chunks of {args.chunk_len} frames in {dt:.1f} s '
          f'({total * args.chunk_len / dt:.0f} frames/s wall-clock, {workers} workers).')
    print(f'saved {len(results)} shard(s) to {out_dir}')


if __name__ == '__main__':
    main()
