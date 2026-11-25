"""Generate distillation data: engine-vs-engine games with random openings.

For every ply after the opening, stores the canonical 2-channel model input
(board, under-attack), the legal-move mask, the engine's move label, and the
final game outcome from the mover's perspective — exactly the tensors the
policy/value net consumes, so pretrain_distill.py can train on them directly.

Why depth 7 by default: policy distillation only needs *plentiful decent*
targets — a CNN cannot represent depth-9 exactness, and depth 7 is ~3x
cheaper per move, so volume wins. Override with --depth if preferred.

Usage:
  uv run --no-sync python scripts/generate_expert_games.py --games 10000 --depth 7 --out data/distill
"""
import argparse
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from othello import OthelloBoard, evaluate, miniMax


def _worker(args):
    """Play n_games engine-vs-engine games; return (x, mask, moves, outcomes)."""
    depth, opening_max, n_games, seed_base, threads = args
    torch.set_num_threads(1)
    engine = miniMax(name='E', depth=depth, workers=threads)
    xs, masks, moves, outs = [], [], [], []
    for g in range(n_games):
        seed = seed_base + g
        np.random.seed(seed)
        random.seed(seed)
        whites = []  # per-game: which recorded positions were white to move
        board = OthelloBoard(size=8)
        # random opening plies for position diversity (engine is deterministic)
        for _ in range(random.randint(0, opening_max)):
            if not board.modelMove(engine, epsilon=1.0):
                break
        while True:
            (x, mask), all_moves = board.getBoard()
            if len(all_moves) == 0:
                # pass (or game over): no label for a pass
                if not board.makeMove(helpers=(x, mask), x=None, y=None,
                                      board_change=None, switching=True):
                    break
                continue
            with torch.no_grad():
                out = engine(torch.from_numpy(x), torch.from_numpy(mask)).numpy()
            I, J = np.unravel_index(np.argmax(out.reshape(-1)), out.shape[-2:])
            xs.append(x[0].copy())
            masks.append(mask[0].copy())
            moves.append((int(I), int(J)))
            whites.append(bool(board.player))
            if not board.makeMove(helpers=(x, mask), x=int(I), y=int(J),
                                  board_change=all_moves[(int(I), int(J))], switching=True):
                break
        # terminal: delta is in the non-mover's frame; the last mover won iff delta < 0
        delta = evaluate(board=board.board)
        if delta == 0:
            outcomes = np.zeros(len(whites), dtype=np.float32)
        else:
            white_won = (delta < 0) == whites[-1]
            outcomes = np.where(np.array(whites) == white_won, 1.0, -1.0).astype(np.float32)
        outs.extend(outcomes.tolist())
    return (np.array(xs, dtype=np.float32), np.array(masks, dtype=np.float32),
            np.array(moves, dtype=np.int8), np.array(outs, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--games', type=int, default=10000, help='total games to generate')
    ap.add_argument('--depth', type=int, default=7, help='engine depth (teacher quality)')
    ap.add_argument('--opening-max', type=int, default=32, help='max random opening plies')
    ap.add_argument('--workers', type=int, default=None,
                    help='parallel game processes (default: min(8, cores//2))')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=str, default='data/distill', help='output directory')
    args = ap.parse_args()

    workers = args.workers or min(8, max(1, (os.cpu_count() or 1) // 2))
    threads = max(1, (os.cpu_count() or 1) // workers)
    chunk, rem = divmod(args.games, workers)
    tasks = []
    for w in range(workers):
        n = chunk + (1 if w < rem else 0)
        tasks.append((args.depth, args.opening_max, n, args.seed + w * chunk, threads))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ctx = mp.get_context('spawn')
    with ctx.Pool(workers) as pool:
        results = pool.map(_worker, tasks)
    dt = time.time() - t0

    total_records = 0
    for w, (xs, masks, moves, outs) in enumerate(results):
        np.savez_compressed(out_dir / f'games_{w:02d}.npz',
                            x=xs, mask=masks, move=moves, outcome=outs)
        total_records += len(outs)
    print(f'{args.games} games -> {total_records} labeled positions in {dt / 60:.1f} min '
          f'({args.games / dt:.1f} games/s wall-clock, {workers} workers x {threads} threads).')
    print(f'saved {len(results)} shard(s) to {out_dir}')


if __name__ == '__main__':
    main()
