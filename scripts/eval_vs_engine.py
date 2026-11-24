"""Play a saved SHA-ZA checkpoint against the minimax engine.

Mirrors Test()/RunTests() from scripts/train.py: each game plays 0-32 random
opening plies (modelVsModel default), colors alternate per game, and draws
are excluded from the win rate — so the numbers are directly comparable
with the training log's "vs E" entries.

--workers N runs N games in parallel (one process per game chunk, each with
its own engine and checkpoint copy). Engine search threads are auto-scaled
to max(1, cores // workers) so the machine is fully used without
oversubscription (override with --engine-workers).

Usage:
  uv run --no-sync python scripts/eval_vs_engine.py \
      --path runs/run_20261124_200119/shaza_new.pth --games 125 --depth 9
  uv run --no-sync python scripts/eval_vs_engine.py \
      --path runs/run_20261124_200119/shaza_new.pth --games 125 --depth 9 --workers 8
"""
import argparse
import os
import time

import numpy as np

from othello import OthelloBoard, miniMax
from othello.checkpoint import load_agent


def _worker(args):
    """Play games [start, end) of the match; returns [wins, losses]."""
    path, depth, start, end, engine_workers = args
    import torch  # per-process (Windows spawn)
    torch.set_num_threads(1)
    agent = load_agent(path, 'cpu').eval()
    engine = miniMax(name='E', depth=depth, workers=engine_workers)
    score = np.array([0, 0])
    t0 = time.time()
    for i in range(start, end):
        Board = OthelloBoard(size=8)
        if i % 2 == 0:
            score += Board.modelVsModel(agent, engine)
        else:
            score += Board.modelVsModel(engine, agent)[::-1]
    wins, losses = int(score[0]), int(score[1])
    print(f'  worker: games {start + 1}-{end} -> {wins}W/{losses}L/'
          f'{end - start - wins - losses}D  [{time.time() - t0:.0f}s]', flush=True)
    return wins, losses


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--path', required=True, help='checkpoint (.pth) to evaluate')
    ap.add_argument('--games', type=int, default=125)
    ap.add_argument('--depth', type=int, default=7, help='minimax engine depth')
    ap.add_argument('--opening-max', type=int, default=32,
                    help='max random opening plies (training eval used 32)')
    ap.add_argument('--workers', type=int, default=1,
                    help='parallel game processes (1 = sequential)')
    ap.add_argument('--engine-workers', type=int, default=None,
                    help='search threads per engine (default: cores // --workers)')
    args = ap.parse_args()

    t0 = time.time()
    workers = max(1, args.workers)
    engine_workers = args.engine_workers or max(1, (os.cpu_count() or 1) // workers)

    if workers == 1:
        from tqdm import tqdm
        agent = load_agent(args.path, 'cpu').eval()
        engine = miniMax(name='E', depth=args.depth, workers=engine_workers)
        score = np.array([0, 0])
        for i in tqdm(range(args.games), desc=f'{agent.name} vs engine depth {args.depth}'):
            Board = OthelloBoard(size=8)
            if i % 2 == 0:
                score += Board.modelVsModel(agent, engine)
            else:
                score += Board.modelVsModel(engine, agent)[::-1]
        wins, losses = int(score[0]), int(score[1])
    else:
        import multiprocessing as mp
        chunk, rem = divmod(args.games, workers)
        tasks = []
        start = 0
        for w in range(workers):
            n = chunk + (1 if w < rem else 0)
            tasks.append((args.path, args.depth, start, start + n, engine_workers))
            start += n
        print(f'running {workers} workers x {engine_workers} engine threads...', flush=True)
        with mp.Pool(workers) as pool:
            results = pool.map(_worker, tasks)
        wins = sum(r[0] for r in results)
        losses = sum(r[1] for r in results)

    dt = time.time() - t0
    draws = args.games - wins - losses
    rate = wins / (wins + losses) * 100 if wins + losses else float('nan')
    print(f'\ncheckpoint vs minimax depth {args.depth}: {wins}W / {losses}L / {draws}D '
          f'({rate:.4f}% win rate, draws excluded) over {args.games} games '
          f'in {dt / 60:.1f} min ({dt / args.games:.1f} s/game, {args.games / dt:.1f} games/s '
          f'wall-clock).')


if __name__ == '__main__':
    main()
