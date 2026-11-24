"""Environment / self-play throughput benchmarks.

  uv run python scripts/benchmark_env.py               # full comparison
  uv run python scripts/benchmark_env.py --quick       # fewer games

Compares:
  1. the sequential environment's transition cost (dummy policy, no torch)
  2. classic per-process self-play (parallelSelfPlay) vs the vectorized
     lockstep environment (batchedSelfPlay) with the real model on the GPU
"""
import argparse
import random
import time

import numpy as np
import torch


def env_only_bench():
    """Sequential env cost per ply with a trivial model (no torch in the loop)."""
    from othello.game.board import OthelloBoard

    class Dummy:
        name = 'dummy'

        def __call__(self, x, mask):
            return torch.from_numpy(np.asarray(mask, dtype=np.float32))

    random.seed(1)
    # warm up: the first games pay numba compilation, not steady-state env cost
    for _ in range(2):
        b = OthelloBoard()
        b.selfPlay(Dummy(), start_explore=(0, 32), epsilon=0.0, boltzmann=True)
    t0 = time.perf_counter()
    games = 40
    plies = 0
    for _ in range(games):
        b = OthelloBoard()
        _, data = b.selfPlay(Dummy(), start_explore=(0, 32), epsilon=0.0,
                             boltzmann=True)
        plies += len(data)
    dt = time.perf_counter() - t0
    print(f'[env-only]  sequential selfPlay: {games} games, {plies} plies, '
          f'{dt / plies * 1e6:.1f} us/ply')


def selfplay_bench(games, batch, workers):
    from othello import OthelloRL
    from othello.rl.selfplay import parallelSelfPlay
    from othello.rl.vecenv import batchedSelfPlay

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OthelloRL(internal_channels=64, res_depth=8, name='bench',
                      return_val=False).to(device).eval()
    print(f'device: {device} ({torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}), '
          f'games: {games}')

    # warm up (numba kernel compilation + torch cudnn warm-up)
    batchedSelfPlay(model, device, 2, Epsilon=0.05, boltzmann=True, batch=2)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    if batch > 0:
        t0 = time.perf_counter()
        data = batchedSelfPlay(model, device, games, Epsilon=0.05, boltzmann=True,
                               batch=batch)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f'[batched]   batch={batch}: {dt:.1f}s -> {games / dt:.1f} games/s, '
              f'{len(data) / dt:.0f} transitions/s')
    else:
        t0 = time.perf_counter()
        data = parallelSelfPlay(model, device, workers=workers, Epsilon=0.05,
                                boltzmann=True, game_per_worker=games // workers)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f'[parallel]  workers={workers}: {dt:.1f}s -> {games / dt:.1f} games/s, '
              f'{len(data) / dt:.0f} transitions/s')
    return len(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quick', action='store_true', help='fewer games')
    ap.add_argument('--games', type=int, default=192)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    env_only_bench()
    games = max(32, args.games // 4) if args.quick else args.games

    print()
    selfplay_bench(games, batch=0, workers=args.workers)  # classic per-process
    print()
    selfplay_bench(games, batch=args.batch, workers=args.workers)  # vectorized


if __name__ == '__main__':
    main()
