"""Evaluate the published SHA-ZA checkpoints (checkpoints/1..8.pth).

Sections:
  1. Sanity: one legal self-play game per checkpoint
  2. Each checkpoint vs a random player
  3. Progression ladder: checkpoint i vs checkpoint i+1, plus first vs last
  4. Each checkpoint vs minimax at --minimax-depth
  5. Last checkpoint vs minimax at --headline-depth

Scoring in play_match() is perspective-correct: +1 pieces always belong to
board.player, and board.player is white.

Usage:
  uv run python scripts/evaluate_checkpoints.py                      # full suite
  uv run python scripts/evaluate_checkpoints.py --checkpoints 1 8    # ckpt 1 & 8 only
  uv run python scripts/evaluate_checkpoints.py --games 4 --minimax-depth 4
"""
import argparse
import os
import random
import time

import numpy as np
import torch

from othello.checkpoint import load_agent
from othello.game.board import OthelloBoard
from othello.game.rules import evaluate
from othello.search.minimax import miniMax

CKPT_DIR = 'checkpoints'
STEPS = {1: 1000, 2: 1950, 3: 2500, 4: 3000, 5: 4000, 6: 4150, 7: 5025, 8: 5725}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def play_match(agent_a, agent_b, device_a=None, device_b=None, opening=(0, 16),
               a_first=True, b_random=False, seed=None):
    """Play one game. Returns 1 if agent_a wins, -1 if agent_b wins, 0 for a draw."""
    if seed is not None:
        random.seed(seed)
    board = OthelloBoard()
    n_open = random.randint(*opening)
    turn_a = a_first  # True => agent_a to move
    nz = np.count_nonzero(board.board)
    calls = 0

    for _ in range(n_open):  # opening phase: both sides play random moves
        board.modelMove(agent_a if turn_a else agent_b, epsilon=1.0,
                        boltzmann=False, device=device_a if turn_a else device_b)
        turn_a = not turn_a
        calls += 1
        new_nz = np.count_nonzero(board.board)
        assert new_nz in (nz, nz + 1), f'illegal board change {nz} -> {new_nz}'
        nz = new_nz

    while True:
        eps = 1.0 if (b_random and not turn_a) else 0.0
        alive = board.modelMove(agent_a if turn_a else agent_b, epsilon=eps,
                                boltzmann=False,
                                device=device_a if turn_a else device_b)
        turn_a = not turn_a
        calls += 1
        if not alive:
            break
        new_nz = np.count_nonzero(board.board)
        assert new_nz in (nz, nz + 1), f'illegal board change {nz} -> {new_nz}'
        nz = new_nz
        assert calls < 100, 'game did not terminate'

    delta = float(evaluate(board.board))
    white_score = delta if board.player else -delta
    if white_score == 0:
        return 0
    return 1 if (white_score > 0) == a_first else -1


def run_series(label, agent_a, agent_b, games, opening=(0, 32), alt_colors=False,
               b_random=False, device_b=DEVICE, seed_base=0):
    wins = draws = losses = 0
    t0 = time.time()
    for g in range(games):
        a_first = (g % 2 == 0) if alt_colors else True
        r = play_match(agent_a, agent_b, device_a=DEVICE, device_b=device_b,
                       opening=opening, a_first=a_first, b_random=b_random,
                       seed=seed_base + g)
        wins += (r == 1)
        draws += (r == 0)
        losses += (r == -1)
        print(f'    [{label}] game {g + 1}/{games}: {r:+d}  '
              f'(elapsed {time.time() - t0:.0f}s)', flush=True)
    played = wins + losses
    rate = 100.0 * wins / played if played else float('nan')
    print(f'  {label:<38} {wins}/{played} wins ({rate:6.2f}%)  draws: {draws}  '
          f'[{time.time() - t0:.0f}s]', flush=True)
    return rate


def main():
    parser = argparse.ArgumentParser(description='Evaluate published SHA-ZA checkpoints.')
    parser.add_argument('--checkpoints', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8],
                        help='checkpoint indices to test (default: all)')
    parser.add_argument('--games', type=int, default=8, help='games per series (vs random / minimax)')
    parser.add_argument('--ladder-games', type=int, default=6, help='games per ladder matchup')
    parser.add_argument('--minimax-depth', type=int, default=5, help='minimax depth for section 4')
    parser.add_argument('--headline-depth', type=int, default=7, help='minimax depth for section 5')
    parser.add_argument('--headline-games', type=int, default=4, help='games for section 5')
    parser.add_argument('--no-sanity', action='store_true', help='skip the self-play sanity games')
    args = parser.parse_args()

    t_start = time.time()
    ckpts = sorted(set(args.checkpoints))
    print(f'device: {DEVICE}')
    print(f'checkpoints to test: {ckpts}')

    agents = {}
    for i in ckpts:
        a = load_agent(os.path.join(CKPT_DIR, f'{i}.pth'), DEVICE)
        a.name = f'ckpt-{i}'
        n = sum(p.numel() for p in a.parameters())
        print(f'  {i}.pth: loaded (channels={a.internal_channels}, res_depth={a.res_depth}, '
              f'params={n})')
        agents[i] = a
    print()

    if not args.no_sanity:
        print('=== 1. Sanity: one legal self-play game per checkpoint ===')
        for i in ckpts:
            t0 = time.time()
            r = play_match(agents[i], agents[i], device_a=DEVICE, device_b=DEVICE,
                           opening=(0, 8), seed=100 + i)
            print(f'  ckpt-{i}: self-play game completed, result={r:+d} '
                  f'({time.time() - t0:.1f}s)', flush=True)
        print()

    print(f'=== 2. Each checkpoint vs random player ({args.games} games, opening (0,32)) ===')
    for i in ckpts:
        run_series(f'vs random', agents[i], agents[i], games=args.games,
                   opening=(0, 32), b_random=True, seed_base=200 + 10 * i)
    print()

    if len(ckpts) > 1:
        print(f'=== 3. Progression ladder ({args.ladder_games} games, alternate colors, opening (0,16)) ===')
        for a, b in zip(ckpts, ckpts[1:]):
            run_series(f'ckpt-{a} vs ckpt-{b}', agents[a], agents[b],
                       games=args.ladder_games, opening=(0, 16), alt_colors=True,
                       seed_base=300 + 10 * a)
        run_series(f'ckpt-{ckpts[0]} vs ckpt-{ckpts[-1]}', agents[ckpts[0]], agents[ckpts[-1]],
                   games=args.ladder_games, opening=(0, 16), alt_colors=True, seed_base=380)
        print()

    print(f'=== 4. Each checkpoint vs minimax depth {args.minimax_depth} '
          f'({args.games} games, opening (0,32)) ===')
    eng = miniMax(depth=args.minimax_depth)
    print('  warming up numba minimax (first call compiles)...', flush=True)
    board0 = OthelloBoard()
    (x0, m0), _ = board0.getBoard()
    eng(torch.from_numpy(x0), torch.from_numpy(m0))
    for i in ckpts:
        run_series(f'vs minimax-{args.minimax_depth}', agents[i], eng,
                   games=args.games, opening=(0, 32), device_b=None,
                   seed_base=400 + 10 * i)
    print()

    print(f'=== 5. Best checkpoint vs minimax depth {args.headline_depth} '
          f'({args.headline_games} games, opening (0,32)) ===')
    eng_h = miniMax(depth=args.headline_depth)
    eng_h(torch.from_numpy(x0), torch.from_numpy(m0))
    run_series(f'ckpt-{ckpts[-1]} vs minimax-{args.headline_depth}', agents[ckpts[-1]],
               eng_h, games=args.headline_games, opening=(0, 32),
               device_b=None, seed_base=500)
    print(f'\nTOTAL WALL TIME: {time.time() - t_start:.0f}s')


if __name__ == '__main__':
    main()
