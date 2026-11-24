"""Play a saved SHA-ZA checkpoint against the minimax engine.

Mirrors Test()/RunTests() from scripts/train.py: each game plays 0-32 random
opening plies (modelVsModel default), colors alternate per game, and draws
are excluded from the win rate — so the numbers are directly comparable
with the training log's "vs E" entries.

Usage:
  uv run --no-sync python scripts/eval_vs_engine.py \
      --path runs/run_20261124_200119/shaza_new.pth --games 125 --depth 9
"""
import argparse
import time

import numpy as np
from tqdm import tqdm

from othello import OthelloBoard, miniMax
from othello.checkpoint import load_agent


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--path', required=True, help='checkpoint (.pth) to evaluate')
    ap.add_argument('--games', type=int, default=125)
    ap.add_argument('--depth', type=int, default=7, help='minimax engine depth')
    ap.add_argument('--opening-max', type=int, default=32,
                    help='max random opening plies (training eval used 32)')
    args = ap.parse_args()

    # CPU, like Test() in train.py: modelMove builds tensors on the agent's device
    agent = load_agent(args.path, 'cpu').eval()
    engine = miniMax(name='E', depth=args.depth)

    score = np.array([0, 0])  # [agent wins, engine wins]; draws add nothing
    t0 = time.time()
    for i in tqdm(range(args.games), desc=f'{agent.name} vs engine depth {args.depth}'):
        Board = OthelloBoard(size=8)
        if i % 2 == 0:
            score += Board.modelVsModel(agent, engine)
        else:
            score += Board.modelVsModel(engine, agent)[::-1]
    dt = time.time() - t0

    wins, losses = int(score[0]), int(score[1])
    draws = args.games - wins - losses
    rate = wins / (wins + losses) * 100 if wins + losses else float('nan')
    print(f'\n{agent.name} vs minimax depth {args.depth}: {wins}W / {losses}L / {draws}D '
          f'({rate:.4f}% win rate, draws excluded) over {args.games} games '
          f'in {dt / 60:.1f} min ({dt / args.games:.1f} s/game).')


if __name__ == '__main__':
    main()
