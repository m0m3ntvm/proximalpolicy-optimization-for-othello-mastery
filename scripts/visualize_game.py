"""Render agent-vs-agent games as videos.

Examples:
  uv run python scripts/visualize_game.py --white ckpt:8 --black minimax:9 \
      --out videos/ckpt8-vs-minimax9.mp4 --seed 42
  uv run python scripts/visualize_game.py --white minimax:9 --black ckpt:8 \
      --games 2 --fps 3 --out videos/mm9-vs-ckpt8.gif
"""
import argparse
import os
import random

from othello.viz import visualize_game


def parse_agent(spec):
    kind, _, value = spec.partition(':')
    if kind == 'ckpt':
        from othello.checkpoint import load_agent
        return load_agent(f"checkpoints/{value}.pth", device='cpu')
    if kind == 'minimax':
        from othello.search.minimax import miniMax
        return miniMax(depth=int(value))
    raise SystemExit(f"unknown agent spec {spec!r} (use ckpt:N or minimax:D)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--white', default='ckpt:8', help="white agent: ckpt:N or minimax:D (default ckpt:8)")
    ap.add_argument('--black', default='minimax:7', help="black agent: ckpt:N or minimax:D (default minimax:7)")
    ap.add_argument('--out', default='videos/game.mp4', help='output path, .mp4 or .gif (default videos/game.mp4)')
    ap.add_argument('--fps', type=float, default=2.0, help='frames per second (default 2)')
    ap.add_argument('--quality', type=int, default=64, help='pixels per board cell (default 64)')
    ap.add_argument('--opening', type=int, nargs=2, default=(0, 32), metavar=('LO', 'HI'),
                    help='random-opening ply range (default 0 32)')
    ap.add_argument('--epsilon-white', type=float, default=0.0)
    ap.add_argument('--epsilon-black', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=None, help='base seed (game g uses seed+g)')
    ap.add_argument('--games', type=int, default=1, help='games to render; outputs get -1, -2, ... suffixes')
    args = ap.parse_args()

    agent_w = parse_agent(args.white)
    agent_b = parse_agent(args.black)

    for g in range(args.games):
        if args.seed is not None:
            random.seed(args.seed + g)  # each game gets its own reproducible opening
        out = args.out
        if args.games > 1:
            stem, ext = os.path.splitext(out)
            out = f"{stem}-{g + 1}{ext}"
        result, path = visualize_game(agent_w, agent_b, out=out, fps=args.fps,
                                      start_explore=tuple(args.opening),
                                      Epsilon_1=args.epsilon_white,
                                      Epsilon_2=args.epsilon_black,
                                      quality=args.quality)
        winner = "white" if result[0] else "black" if result[1] else "draw"
        print(f"game {g + 1}: {winner} -> {path}")


if __name__ == "__main__":
    main()
