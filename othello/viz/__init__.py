"""Game visualization engine: record agent games and render them as videos.

    result, path = visualize_game(agent_w, agent_b, out="videos/game.mp4", seed=42)

Plays a full game between any two agents (RL checkpoints, minimax engines, or
anything with modelMove's interface), snapshots every ply in a fixed board
orientation, renders an animated video (title card, pop-in moves, coin-flip
flips, score panel, winner overlay) and writes an MP4 (imageio-ffmpeg) or
animated GIF (Pillow).
"""
import os
import random

from othello.viz.record import display_board, play_game
from othello.viz.render import render_animation, render_frame, render_game
from othello.viz.writer import write_gif, write_video

__all__ = ["display_board", "play_game", "render_frame", "render_game",
           "render_animation", "write_gif", "write_video", "visualize_game"]


def visualize_game(agent_w, agent_b, out="videos/game.mp4", fps=12, interp=6,
                   start_explore=(0, 32), Epsilon_1=0.0, Epsilon_2=0.0,
                   boltzmann=False, board=None, quality=64, seed=None):
    """Play agent_w (white) vs agent_b and write the game as an animated video.

    Returns (result, path) with result = [white_won, black_won]. fps is the
    output frame rate and interp the number of interpolated frames per ply
    (use small values, e.g. fps=4/interp=2, for GIFs to keep them compact).
    The final frame is held ~2s with a fading winner overlay.
    """
    if seed is not None:
        random.seed(seed)

    result, states = play_game(agent_w, agent_b, start_explore=start_explore,
                               Epsilon_1=Epsilon_1, Epsilon_2=Epsilon_2,
                               boltzmann=boltzmann, board=board)

    name_w = getattr(agent_w, 'name', agent_w.__class__.__name__)
    name_b = getattr(agent_b, 'name', agent_b.__class__.__name__)
    winner = 1 if result[0] else -1 if result[1] else 0
    frames = render_animation(states, fps=fps, interp=interp, quality=quality,
                              title=f"{name_w} (White)  vs  {name_b} (Black)",
                              winner=winner)

    out_dir = os.path.dirname(os.path.abspath(out))
    os.makedirs(out_dir, exist_ok=True)
    write_video(frames, out, fps=fps)
    return result, out
