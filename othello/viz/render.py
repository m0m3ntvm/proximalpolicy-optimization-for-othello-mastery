"""Frame rendering for game videos: board, markers, and a score panel (PIL + numpy).

One frame per recorded ply. Pieces flipped by the last move flash gold, the last
move is marked with a red ring, legal moves for the side to move are hinted, and
a right-side panel shows the title, ply, piece-count bars, and who is to move.
The winner overlay is drawn on the final frame.
"""
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from othello.game.rules import get_valid_moves

_BG = (238, 238, 210)      # cream board, matching the repo's drawBoard look
_GRID = (92, 64, 51)       # brown grid
_PANEL = (44, 44, 44)      # dark score panel
_TEXT = (235, 235, 235)
_MUTED = (170, 170, 170)
_WHITE = (255, 255, 255)
_WHITE_RIM = (150, 150, 150)
_BLACK = (20, 20, 20)
_HINT = (105, 105, 95)
_FLIP = (240, 190, 60)     # pieces flipped by the last move flash gold
_LAST = (205, 45, 45)      # red ring on the last move

_fonts = {}


def _font(size):
    if size not in _fonts:
        font = None
        for path in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"):
            try:
                font = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        _fonts[size] = font or ImageFont.load_default()
    return _fonts[size]


def _circle_masks(q):
    yy, xx = np.mgrid[0:q, 0:q]
    rr = (xx - q / 2.0) ** 2 + (yy - q / 2.0) ** 2
    return {
        'disc': rr <= (0.36 * q) ** 2,
        'rim': (rr <= (0.39 * q) ** 2) & (rr > (0.36 * q) ** 2),
        'hint': rr <= (0.13 * q) ** 2,
        'ring': (rr <= (0.47 * q) ** 2) & (rr > (0.42 * q) ** 2),
    }


def render_frame(state, quality=64, title=None, winner=0):
    """Render one recorded state as a uint8 RGB image (board + score panel).

    state: dict from play_game. winner: 1 = white, -1 = black, 0 = none (the
    overlay is drawn only when nonzero, on the final frame).
    """
    board = state['board']
    n = board.shape[0]
    q = quality
    panel_w = 208  # keeps the default 8x64 board + panel at 720px, divisible by 16
    img = np.zeros((n * q, n * q + panel_w, 3), dtype=np.uint8)
    img[:] = _BG

    # grid lines (board region only)
    for i in range(n + 1):
        x = i * q
        img[:, max(0, x - 1):min(n * q, x + 1)] = _GRID
        img[max(0, x - 1):min(n * q, x + 1), :n * q] = _GRID

    mask, _, _ = get_valid_moves(board if state['white_to_move'] else -board)
    m = _circle_masks(q)
    for r in range(n):
        for c in range(n):
            v = board[r, c]
            patch = img[r * q:(r + 1) * q, c * q:(c + 1) * q]
            if v == 1:
                patch[m['rim']] = _WHITE_RIM
                patch[m['disc']] = _WHITE
            elif v == -1:
                patch[m['disc']] = _BLACK
            elif mask[r, c]:
                patch[m['hint']] = _HINT

    if state['flipped'] is not None:
        for r, c in state['flipped']:
            img[int(r) * q:(int(r) + 1) * q, int(c) * q:(int(c) + 1) * q][m['disc']] = _FLIP

    if state['last_move'] is not None:
        r, c = state['last_move']
        img[int(r) * q:(int(r) + 1) * q, int(c) * q:(int(c) + 1) * q][m['ring']] = _LAST

    if winner:
        img[:, :n * q] = (img[:, :n * q] * 0.45).astype(np.uint8)

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)

    if winner:
        text = 'White wins!' if winner == 1 else 'Black wins!' if winner == -1 else 'Draw'
        font = _font(int(q * 0.55))
        bb = draw.textbbox((0, 0), text, font=font)
        draw.text(((n * q - (bb[2] - bb[0])) / 2, (n * q - (bb[3] - bb[1])) / 2),
                  text, font=font, fill=(255, 255, 255))

    # score panel
    px = n * q
    draw.rectangle([px, 0, px + panel_w, n * q], fill=_PANEL)
    pad = 14
    small = _font(13)
    y = 12
    if title:
        for line in textwrap.wrap(title, width=24):
            draw.text((px + pad, y), line, font=small, fill=_TEXT)
            y += 17
        y += 8

    draw.text((px + pad, y), f"Ply {state['ply']}", font=_font(15), fill=_TEXT)
    if state['is_opening']:
        draw.text((px + pad, y + 20), "(random opening)", font=small, fill=_MUTED)
    y += 54

    total = n * n
    bar_x = px + pad + 78
    bar_max = panel_w - 2 * pad - 78
    for name, count, color in (
            ("White", state['white_count'], (245, 245, 245)),
            ("Black", state['black_count'], (95, 95, 95))):
        draw.text((px + pad, y), f"{name} {count}", font=_font(15), fill=_TEXT)
        draw.rectangle([bar_x, y + 4, bar_x + bar_max, y + 18], outline=_MUTED)
        draw.rectangle([bar_x, y + 4, bar_x + int(bar_max * count / total), y + 18], fill=color)
        y += 32

    y += 14
    draw.text((px + pad, y), "To move:", font=_font(14), fill=_MUTED)
    dot = (245, 245, 245) if state['white_to_move'] else (25, 25, 25)
    dot_x = px + pad + 72
    draw.ellipse([dot_x, y + 2, dot_x + 14, y + 16], fill=dot, outline=_MUTED)

    return np.asarray(pil)


def render_game(states, quality=64, title=None, winner=0):
    """Render every recorded state; the winner overlay is drawn on the final frame."""
    frames = []
    for i, state in enumerate(states):
        frames.append(render_frame(state, quality=quality, title=title,
                                   winner=winner if i == len(states) - 1 else 0))
    return frames
