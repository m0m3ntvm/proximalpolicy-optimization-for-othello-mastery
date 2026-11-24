"""Frame rendering for game videos (PIL + numpy).

Static frames (render_frame / render_game) show each ply's final position with a
classic green board, glossy shaded discs, legal-move hints, a red ring on the
last move, gold-highlighted flips, and a score panel.

Animated frames (render_animation) interpolate between plies so the video moves:
the placed disc pops in with a bounce, flipped discs rotate like coins sweeping
outward from the move, hints pulse, counts tick up, a title card fades in and
out, and the winner overlay fades in over a held final frame.
"""
import math
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from othello.game.rules import get_valid_moves

# palette
_CANVAS = (34, 37, 46)     # page background around the board
_FRAME = (26, 28, 34)      # board frame
_BG_TOP = (60, 165, 104)   # board gradient
_BG_BOT = (36, 126, 78)
_GRID = (24, 88, 56)
_PANEL = (42, 46, 58)      # score panel card
_TEXT = (238, 240, 244)
_MUTED = (160, 166, 178)
_WHITE = (245, 245, 245)
_BLACK = (22, 24, 28)
_GOLD = (240, 190, 60)
_LAST = (225, 60, 55)
_HINT = (255, 255, 255)

_fonts = {}
_shade_cache = {}
_bg_cache = {}


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


def _layout(n, q):
    """Board-frame width f, panel width, margin, and total canvas W x H."""
    f = 16 if n * q % 16 == 0 else int(round(q * 0.25))  # keep W/H divisible by 16
    panel_w = 220
    margin = 20
    H = n * q + 2 * f
    W = H + margin + panel_w
    return f, panel_w, margin, W, H


def _board_background(bw):
    if bw not in _bg_cache:
        t = np.linspace(0, 1, bw)[:, None, None]
        bg = (np.array(_BG_TOP)[None, None, :] * (1 - t)
              + np.array(_BG_BOT)[None, None, :] * t)
        _bg_cache[bw] = np.repeat(bg, bw, axis=1).astype(np.uint8)
    return _bg_cache[bw]


def _disc_shade(q):
    """(lighting multiplier, specular highlight) for a glossy disc."""
    if q not in _shade_cache:
        yy, xx = np.mgrid[0:q, 0:q]
        c = q / 2.0
        r = 0.38 * q
        rr = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
        shade = np.clip(1.0 - 0.55 * (rr / r) ** 1.5, 0.0, 1.0)
        sr = 0.16 * q
        spec = np.exp(-(((xx - (c - 0.28 * r)) ** 2 + (yy - (c - 0.28 * r)) ** 2)
                        / (2 * sr ** 2)))
        _shade_cache[q] = (shade, spec)
    return _shade_cache[q]


def _cell_mask(q, scale=1.0, squash=1.0, cx=None, cy=None):
    yy, xx = np.mgrid[0:q, 0:q]
    cx = q / 2.0 if cx is None else cx
    cy = q / 2.0 if cy is None else cy
    w = 0.38 * q * scale
    sq = max(squash, 0.06)
    return ((xx - cx) / sq) ** 2 + (yy - cy) ** 2 <= w ** 2


def _ring_mask(q):
    yy, xx = np.mgrid[0:q, 0:q]
    rr = np.sqrt((xx - q / 2.0) ** 2 + (yy - q / 2.0) ** 2)
    return (rr <= 0.47 * q) & (rr > 0.42 * q)


def _cell_slice(img, pr, pc, q):
    """q x q patch at pixel offset (pr, pc)."""
    return img[pr:pr + q, pc:pc + q]


def _draw_shadow(img, pr, pc, q, scale=1.0):
    """Soft drop shadow, offset toward the lower right."""
    m = _cell_mask(q, scale=scale * 1.1, squash=1.05,
                   cx=q / 2.0 + 0.07 * q, cy=q / 2.0 + 0.09 * q)
    patch = _cell_slice(img, pr, pc, q)
    patch[m] = (patch[m].astype(np.float64) * 0.65).astype(np.uint8)


def _draw_disc(img, pr, pc, q, color, scale=1.0, squash=1.0):
    """Glossy disc with radial shading and a specular highlight."""
    m = _cell_mask(q, scale=scale, squash=squash)
    shade, spec = _disc_shade(q)
    patch = _cell_slice(img, pr, pc, q)
    lit = np.array(color, dtype=np.float64)[None, None, :] * shade[..., None]
    lit = lit + spec[..., None] * 70.0
    patch[m] = np.clip(lit, 0, 255)[m].astype(np.uint8)


def _draw_hint(img, pr, pc, q, alpha):
    m = _cell_mask(q, scale=0.16)
    patch = _cell_slice(img, pr, pc, q)
    patch[m] = (patch[m].astype(np.float64) * (1 - alpha)
                + np.array(_HINT) * alpha).astype(np.uint8)


def _blend_ring(img, pr, pc, q, color, alpha):
    m = _ring_mask(q)
    patch = _cell_slice(img, pr, pc, q)
    patch[m] = (patch[m].astype(np.float64) * (1 - alpha)
                + np.array(color) * alpha).astype(np.uint8)


def _fade_alpha(i, n):
    a_in = min(1.0, i / max(1, n * 0.35))
    a_out = min(1.0, (n - i) / max(1, n * 0.25))
    return min(a_in, a_out)


def _title_card(W, H, title, alpha):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (30, 32, 40)
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    cy = H // 2 - 70
    for color, x0 in (((242, 242, 242), W // 2 - 130), ((28, 28, 32), W // 2 + 30)):
        d.ellipse([x0, cy - 50, x0 + 100, cy + 50], fill=color, outline=(150, 150, 150), width=2)
    d.text((W // 2 - 24, cy - 20), "vs", font=_font(38), fill=(240, 240, 240))
    lines = (title or "Othello").split("  vs  ")
    y = cy + 90
    for line in lines:
        fnt = _font(30)
        bb = d.textbbox((0, 0), line, font=fnt)
        d.text(((W - (bb[2] - bb[0])) / 2, y), line, font=fnt, fill=(235, 238, 244))
        y += 42
    img = np.asarray(pil)
    return (img.astype(np.float64) * alpha).astype(np.uint8)


def _draw_panel(img, panel_x, W, H, q, title, ply, total, wc, bc, white_to_move, is_opening, n):
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    d.rounded_rectangle([panel_x, 0, W, H], radius=14, fill=_PANEL)
    pad = 16
    y = 18
    small = _font(13)
    if title:
        for line in textwrap.wrap(title, width=26):
            d.text((panel_x + pad, y), line, font=small, fill=_TEXT)
            y += 17
        y += 8

    ply_text = f"Ply {ply}" + (f" / {total}" if total else "")
    d.text((panel_x + pad, y), ply_text, font=_font(21), fill=_TEXT)
    y += 30
    if total:
        track = [panel_x + pad, y, W - pad, y + 6]
        d.rounded_rectangle(track, radius=3, fill=(60, 64, 78))
        fill_w = int((W - 2 * pad) * (ply / max(1, total)))
        d.rounded_rectangle([panel_x + pad, y, panel_x + pad + fill_w, y + 6],
                            radius=3, fill=(90, 190, 130))
        y += 14
    if is_opening:
        d.text((panel_x + pad, y), "(random opening)", font=small, fill=_MUTED)
        y += 16
    y += 14

    for name, count, color, icon in (
            ("White", wc, (245, 245, 245), (242, 242, 242)),
            ("Black", bc, (95, 95, 95), (30, 30, 34))):
        d.ellipse([panel_x + pad, y + 1, panel_x + pad + 16, y + 17], fill=icon,
                  outline=_MUTED)
        d.text((panel_x + pad + 24, y), f"{name} {count}", font=_font(16), fill=_TEXT)
        bar_x = panel_x + pad + 96
        bar_max = W - 2 * pad - 96
        d.rounded_rectangle([bar_x, y + 5, bar_x + bar_max, y + 15], radius=5,
                            outline=_MUTED)
        d.rounded_rectangle([bar_x, y + 5, bar_x + int(bar_max * count / (n * n)), y + 15],
                            radius=5, fill=color)
        y += 32

    y += 16
    d.text((panel_x + pad, y), "To move", font=_font(14), fill=_MUTED)
    dot = (245, 245, 245) if white_to_move else (28, 28, 32)
    d.ellipse([panel_x + pad + 74, y + 2, panel_x + pad + 88, y + 16], fill=dot,
              outline=(170, 170, 170))
    img[:] = np.asarray(pil)


def _winner_overlay(frame, winner, alpha, n, q, f):
    img = frame.copy()
    bw = n * q
    if alpha > 0:
        dark = img.copy()
        dark[f:f + bw, f:f + bw] = (dark[f:f + bw, f:f + bw] * (1 - 0.55 * alpha)).astype(np.uint8)
        pil = Image.fromarray(dark)
        d = ImageDraw.Draw(pil)
        text = 'White wins!' if winner == 1 else 'Black wins!' if winner == -1 else 'Draw'
        fnt = _font(int(q * 0.55))
        bb = d.textbbox((0, 0), text, font=fnt)
        d.text(((bw + 2 * f - (bb[2] - bb[0])) / 2, (bw + 2 * f - (bb[3] - bb[1])) / 2),
               text, font=fnt, fill=(255, 255, 255))
        txt = np.asarray(pil)
        img = (img.astype(np.float64) * (1 - alpha)
               + txt.astype(np.float64) * alpha).astype(np.uint8)
    return img


def _frame_between(before, after, t, total, q, title):
    """One interpolated frame between two consecutive recorded plies (t in [0, 1])."""
    board = after['board']
    n = board.shape[0]
    f, panel_w, margin, W, H = _layout(n, q)
    bw = n * q

    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = _CANVAS
    img[0:H, 0:H] = _FRAME
    img[f:f + bw, f:f + bw] = _board_background(bw)

    for i in range(n + 1):  # grid lines
        x = f + i * q
        img[f:f + bw, max(0, x - 1):x + 1] = _GRID
        img[max(0, x - 1):x + 1, f:f + bw] = _GRID

    # side-to-move frame glow
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    mover = (235, 235, 235) if after['white_to_move'] else (96, 100, 112)
    d.rectangle([1, 1, H - 2, H - 2], outline=mover, width=4)
    img = np.array(pil)  # copy: PIL buffers are read-only

    mask, _, _ = get_valid_moves(before['board'] if before['white_to_move'] else -before['board'])
    placed = after['last_move']
    flipped_set = set(after['flipped'] or ())

    for r in range(n):
        for c in range(n):
            v = board[r, c]
            if v == 0:
                if mask[r, c]:
                    pulse = 0.45 + 0.25 * math.sin(2 * math.pi * (t + 0.13 * (r + c)))
                    _draw_hint(img, f + r * q, f + c * q, q, pulse)
                continue
            if (r, c) == placed or (r, c) in flipped_set:
                continue  # animated below
            _draw_shadow(img, f + r * q, f + c * q, q)
            _draw_disc(img, f + r * q, f + c * q, q, _WHITE if v == 1 else _BLACK)

    if placed is not None:
        u = min(1.0, max(0.0, (t - 0.30) / 0.35))  # pop-in with a bounce
        if u > 0:
            e = u * u * (3 - 2 * u)
            pop = 1 + 0.35 * (1 - e) * math.sin(math.pi * e)
            pr, pc = f + placed[0] * q, f + placed[1] * q
            _draw_shadow(img, pr, pc, q, scale=pop)
            _draw_disc(img, pr, pc, q, _WHITE if board[placed] == 1 else _BLACK, scale=pop)

    flips = list(after['flipped'] or ())
    if placed is not None:
        flips.sort(key=lambda rc: (rc[0] - placed[0]) ** 2 + (rc[1] - placed[1]) ** 2)
    for j, (r, c) in enumerate(flips):
        # sweep: first flip starts at t=0.45, last flip completes by t=0.93 so
        # even long flip lists finish before the ply settles (no visual snap)
        start = 0.45 + 0.30 * (j / max(1, len(flips)))
        p = min(1.0, max(0.0, (t - start) / 0.20))
        pr, pc = f + r * q, f + c * q
        old = _WHITE if before['board'][r, c] == 1 else _BLACK
        if p <= 0:
            _draw_shadow(img, pr, pc, q)
            _draw_disc(img, pr, pc, q, old)
        else:
            e = p * p * (3 - 2 * p)
            new = _WHITE if after['board'][r, c] == 1 else _BLACK
            gold = math.sin(math.pi * e)
            color = tuple(int((1 - gold) * (o + (nn - o) * e) + gold * g)
                          for o, nn, g in zip(old, new, _GOLD))
            _draw_shadow(img, pr, pc, q, scale=0.9)
            _draw_disc(img, pr, pc, q, color, squash=abs(math.cos(math.pi * e)))

    if placed is not None:
        a = min(1.0, max(0.0, (t - 0.85) / 0.15))  # last-move ring fades in
        if a > 0:
            _blend_ring(img, f + placed[0] * q, f + placed[1] * q, q, _LAST, a)

    wc = int(round(before['white_count'] + (after['white_count'] - before['white_count']) * t))
    bc = int(round(before['black_count'] + (after['black_count'] - before['black_count']) * t))
    _draw_panel(img, H + margin, W, H, q, title, after['ply'], total,
                wc, bc, after['white_to_move'], after['is_opening'], n)
    return img


def render_frame(state, quality=64, title=None, winner=0, total=None):
    """Render one recorded state as a uint8 RGB image (static, post-move look).

    state: dict from play_game. winner: 1 = white, -1 = black, 0 = none (the
    overlay is drawn only when nonzero, on the final frame).
    """
    img = _frame_between(state, state, 1.0, total, quality, title)
    if winner:
        n = state['board'].shape[0]
        f, _, _, _, _ = _layout(n, quality)
        img = _winner_overlay(img, winner, 1.0, n, quality, f)
    return img


def render_game(states, quality=64, title=None, winner=0):
    """Render every recorded state; the winner overlay is drawn on the final frame."""
    total = len(states) - 1
    frames = []
    for i, state in enumerate(states):
        frames.append(render_frame(state, quality=quality, title=title,
                                   winner=winner if i == len(states) - 1 else 0,
                                   total=total))
    return frames


def render_animation(states, fps=12, interp=6, quality=64, title=None, winner=0,
                     hold=2.0, card=1.5):
    """Animated video frames: title card, smooth per-ply moves, fading winner hold.

    frames = 1 title card + (len(states)-1) * interp move frames + hold * fps
    final frames (the winner overlay fades in over the first 0.4s of the hold).
    """
    n = states[0]['board'].shape[0]
    f, _, _, W, H = _layout(n, quality)
    total = len(states) - 1
    frames = []

    card_n = int(fps * card)
    for i in range(card_n):
        frames.append(_title_card(W, H, title, _fade_alpha(i, card_n)))

    for i in range(1, len(states)):
        before, after = states[i - 1], states[i]
        for k in range(interp):
            # (k+1)/interp so each ply ends on its fully settled state:
            # the last-move ring and the final flips must land (they need t >= 0.85)
            frames.append(_frame_between(before, after, (k + 1) / interp,
                                         total, quality, title))

    hold_n = int(fps * hold)
    last = frames[-1]
    for i in range(hold_n):
        a = min(1.0, i / max(1, fps * 0.4))
        frames.append(_winner_overlay(last, winner, a, n, quality, f))
    return frames
