"""Visualization engine tests: recording, frame rendering, video writing."""
import os
import random

import numpy as np
import pytest
from PIL import Image

from othello.game.board import OthelloBoard
from othello.search.minimax import miniMax
from othello.viz import (display_board, play_game, render_animation, render_game,
                         visualize_game, write_gif, write_video)


@pytest.fixture
def agents():
    return miniMax(depth=1, workers=1), miniMax(depth=2, workers=1)


def test_play_game_states_fixed_orientation(agents):
    a, b = agents
    result, states = play_game(a, b, board=OthelloBoard(size=4))
    assert result.shape == (2,)
    assert len(states) >= 5
    first = states[0]
    assert first['board'][1, 1] == 1.0  # white pieces are +1 in every snapshot
    assert first['board'][1, 2] == -1.0
    assert first['ply'] == 0 and first['last_move'] is None
    for s in states:
        assert np.all(np.abs(s['board']) <= 1)
        assert s['white_count'] + s['black_count'] == int(np.count_nonzero(s['board']))
        assert s['white_count'] == int(np.count_nonzero(s['board'] == 1))
        if s['last_move'] is not None:
            r, c = s['last_move']
            assert s['board'][r, c] != 0


def test_play_game_matches_model_vs_model():
    """The recording loop must replay modelVsModel exactly: same result and same
    final board under the same random seed."""
    a, b = miniMax(depth=1, workers=1), miniMax(depth=2, workers=1)
    random.seed(7)
    result_viz, states = play_game(a, b, board=OthelloBoard(size=4))
    random.seed(7)
    board_ref = OthelloBoard(size=4)
    result_ref = board_ref.modelVsModel(a, b)
    assert np.array_equal(result_viz, result_ref)
    assert np.array_equal(states[-1]['board'], display_board(board_ref))


def test_move_delta_detection(agents):
    """Each recorded ply must detect the placed square (empty before, filled after)
    and the flipped squares (sign change), or a pass (no change)."""
    a, b = agents
    result, states = play_game(a, b, board=OthelloBoard(size=4))
    assert any(s['last_move'] is not None for s in states[1:])
    for i in range(1, len(states)):
        s = states[i]
        before = states[i - 1]['board']
        if s['last_move'] is None:
            continue
        r, c = s['last_move']
        assert before[r, c] == 0
        assert s['board'][r, c] != 0
        for fr, fc in s['flipped']:
            assert before[fr, fc] == -s['board'][fr, fc]
            assert before[fr, fc] != 0


def test_render_game_frames(agents):
    a, b = agents
    result, states = play_game(a, b, board=OthelloBoard(size=4))
    frames = render_game(states, quality=16)
    assert len(frames) == len(states)
    f0 = frames[0]
    assert f0.dtype == np.uint8 and f0.ndim == 3 and f0.shape[2] == 3
    assert f0.shape[0] >= 4 * 16  # board plus frame borders
    assert not np.array_equal(frames[0], frames[-1])  # the game progressed


def test_render_animation_frames(agents):
    a, b = agents
    result, states = play_game(a, b, board=OthelloBoard(size=4))
    fps, interp, hold, card = 10, 4, 2.0, 1.5
    frames = render_animation(states, fps=fps, interp=interp, quality=16,
                              title="a vs b", winner=1)
    expected = int(fps * card) + (len(states) - 1) * interp + int(fps * hold)
    assert len(frames) == expected
    f0 = frames[0]
    assert f0.dtype == np.uint8 and f0.ndim == 3 and f0.shape[2] == 3
    assert all(f.shape == f0.shape for f in frames)
    first_move = int(fps * card)  # skip the title card
    assert not np.array_equal(frames[first_move], frames[first_move + 1])  # motion
    hold_n = int(fps * hold)
    assert not np.array_equal(frames[-hold_n - 1], frames[-1])  # winner overlay appears


def test_write_gif_roundtrip(agents, tmp_path):
    a, b = agents
    result, states = play_game(a, b, board=OthelloBoard(size=4))
    path = str(tmp_path / "game.gif")
    write_gif(render_game(states, quality=16), path, fps=2)
    with Image.open(path) as im:
        assert im.n_frames == len(states)
    assert os.path.getsize(path) > 500


def test_write_mp4_roundtrip(agents, tmp_path):
    imageio = pytest.importorskip('imageio.v2')
    a, b = agents
    result, states = play_game(a, b, board=OthelloBoard(size=4))
    path = str(tmp_path / "game.mp4")
    write_video(render_game(states, quality=16), path, fps=2)
    with imageio.get_reader(path) as reader:
        assert reader.count_frames() == len(states)
    assert os.path.getsize(path) > 500


def test_visualize_game_pipeline(tmp_path):
    a = miniMax(depth=1, workers=1)
    result, path = visualize_game(a, a, out=str(tmp_path / "match.gif"), seed=3,
                                  board=OthelloBoard(size=4), quality=16,
                                  fps=4, interp=2)
    assert len(result) == 2
    assert os.path.exists(path)
    with Image.open(path) as im:
        assert im.n_frames >= 4  # title card + interpolated plies + held final frame
