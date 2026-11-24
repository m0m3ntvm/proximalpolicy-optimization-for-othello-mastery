"""Vectorized self-play: kernel equivalence, per-game RNG independence
(batch-size invariance, determinism), move legality, and return semantics.

Lockstep batching cannot replay the sequential selfPlay games exactly — with a
shared global RNG stream, interleaving draws between games changes the bits
each game sees. Instead each game gets its own pair of streams, which is the
stronger property for training: games are independent of each other and of the
batch size.
"""
import numpy as np
import torch

from othello.game.board import OthelloBoard
from othello.game.rules import get_valid_moves
from othello.models.agent import OthelloRL
from othello.rl.vecenv import VectorSelfPlay, _vec_movegen, batchedSelfPlay


class DetPolicy:
    """Per-sample pure function of its input: uniform over legal moves."""
    name = 'det'

    def __call__(self, x, mask):
        return mask.to(torch.float32) * 100.0


class ExplodingModel:
    """Must never be consulted (epsilon=1.0 makes every move random)."""
    name = 'exploding'

    def __call__(self, x, mask):
        raise RuntimeError('model consulted despite epsilon=1.0')


def test_vec_movegen_matches_get_valid_moves():
    """The vectorized kernel must agree exactly with the scalar rules kernel."""
    rng = np.random.default_rng(0)
    for _ in range(30):
        b = rng.integers(-1, 2, size=(8, 8)).astype(np.short)
        mask_ref, under_ref, moves = get_valid_moves(b)
        mask, under = _vec_movegen(b[None], np.ones(1, dtype=bool))
        assert np.array_equal(mask[0], mask_ref)
        assert np.array_equal(under[0], under_ref)


def test_vec_batch_size_invariance():
    """The same seed must produce the same games regardless of the batch size:
    game g's streams are derived from the seed, not from the lockstep order.
    (The concatenated list follows game-finish order, which differs across
    batch compositions — so compare as multisets, as the shuffled DataLoader
    effectively sees them.)"""
    def key(entry):
        (s, m), a, r = entry
        return (s.tobytes(), m.tobytes(), a.tobytes(), float(r))

    data_small = sorted(batchedSelfPlay(DetPolicy(), None, games=8, Epsilon=0.1,
                                        boltzmann=True, batch=2,
                                        start_explore=(2, 10), seed=123), key=key)
    data_big = sorted(batchedSelfPlay(DetPolicy(), None, games=8, Epsilon=0.1,
                                      boltzmann=True, batch=8,
                                      start_explore=(2, 10), seed=123), key=key)
    assert len(data_small) == len(data_big)
    for ((s1, m1), a1, r1), ((s2, m2), a2, r2) in zip(data_small, data_big):
        assert np.array_equal(s1, s2)
        assert np.array_equal(m1, m2)
        assert np.array_equal(a1, a2)
        assert float(r1) == float(r2)


def test_vec_deterministic_same_seed():
    """Same seed, same games — the per-game streams make the whole run reproducible."""
    run = lambda: batchedSelfPlay(DetPolicy(), None, games=6, Epsilon=0.1,
                                  boltzmann=True, batch=3, start_explore=(0, 16),
                                  seed=7)
    d1, d2 = run(), run()
    assert len(d1) == len(d2)
    for ((s1, m1), a1, r1), ((s2, m2), a2, r2) in zip(d1, d2):
        assert np.array_equal(s1, s2)
        assert np.array_equal(m1, m2)
        assert np.array_equal(a1, a2)
        assert float(r1) == float(r2)


def test_model_batched_output_matches_single():
    """The real model's batched output must match per-sample single forwards
    (GroupNorm/softmax/attention all act per-sample)."""
    torch.manual_seed(1)
    model = OthelloRL(internal_channels=8, res_depth=2, name='tiny',
                      return_val=False).eval()
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 2, 8, 8)).astype(np.float32)
    mk = rng.integers(0, 2, (4, 1, 8, 8)).astype(np.float32)
    with torch.no_grad():
        batch = model(torch.from_numpy(x), torch.from_numpy(mk)).numpy()
        singles = np.concatenate([
            model(torch.from_numpy(x[i:i + 1]), torch.from_numpy(mk[i:i + 1])).numpy()
            for i in range(4)])
    assert np.allclose(batch, singles, atol=1e-6)


def test_vec_actions_are_legal_moves():
    """Every recorded action must be a legal move from its recorded state."""
    data = batchedSelfPlay(ExplodingModel(), None, games=8, Epsilon=1.0,
                           boltzmann=False, batch=3, start_explore=(1, 20), seed=11)
    assert len(data) > 0
    for (state, _mask), action, _ret in data:
        r, c = np.unravel_index(int(np.argmax(action)), action.shape[1:])
        legal, _, _ = get_valid_moves(state[0].astype(np.short))
        assert legal[r, c], f'illegal action {(r, c)} for state\n{state[0]}'


def test_vec_returns_are_gamma_scaled_and_signed():
    """Targets: 0 for draws, else +-gamma**empties — exactly the sequential
    selfPlay formula (scale is computed from the pre-move state)."""
    data = batchedSelfPlay(ExplodingModel(), None, games=12, Epsilon=1.0,
                           boltzmann=False, batch=5, seed=7)
    gamma = OthelloBoard().gamma
    assert len(data) > 0
    for (state, _mask), _action, ret in data:
        empties = np.count_nonzero(state[0] == 0)
        scale = float(gamma ** empties)
        r = float(ret)
        assert r == 0.0 or abs(abs(r) - scale) < 1e-6, f'ret={r} scale={scale}'


def test_vec_selfplay_terminates_and_counts_pieces():
    """Every game ends legally: recorded states add exactly one piece per move."""
    env = VectorSelfPlay(6, ExplodingModel(), None, size=8, epsilon=1.0,
                         boltzmann=False, start_explore=(0, 8),
                         game_seeds=np.random.SeedSequence(3).spawn(6))
    env.run()
    assert not env.active.any()
    for g in range(6):
        counts = [int(np.count_nonzero(s[0])) for s in env.states[g]]
        # passes are not recorded, so each recorded state adds exactly one piece
        assert all(c1 - c0 == 1 for c0, c1 in zip(counts, counts[1:])), \
            f'game {g}: illegal count sequence {counts}'
        assert len(env.data_by_game[g]) == len(env.actions[g])
