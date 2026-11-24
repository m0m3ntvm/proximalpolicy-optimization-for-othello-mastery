"""DEPRECATED compatibility shim for the original single-file module.

The code formerly in this file now lives in the `othello` package
(game/, models/, search/, rl/). This shim re-exports the public API so that:

  1. old notebooks using `from _othello import ...` keep working, and
  2. the published checkpoints — full-object pickles whose classes resolve
     via `sys.modules['workspace']` — can be loaded by aliasing 'workspace'
     to this module (or to `othello` directly; see othello/checkpoint.py).

New code should import from `othello` instead.
"""
from othello import (  # noqa: F401  (re-exported for backwards compatibility)
    CBAM,
    ChannelAttention,
    OthelloBoard,
    OthelloRL,
    PPO_Loss,
    ResidualBlock,
    SpatialAttention,
    asyncSelfPlay,
    board_hash,
    decide,
    evaluate,
    get_valid_moves,
    miniMax,
    parallelSelfPlay,
    value_Net,
)
