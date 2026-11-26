"""SHA-ZA: PPO self-play Othello agent.

Public API mirrors the original `_othello.py` module so that old notebooks and
the published checkpoints (pickled under the module name 'workspace') keep
working — every name below must stay exportable, because the checkpoints
reference the classes by attribute lookup on the aliased module.
"""
from othello.game.board import OthelloBoard
from othello.game.rules import board_hash, decide, evaluate, get_valid_moves
from othello.models.agent import OthelloRL, value_Net
from othello.models.attention import CBAM, ChannelAttention, SpatialAttention
from othello.models.blocks import ResidualBlock
from othello.models.factory import (ARCHS, AgentParallel, make_agent,
                                     make_from_spec, parallelize, unwrap)
from othello.models.lstm import LSTMEncoder
from othello.models.transformer import TransformerEncoder
from othello.rl.ppo import PPO_Loss
from othello.rl.selfplay import asyncSelfPlay, parallelSelfPlay
from othello.search.minimax import miniMax

__all__ = [
    "OthelloRL",
    "value_Net",
    "ResidualBlock",
    "CBAM",
    "ChannelAttention",
    "SpatialAttention",
    "miniMax",
    "OthelloBoard",
    "PPO_Loss",
    "get_valid_moves",
    "board_hash",
    "evaluate",
    "decide",
    "asyncSelfPlay",
    "parallelSelfPlay",
    "TransformerEncoder",
    "LSTMEncoder",
    "make_agent",
    "make_from_spec",
    "AgentParallel",
    "parallelize",
    "unwrap",
    "ARCHS",
]
