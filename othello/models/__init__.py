"""Neural network architecture: attention modules, residual blocks, agent,
and the alternative encoders (Transformer / LSTM) with the factory."""
from othello.models.agent import OthelloRL, value_Net
from othello.models.attention import CBAM, ChannelAttention, SpatialAttention
from othello.models.blocks import ResidualBlock
from othello.models.factory import (ARCHS, AgentParallel, make_agent,
                                     make_from_spec, parallelize, unwrap)
from othello.models.lstm import LSTMEncoder
from othello.models.transformer import TransformerEncoder

__all__ = ["OthelloRL", "value_Net", "CBAM", "ChannelAttention", "SpatialAttention",
           "ResidualBlock", "TransformerEncoder", "LSTMEncoder",
           "make_agent", "make_from_spec", "AgentParallel", "parallelize",
           "unwrap", "ARCHS"]
