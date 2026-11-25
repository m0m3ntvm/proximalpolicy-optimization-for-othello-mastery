"""Neural network architecture: attention modules, residual blocks, agent,
and the alternative encoders (Vision Mamba / Vision-LSTM) with the factory."""
from othello.models.agent import OthelloRL, value_Net
from othello.models.attention import CBAM, ChannelAttention, SpatialAttention
from othello.models.blocks import ResidualBlock
from othello.models.factory import ARCHS, make_agent, make_from_spec
from othello.models.vim import VimEncoder
from othello.models.xlstm import XLSTMEncoder

__all__ = ["OthelloRL", "value_Net", "CBAM", "ChannelAttention", "SpatialAttention",
           "ResidualBlock", "VimEncoder", "XLSTMEncoder", "make_agent",
           "make_from_spec", "ARCHS"]
