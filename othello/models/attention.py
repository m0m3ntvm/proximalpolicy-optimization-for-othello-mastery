"""Convolutional Block Attention Module (CBAM) building blocks."""
import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int = 64, scale_factor: int = 8) -> None:
        super(ChannelAttention, self).__init__()
        self.scale_factor = scale_factor

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // self.scale_factor, kernel_size=1, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels // self.scale_factor, channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 3) -> None:
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel attention then spatial attention."""

    def __init__(self, channels: int = 64) -> None:
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels=channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
