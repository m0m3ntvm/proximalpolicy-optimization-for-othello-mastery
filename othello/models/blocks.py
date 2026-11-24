"""ConvNeXt-inspired residual block with CBAM attention."""
import torch
from torch import nn

from othello.models.attention import CBAM


class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 64, groups: int = 4, stride: int = 1, downsample=None) -> None:
        super(ResidualBlock, self).__init__()
        group_width = channels * groups
        self.conv = nn.Sequential(
            #########################
            nn.Conv2d(channels, group_width, kernel_size=3, stride=1, padding=1, bias=False),  # large kernel inspired by ConvNeXt
            nn.GroupNorm(num_groups=groups, num_channels=group_width, affine=True),
            ##########################
            CBAM(channels=group_width),  # apply attention
            ##########################
            nn.Conv2d(group_width, group_width, kernel_size=1, stride=stride, padding=0, groups=groups),
            nn.LeakyReLU(inplace=True),
            #########################
            nn.Conv2d(group_width, channels, kernel_size=1, stride=1, padding=0),
        )

        self.shortcut = downsample if downsample else nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv(x)
        x += shortcut
        return x
