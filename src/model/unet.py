import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import const
from .ffconv import FFTConvCell

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        kernel_size: int = 3,
        batch_norm: bool = True,
        with_fft: bool = False,
    ) -> None:
        padding = kernel_size // 2
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)]
        )
        if with_fft:
            self.convs.append(FFTConvCell(out_channels, out_channels, kernel_size))
        self.convs.extend(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
                for _ in range(depth)
            ]
        )
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x_orig = self.shortcut(x)
        for conv in self.convs:
            x = F.relu(conv(x))
        if self.batch_norm:
            x = self.bn(x)
        return F.relu(x + x_orig)


class UpConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        kernel_size: int = 3,
        with_fft: bool = False,
    ) -> None:
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            output_padding=0,
        )
        self.conv_block = ConvBlock(
            out_channels * 2, out_channels, depth, kernel_size, with_fft=with_fft
        )

    def forward(self, x, x_skip):
        x = F.relu(self.upconv(x))
        if x.size()[2:] != x_skip.size()[2:]:
            x = F.interpolate(
                x, size=x_skip.size()[2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, x_skip], dim=1)
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        block_depth: int,
        in_channels: int,
        kernel_size: int = 3,
        with_fft: bool = False,
    ) -> None:
        super().__init__()
        features = [in_channels]
        for i in range(n_blocks):
            features.append(2 ** (i + 6))
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(
                    features[i],
                    features[i + 1],
                    block_depth,
                    kernel_size,
                    with_fft=with_fft if i > 0 else False,
                )
                for i in range(n_blocks)
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                UpConvBlock(
                    features[i + 1],
                    features[i],
                    block_depth,
                    kernel_size,
                    with_fft=with_fft if i > 0 else False,
                )
                for i in range(n_blocks - 1, 0, -1)
            ]
        )
        self.final = nn.Conv2d(features[1], const.CHANNELS_OUT, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skips = []
        for block in self.down_blocks:
            x = block(x)
            x_skips.append(x)
            x = F.max_pool2d(x, 2)

        x_skips = x_skips[:-1]
        for block, x_skip in zip(self.up_blocks, x_skips[::-1]):
            x = block(x, x_skip)

        x = self.final(x)
        return x


class HalfUNetDiscriminator(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        block_depth: int,
        in_channels: int,
        kernel_size: int = 3,
        with_fft: bool = False,
    ) -> None:
        super().__init__()
        features = [in_channels]
        for i in range(n_blocks):
            features.append(2 ** (i + 6))
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(
                    features[i],
                    features[i + 1],
                    block_depth,
                    kernel_size,
                    with_fft=with_fft if i > 0 else False,
                )
                for i in range(n_blocks)
            ]
        )
        div = 2 ** (n_blocks)
        linear_params = 4 * const.IMG_WIDTH * const.IMG_HEIGHT // (div**2)
        self.flat_section = nn.Sequential(
            nn.Conv2d(features[-1], 4, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(linear_params, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.down_blocks:
            x = block(x)
            x = F.max_pool2d(x, 2)
        x = self.flat_section(x)
        return x
