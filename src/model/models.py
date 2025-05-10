import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import const
from .ffconv import FFTConvCell

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AttentionGate(nn.Module):
    def __init__(
        self, in_channels: int, gating_channels: int, inter_channels: int | None = None
    ):
        super(AttentionGate, self).__init__()
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels or in_channels // 2

        self.W_x = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.W_g = nn.Conv2d(
            in_channels=self.gating_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.psi = nn.Conv2d(
            in_channels=self.inter_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        nn.init.xavier_normal_(self.W_x.weight)
        nn.init.xavier_normal_(self.W_g.weight)
        nn.init.xavier_normal_(self.psi.weight)
        nn.init.constant_(self.W_x.bias, 0)
        nn.init.constant_(self.W_g.bias, 0)
        nn.init.constant_(self.psi.bias, 0)
        self.bn = nn.BatchNorm2d(self.inter_channels)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # Project input feature map and gating signal
        x1 = self.W_x(x)
        g1 = self.W_g(g)

        # If dimensions don't match, upsample g1
        if x1.size(2) != g1.size(2) or x1.size(3) != g1.size(3):
            g1 = F.interpolate(
                g1, size=(x1.size(2), x1.size(3)), mode="bilinear", align_corners=False
            )

        # Element-wise addition followed by non-linearity
        combined = self.bn(x1 + g1)
        combined = F.relu(combined)

        # Compute attention coefficients
        attention_map = self.psi(combined)
        attention_map = torch.sigmoid(attention_map)

        # Apply attention coefficients to input feature map
        return x * attention_map


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        kernel_size: int = 3,
        activation_func: str = "relu",
        batch_norm: bool = True,
        with_fft: bool = False,
    ) -> None:
        assert activation_func in ACTIVATIONS
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        padding = kernel_size // 2
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
        self.activation = ACTIVATIONS[activation_func]

    def forward(self, x):
        x_orig = self.shortcut(x)
        for conv in self.convs:
            x = self.activation(conv(x))
        if self.batch_norm:
            x = self.bn(x)
        return self.activation(x + x_orig)


class UpConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        kernel_size: int = 3,
        activation_func: str = "relu",
        batch_norm: bool = True,
        with_fft: bool = False,
        with_ag: bool = False,
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
            out_channels * 2,
            out_channels,
            depth,
            kernel_size,
            with_fft=with_fft,
            activation_func=activation_func,
            batch_norm=batch_norm,
        )
        self.attention_gate = None  # TODO: implement attention gate
        if with_ag:
            self.attention_gate = AttentionGate(
                in_channels=out_channels,
                gating_channels=in_channels,
            )

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x_up = self.conv_block.activation(self.upconv(x))

        if x_up.size()[2:] != x_skip.size()[2:]:
            x_up = F.interpolate(
                x_up, size=x_skip.size()[2:], mode="bilinear", align_corners=False
            )
        if self.attention_gate is not None:
            x_skip = self.attention_gate(x_skip, x)

        x = torch.cat([x_up, x_skip], dim=1)
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        block_depth: int,
        in_channels: int,
        kernel_size: int = 3,
        activation_func: str = "relu",
        batch_norm: bool = True,
        with_fft: bool = False,
        with_ag: bool = False,
        out_channels: int = const.CHANNELS_OUT,
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
                    activation_func=activation_func,
                    batch_norm=batch_norm,
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
                    activation_func=activation_func,
                    batch_norm=batch_norm,
                    with_fft=with_fft if i > 0 else False,
                    with_ag=with_ag,
                )
                for i in range(n_blocks - 1, 0, -1)
            ]
        )
        self.final = nn.Conv2d(features[1], out_channels, 1)

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


class NoOp(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x):
        return x


MODELS = {
    "unet": UNet,
    "noop": NoOp,
}


ACTIVATIONS = {
    "relu": F.relu,
    "gelu": F.gelu,
    "elu": F.elu,
    "leakyrelu": F.leaky_relu,
    "mish": F.mish,
}


def get_model(model_name: str, **kwargs) -> nn.Module:
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found")
    return MODELS[model_name](**kwargs)
