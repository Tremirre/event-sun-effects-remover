import logging

import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FFTConvCell(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        half_channels = out_channels // 2
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.half_conv = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=half_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.fftc = FourierConvolution(half_channels, half_channels, kernel_size)
        self.agg_conv = torch.nn.Conv2d(
            in_channels=half_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.relu(x)
        x = self.half_conv(x)
        x = torch.relu(x)
        x2 = x.detach().clone()
        x = self.fftc(x)
        x = torch.hstack((x, x2))
        x = self.agg_conv(x)
        x = torch.relu(x)
        return x


class FourierConvolution(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        # bn_layer not used
        self.conv_layer = torch.nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, c, h, w = x.size()
        r_size = x.size()

        # (batch, c, h, w/2+1)
        ffted = torch.fft.rfft2(x, norm="forward")
        ffted = torch.hstack((ffted.real, ffted.imag))
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))
        ffted_real = ffted[:, :c, :, :]
        ffted_imag = ffted[:, c:, :, :]
        ffted = torch.complex(ffted_real, ffted_imag)
        output = torch.fft.irfft2(ffted, s=r_size[2:], norm="forward")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return output
