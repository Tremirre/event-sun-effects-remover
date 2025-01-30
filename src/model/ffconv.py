import logging

import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ChannelFusionModule(torch.nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels * 2, in_channels // 4, 1, bias=False
            ),  # Reduce channels
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                in_channels // 4, in_channels * 2, 1, bias=False
            ),  # Restore channels
            torch.nn.Sigmoid(),
        )

    def forward(
        self, fft_features: torch.Tensor, multi_features: torch.Tensor
    ) -> torch.Tensor:
        # Concatenate features along the channel dimension
        combined_features = torch.cat([fft_features, multi_features], dim=1)

        # Generate attention weights
        attention_weights = self.fc(self.global_avg_pool(combined_features))

        # Split attention weights for fft_features and multi_features
        fft_weight, multi_weight = torch.split(
            attention_weights, fft_features.size(1), dim=1
        )

        # Apply attention weights
        fused_features = fft_weight * fft_features + multi_weight * multi_features
        return fused_features


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


class FFTConvCell(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.down_conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding
        )
        self.fftc = FourierConvolution(in_channels, out_channels, kernel_size)
        self.aggmod = ChannelFusionModule(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_intensity = x.detach().clone()
        x_intensity = self.down_conv(x_intensity)
        x_fft = self.fftc(x)
        x = self.aggmod(x_fft, x_intensity)
        return x
