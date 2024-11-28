import logging

import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FourierConvolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
        # bn_layer not used
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.groups,
            bias=False,
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
