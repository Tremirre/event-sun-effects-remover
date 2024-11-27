import torch


class FastFourierConvolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        """
        Fast Fourier Convolution module.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (tuple or int): Size of the convolution kernel.
        """
        super(FastFourierConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )

        # Initialize the convolution kernel in the spatial domain
        self.weight = torch.nn.Parameter(
            torch.randn(
                out_channels, in_channels, *self.kernel_size, dtype=torch.float32
            )
            * 0.01
        )

    def forward(self, x):
        """
        Forward pass for Fast Fourier Convolution.
        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch, out_channels, height, width).
        """
        batch_size, _, height, width = x.shape
        padded_height = height + self.kernel_size[0] - 1
        padded_width = width + self.kernel_size[1] - 1

        # Perform FFT on the input tensor
        x_fft = torch.fft.rfft2(x, s=(padded_height, padded_width))

        # Pad and perform torch.fft on the kernel
        kernel_padded = torch.zeros(
            self.out_channels,
            self.in_channels,
            padded_height,
            padded_width,
            device=x.device,
            dtype=x.dtype,
        )
        kernel_padded[:, :, : self.kernel_size[0], : self.kernel_size[1]] = self.weight
        kernel_fft = torch.fft.rfft2(kernel_padded)

        # Multiply in the frequency domain (convolution theorem)
        output_fft = torch.einsum("bihw,oihw->bohw", x_fft, kernel_fft)

        # Perform the inverse torch.fft to get back to the spatial domain
        output = torch.fft.irfft2(output_fft, s=(padded_height, padded_width))

        # Crop to the original spatial size
        output = output[..., :height, :width]
        return output


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
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view(
            (
                batch,
                -1,
            )
            + ffted.size()[3:]
        )

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = (
            ffted.view(
                (
                    batch,
                    -1,
                    2,
                )
                + ffted.size()[2:]
            )
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )  # (batch,c, t, h, w/2+1, 2)

        output = torch.fft.irfft(
            ffted, signal_ndim=2, signal_sizes=r_size[2:], normalized=True
        )

        return output
