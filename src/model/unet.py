import pytorch_lightning as pl
import pytorch_msssim as msssim
import torch
import torch.nn as nn
import torch.nn.functional as F

CHANNELS_IN = 5  # RGB + Event + Mask
CHANNELS_OUT = 3  # RGB

IMG_HEIGHT = 480
IMG_WIDTH = 640


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, batch_norm: bool = True
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.batch_norm:
            x = self.bn(x)
        return x


class UpConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x, x_skip):
        x = F.relu(self.upconv(x))
        x = torch.cat([x, x_skip], dim=1)
        x = self.conv_block(x)
        return x


class UNet(pl.LightningModule):
    def __init__(self, n_blocks: int) -> None:
        super().__init__()
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(CHANNELS_IN if i == 0 else 2 ** (i - 1), 2**i)
                for i in range(n_blocks)
            ]
        )
        self.up_blocks = nn.ModuleList(
            [UpConvBlock(2**i, 2 ** (i - 1)) for i in range(n_blocks, 1, -1)]
        )
        self.final = nn.Conv2d(8, CHANNELS_OUT, 1)

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # mae + ssim
        mae = F.l1_loss(y_hat, y)
        ssim = 1 - msssim.ms_ssim(y_hat, y)
        self.log("mae", mae)
        self.log("ssim", ssim)
        return mae + ssim

    def forward(self, x):
        x_skips = []
        for block in self.down_blocks:
            x = block(x)
            x_skips.append(x)
            x = F.max_pool2d(x, 2)
        for block, x_skip in zip(self.up_blocks, x_skips[::-1]):
            x = block(x, x_skip)
        x = self.final(x)
        return x

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)  # type: ignore
