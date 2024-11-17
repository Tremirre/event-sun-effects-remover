import pytorch_lightning as pl
import pytorch_msssim as msssim
import torch
import torch.nn as nn
import torch.nn.functional as F

CHANNELS_IN = 5  # RGB + Event + Mask
CHANNELS_OUT = 3  # RGB


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
        # Added output_padding=1 to ensure output size matches skip connection
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, output_padding=0
        )
        self.conv_block = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x, x_skip):
        x = F.relu(self.upconv(x))
        if x.size()[2:] != x_skip.size()[2:]:
            x = F.interpolate(
                x, size=x_skip.size()[2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, x_skip], dim=1)
        x = self.conv_block(x)
        return x


class UNet(pl.LightningModule):
    def __init__(self, n_blocks: int) -> None:
        super().__init__()
        features = [CHANNELS_IN]
        for i in range(n_blocks):
            features.append(2 ** (i + 6))
        self.down_blocks = nn.ModuleList(
            [ConvBlock(features[i], features[i + 1]) for i in range(n_blocks)]
        )
        self.up_blocks = nn.ModuleList(
            [
                UpConvBlock(features[i + 1], features[i])
                for i in range(n_blocks - 1, 0, -1)
            ]
        )
        self.final = nn.Conv2d(features[1], CHANNELS_OUT, 1)

    def loss(
        self, y_hat: torch.Tensor, y: torch.Tensor, stage: str = ""
    ) -> torch.Tensor:
        # mae + ssim
        mae = F.l1_loss(y_hat, y)
        ssim = 1 - msssim.ms_ssim(y_hat, y)
        if stage:
            self.log(f"{stage}_mae", mae)
            self.log(f"{stage}_ssim", ssim)
        return mae + ssim

    def forward(self, x):
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

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y, stage="train")
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y, stage="val")
        self.log("val_loss", loss)
        if batch_idx == 0:
            x_rgb = x[:, :3]
            x_event = x[:, 3:4]
            x_event = torch.cat([x_event, x_event, x_event], dim=1)
            x_mask = x[:, 4:5]
            x_mask = torch.cat([x_mask, x_mask, x_mask], dim=1)
            x_rgb = torch.where(x_mask > 0, x_event, x_rgb)
            self.logger.experiment.add_images(
                "input", x_rgb, global_step=self.global_step
            )
            self.logger.experiment.add_images(
                "output", y_hat, global_step=self.global_step
            )
            self.logger.experiment.add_images("target", y, global_step=self.global_step)
        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y, stage="test")
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)  # type: ignore
