import pytorch_lightning as pl
import pytorch_msssim as msssim
import torch
import torch.nn.functional as F

from src import const

from ..loss import TVLoss, VGGLoss
from .unet import UNet


class BaseInpaintingModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.tv_loss = TVLoss(2)
        self.vgg_loss = VGGLoss()

    def loss(
        self, y_hat: torch.Tensor, y: torch.Tensor, stage: str = ""
    ) -> torch.Tensor:
        # mae + ssim
        mae = F.l1_loss(y_hat, y)
        ssim = 1 - msssim.ms_ssim(y_hat, y)
        vgg_loss = self.vgg_loss(y_hat, y)
        tv_loss = self.tv_loss(y_hat)
        if stage:
            self.log(f"{stage}_mae", mae)
            self.log(f"{stage}_ssim", ssim)
            self.log(f"{stage}_vgg", vgg_loss)
            self.log(f"{stage}_tv", tv_loss)
        return 1.0 * mae + 0.5 * ssim + 0.1 * vgg_loss + 0.05 * tv_loss

    def forward(self, x):
        return x

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y, stage=stage)
        self.log(f"{stage}_loss", loss)
        return {
            "loss": loss,
            "pred": y_hat,
        }

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self._shared_step(batch, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)  # type: ignore


class NoOp(BaseInpaintingModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :3]

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str):
        x, y = batch
        return {
            "loss": torch.tensor(0.0),
            "pred": self(x),
        }

    def configure_optimizers(self):
        return None

    def backward(self, loss: torch.Tensor, *args, **kwargs) -> None:
        pass


class UNetModule(BaseInpaintingModule):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__()
        self.unet = UNet(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unet(x)
        x = torch.sigmoid(x)
        return x


class UNetDualWithFFT(BaseInpaintingModule):
    def __init__(
        self,
        n_blocks: int,
        block_depth: int,
        in_channels: int,
        kernel_size: int = 3,
        **kwargs,
    ) -> None:
        super().__init__()
        self.std_unet = UNet(n_blocks, block_depth, in_channels, kernel_size)
        self.fft_unet = UNet(
            n_blocks, block_depth, in_channels, kernel_size, with_fft=True
        )
        self.combine_conv = torch.nn.Conv2d(
            const.CHANNELS_OUT * 2, const.CHANNELS_OUT, 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_copy = x.clone()
        x_std = self.std_unet(x)
        x_fft = self.fft_unet(x_copy)
        x = torch.cat([x_std, x_fft], dim=1)
        x = self.combine_conv(x)
        x = torch.sigmoid(x)
        return x


NAMES = {
    "unet": UNetModule,
    "unet_dual": UNetDualWithFFT,
    "noop": NoOp,
}
