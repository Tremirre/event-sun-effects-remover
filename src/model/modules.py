import logging

import pytorch_lightning as pl
import pytorch_msssim as msssim
import torch
import torch.nn.functional as F

from src import const

from ..loss import TVLoss, VGGLoss
from .unet import HalfUNetDiscriminator, UNet

logger = logging.getLogger(__name__)


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
        self.manual_backward = True
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :3]

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


class UNetInfillOnlyModule(UNetModule):
    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str):
        x, y = batch
        y_hat = self(x)
        mask = x[:, -1]
        mask = torch.stack([mask] * 3, dim=1)
        y_hat = torch.where(mask == 0, x[:, :3], y_hat)
        loss = self.loss(y_hat, y, stage=stage)
        self.log(f"{stage}_loss", loss)
        return {
            "loss": loss,
            "pred": y_hat,
        }


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


class UNetTwoStage(BaseInpaintingModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.pre_unet = UNet(**kwargs)
        kwargs["in_channels"] = const.CHANNELS_OUT
        self.post_unet = UNet(**kwargs)
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mid = self.pre_unet(x)
        mid = torch.sigmoid(mid)
        final = self.post_unet(mid)
        final = torch.sigmoid(final)
        return mid, final

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str):
        x, y = batch
        mid, final = self(x)
        mid_loss = F.mse_loss(mid, y)

        self.log(f"{stage}_mid_loss", mid_loss)
        loss = self.loss(final, y, stage=stage)
        self.log(f"{stage}_loss", loss)
        if stage == "train":
            optimizer = self.optimizers()
            optimizer.zero_grad()
            self.manual_backward(mid_loss, retain_graph=True)
            self.manual_backward(loss)
            optimizer.step()

        return {
            "pred_mid": mid,
            "loss": loss,
            "pred": final,
        }


class GANInpainter(BaseInpaintingModule):
    def __init__(
        self,
        gan_adv_weight: float,
        **kwargs,
    ) -> None:
        super().__init__()
        self.generator = UNet(**kwargs)
        kwargs["in_channels"] = const.CHANNELS_OUT
        self.discriminator = HalfUNetDiscriminator(**kwargs)
        self.automatic_optimization = False
        self.gan_adv_weight = gan_adv_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.generator(x)
        return torch.sigmoid(x)

    def adv_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        g_opt, d_opt = self.optimizers()  # type: ignore

        # Generate fake image
        y_hat = self(x)

        # Real and Fake Labels
        real_labels = torch.ones((x.size(0), 1), device=self.device)
        fake_labels = torch.zeros((x.size(0), 1), device=self.device)

        # --- Step 1: Train Discriminator ---
        d_real = self.discriminator(y)
        d_fake = self.discriminator(y_hat.detach())

        d_loss_real = self.adv_loss(d_real, real_labels)
        d_loss_fake = self.adv_loss(d_fake, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) / 2

        self.toggle_optimizer(d_opt)
        self.manual_backward(d_loss)
        d_opt.step()
        d_opt.zero_grad()
        self.untoggle_optimizer(d_opt)

        # --- Step 2: Train Generator ---
        d_fake = self.discriminator(y_hat)  # Re-evaluate fake sample
        g_loss_adv = self.adv_loss(d_fake, real_labels)  # Generator tries to fool D

        # Reconstruction loss
        rec_loss = self.loss(y_hat, y, stage="traing")

        # Total Generator Loss
        g_loss = rec_loss + self.gan_adv_weight * g_loss_adv

        self.toggle_optimizer(g_opt)
        self.manual_backward(g_loss)
        g_opt.step()
        g_opt.zero_grad()
        self.untoggle_optimizer(g_opt)

        # Logging
        self.log("train_g_loss", g_loss)
        self.log("train_d_loss", d_loss)
        self.log("train_rec_loss", rec_loss)
        self.log("train_adv_loss", g_loss_adv)

        return {"loss": g_loss, "d_loss": d_loss, "pred": y_hat}

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(  # type: ignore
            self.generator.parameters(), lr=1e-4, betas=(0.5, 0.999)
        )
        d_opt = torch.optim.Adam(  # type: ignore
            self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999)
        )
        return [g_opt, d_opt], []


NAMES = {
    "unet": UNetModule,
    "unet_infill_only": UNetInfillOnlyModule,
    "unet_dual": UNetDualWithFFT,
    "unet_two_stage": UNetTwoStage,
    "gan": GANInpainter,
    "noop": NoOp,
}
