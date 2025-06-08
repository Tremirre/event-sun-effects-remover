import logging

import pytorch_lightning as pl
import pytorch_msssim as msssim
import torch
import torch.nn.functional as F

from src.loss import TVLoss, VGGLoss

logger = logging.getLogger(__name__)


class BaseModule(pl.LightningModule):
    def _shared_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, stage: str
    ):
        raise NotImplementedError

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self._shared_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)  # type: ignore


class DetectorInpainterModule(BaseModule):
    def __init__(
        self,
        detector: torch.nn.Module,
        combiner: torch.nn.Module,
        inpainter: torch.nn.Module,
        learning_rate: float = 1e-4,
        apply_non_mask_penalty: bool = False,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.combiner = combiner
        self.inpainter = inpainter
        self.tv_loss = TVLoss(2)
        self.vgg_loss = VGGLoss()
        self.apply_non_mask_penalty = apply_non_mask_penalty
        self.learning_rate = learning_rate
        if apply_non_mask_penalty:
            logger.info("Non-mask penalty will be applied")

    def detector_loss(
        self, artifact_map: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return F.huber_loss(artifact_map.squeeze(1), target, delta=0.1)

    def inpainter_loss(
        self, inpaint_out: torch.Tensor, target: torch.Tensor, stage: str = ""
    ) -> torch.Tensor:
        mae = F.l1_loss(inpaint_out, target)
        ssim = 1 - msssim.ms_ssim(inpaint_out, target, data_range=1.0)
        vgg_loss = self.vgg_loss(inpaint_out, target)
        tv_loss = self.tv_loss(inpaint_out)
        if stage:
            self.log(f"{stage}_mae", mae, on_epoch=True)
            self.log(f"{stage}_ssim", ssim, on_epoch=True)
            self.log(f"{stage}_vgg_loss", vgg_loss, on_epoch=True)
            self.log(f"{stage}_tv_loss", tv_loss, on_epoch=True)
        return 1.0 * mae + 0.5 * ssim + 0.1 * vgg_loss + 0.05 * tv_loss

    def non_mask_edit_loss(
        self,
        inpaint_in: torch.Tensor,
        inpaint_out: torch.Tensor,
        artifact_map: torch.Tensor,
    ) -> torch.Tensor:
        inpaint_in_masked = inpaint_in * (1 - artifact_map)
        inpaint_out_masked = inpaint_out * (1 - artifact_map)
        mae = F.l1_loss(inpaint_out_masked, inpaint_in_masked)
        ssim = 1 - msssim.ms_ssim(inpaint_out_masked, inpaint_in_masked, data_range=1.0)
        return mae + 0.5 * ssim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # input Bx4xHxW
        # channels: BGR + Event Reconstruction
        # output [BxHxW, Bx3xHxW]
        # channels: BGR

        artifact_map = F.sigmoid(self.detector(x[:, :3, :, :]))
        inpaint_input = self.combiner(x, artifact_map)
        inpaint_out = F.sigmoid(self.inpainter(inpaint_input))
        return artifact_map, inpaint_out

    def _shared_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        stage: str,
    ) -> dict[str, torch.Tensor]:
        x, y = batch
        artifact_map, inpaint_out = self(x)
        detection_loss = torch.tensor(0.0, device=x.device)  # Default to zero

        if any(p.requires_grad for p in self.detector.parameters()):
            detection_loss = self.detector_loss(artifact_map, y[:, 3, :, :])
        inpaint_loss = self.inpainter_loss(inpaint_out, y[:, :3, :, :], stage=stage)
        total_loss = detection_loss + inpaint_loss
        non_mask_res = {}
        if self.apply_non_mask_penalty:
            non_mask_edit_loss = self.non_mask_edit_loss(
                x[:, :3, :, :], inpaint_out, artifact_map
            )
            self.log(f"{stage}_non_mask_edit_loss", non_mask_edit_loss, on_epoch=True)
            total_loss += non_mask_edit_loss
            non_mask_res = {"non_mask_edit_loss": non_mask_edit_loss}
        self.log(f"{stage}_total_loss", total_loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_detection_loss", detection_loss, on_epoch=True)
        self.log(f"{stage}_inpaint_loss", inpaint_loss, on_epoch=True)

        return {
            "loss": total_loss,
            "detection_loss": detection_loss,
            "inpaint_loss": inpaint_loss,
            "artifact_map": artifact_map,
            "inpaint_out": inpaint_out,
            **non_mask_res,
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # type: ignore
