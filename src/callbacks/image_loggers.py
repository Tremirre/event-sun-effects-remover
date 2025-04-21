import logging

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src import const
from src.utils import log_image_batch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ReferenceImageLogger(pl.Callback):
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader

    def log_ref_images(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        pl_module.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.loader):
                x, y = batch
                x = x.to(pl_module.device)
                y = y.to(pl_module.device)
                est_artifact_map, bgr_inpaint = pl_module(x)
                artifact_map = y[:, 3:4]
                bgr_gt = y[:, :3]
                bgr_with_artifact = x[:, :3]
                event_gt = x[:, 3:4]

                log_image_batch(
                    event_gt,
                    bgr_gt,
                    bgr_with_artifact,
                    bgr_inpaint,
                    est_artifact_map,
                    artifact_map,
                    trainer.logger,  # type: ignore
                    trainer.global_step,
                    f"ref_{i}",
                )

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if (
            trainer.current_epoch % const.REFERENCE_LOGGING_FREQ != 0
            or trainer.current_epoch == 0
        ):
            return
        logger.info("Logging reference images")
        self.log_ref_images(trainer, pl_module)


class ValBatchImageLogger(pl.Callback):
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        if batch_idx != 0:
            return

        if trainer.current_epoch % const.VALIDATION_LOGGING_FREQ != 0:
            return

        logger.info("Logging validation batch")
        x, y = batch
        x = x[:8]
        y = y[:8]
        est_artifact_map = outputs["artifact_map"][:8]  # type: ignore
        bgr_inpaint = outputs["inpaint_out"][:8]  # type: ignore
        artifact_map = y[:, 3:4][:8]
        bgr_gt = y[:, :3][:8]
        bgr_with_artifact = x[:, :3][:8]
        event_gt = x[:, 3:4][:8]
        log_image_batch(
            event_gt,
            bgr_gt,
            bgr_with_artifact,
            bgr_inpaint,
            est_artifact_map,
            artifact_map,
            trainer.logger,  # type: ignore
            trainer.global_step,
            "val_batch",
        )
