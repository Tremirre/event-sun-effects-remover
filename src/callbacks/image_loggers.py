import logging

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.utils import log_image_batch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ReferenceImageLogger(pl.Callback):
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        logger.info("Logging reference images")
        pl_module.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.loader):
                x, y = batch
                y_hat = pl_module(x)
                log_image_batch(
                    x, y_hat, y, trainer.logger, trainer.global_step, f"ref_{i}"
                )


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

        logger.info("Logging validation batch")
        x, y = batch
        y_hat = outputs["pred"]
        log_image_batch(x, y_hat, y, trainer.logger, trainer.global_step, "val_0")
