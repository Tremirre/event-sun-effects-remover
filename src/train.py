import pytorch_lightning as pl
import torch

from src import const
from src.data import datamodule
from src.model import unet

torch.set_float32_matmul_precision("medium")
if __name__ == "__main__":
    dm = datamodule.EventDataModule(
        const.DATA_FOLDER,
    )
    model = unet.UNet(3)
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, dm)
    trainer.test(model, dm)
