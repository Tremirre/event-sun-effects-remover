import argparse
import dataclasses

import dotenv
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers, profilers
from torch.profiler import tensorboard_trace_handler

from src import const
from src.data import datamodule
from src.model import unet

torch.set_float32_matmul_precision("medium")
torch.manual_seed(0)
np.random.seed(0)

dotenv.load_dotenv()


@dataclasses.dataclass
class Config:
    batch_size: int
    unet_blocks: int
    unet_depth: int
    max_epochs: int
    frac_used: float
    num_workers: int = 0
    profile: bool = False
    log_tensorboard: bool = False

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
        parser.add_argument(
            "--unet-blocks", type=int, default=2, help="Number of U-Net blocks"
        )
        parser.add_argument(
            "--unet-depth",
            type=int,
            default=1,
            help="Depth of U-Net blocks (number of 1-conv layers per block)",
        )
        parser.add_argument(
            "--max-epochs", type=int, default=10, help="Number of epochs"
        )
        parser.add_argument(
            "--frac-used", type=float, default=1, help="Fraction of data used"
        )
        parser.add_argument(
            "--num-workers", type=int, default=0, help="Number of workers"
        )
        parser.add_argument("--profile", action="store_true", help="Enable profiling")
        parser.add_argument(
            "--log-tensorboard", action="store_true", help="Log to TensorBoard"
        )
        return cls(**vars(parser.parse_args()))


if __name__ == "__main__":
    config = Config.from_args()

    dm = datamodule.EventDataModule(
        const.DATA_FOLDER,
        batch_size=config.batch_size,
        frac_used=config.frac_used,
        num_workers=config.num_workers,
    )
    if config.log_tensorboard:
        logger = loggers.TensorBoardLogger(
            "lightning_logs",
            name="unet",
        )
    else:
        logger = loggers.NeptuneLogger(log_model_checkpoints=False)
    logger.experiment["metadata/config"] = dataclasses.asdict(config)
    profiler = None
    if config.profile:
        profiler = profilers.PyTorchProfiler(
            on_trace_ready=tensorboard_trace_handler("lightning_logs/profiler0"),
            profile_memory=True,
            record_shapes=True,
            record_functions=True,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        )
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        logger=logger,
        profiler=profiler,
        log_every_n_steps=10,
    )
    if trainer.logger is None:
        print(config)
    else:
        trainer.logger.log_hyperparams(dataclasses.asdict(config))

    model = unet.UNet(config.unet_blocks, config.unet_depth)

    trainer.fit(model, dm)
    trainer.test(model, dm)

    logger.experiment.stop()
