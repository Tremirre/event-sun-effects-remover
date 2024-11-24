import argparse
import dataclasses
import logging
import pathlib

import dotenv
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers, profilers
from torch.profiler import tensorboard_trace_handler

from src import const
from src.callbacks.ref_logger import ReferenceImageLogger
from src.data import datamodule
from src.model import noop, unet

torch.set_float32_matmul_precision("medium")
torch.manual_seed(0)
np.random.seed(0)

dotenv.load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s-%(name)s-%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


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
    run_tags: str = "default"
    ref_path: pathlib.Path = None

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
        parser.add_argument(
            "--run-tags",
            type=str,
            default="default",
            help="Tags for the run, comma-separated",
        )
        parser.add_argument(
            "--ref-path",
            type=pathlib.Path,
            default=None,
            help="Path to reference images",
        )
        return cls(**vars(parser.parse_args()))


def logger_from_config(config: Config) -> loggers.Logger:
    if config.log_tensorboard:
        return loggers.TensorBoardLogger(
            "lightning_logs",
            name="unet",
        )
    return loggers.NeptuneLogger(
        log_model_checkpoints=False, tags=config.run_tags.split(",")
    )


def profiler_from_config(config: Config) -> profilers.Profiler | None:
    if config.profile:
        return profilers.PyTorchProfiler(
            on_trace_ready=tensorboard_trace_handler("lightning_logs/profiler0"),
            profile_memory=True,
            record_shapes=True,
            record_functions=True,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        )
    return None


def model_from_config(config: Config) -> pl.LightningModule:
    if config.unet_blocks > 0:
        return unet.UNet(config.unet_blocks, config.unet_depth)
    logger.info("Using NoOp model")
    return noop.NoOp()


if __name__ == "__main__":
    config = Config.from_args()

    dm = datamodule.EventDataModule(
        const.DATA_FOLDER,
        batch_size=config.batch_size,
        frac_used=config.frac_used,
        num_workers=config.num_workers,
    )
    run_logger = logger_from_config(config)
    profiler = profiler_from_config(config)
    callbacks = []
    if config.ref_path:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        callbacks.append(ReferenceImageLogger(config.ref_path, device))

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        logger=run_logger,
        profiler=profiler,
        log_every_n_steps=10,
    )
    dataset_sizes = dm.get_dataset_sizes()
    config_dict = dataclasses.asdict(config)
    trainer.logger.log_hyperparams(config_dict)
    trainer.logger.log_hyperparams(dataset_sizes)

    model = model_from_config(config)
    trainer.fit(model, dm)
    if config.unet_blocks:
        trainer.test(model, dm)

    run_logger.experiment.stop()
