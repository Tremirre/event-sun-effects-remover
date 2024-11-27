import argparse
import dataclasses
import logging

import dotenv
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers, profilers
from torch.profiler import tensorboard_trace_handler

from src import const
from src.callbacks import image_loggers
from src.data import datamodule
from src.model import noop, unet

torch.set_float32_matmul_precision("medium")
torch.manual_seed(0)
np.random.seed(0)

dotenv.load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s %(name)s %(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Config:
    batch_size: int
    unet_blocks: int
    unet_depth: int
    unet_kernel: int
    unet_fft: bool
    max_epochs: int
    frac_used: float
    num_workers: int = 0
    profile: bool = False
    log_tensorboard: bool = False
    run_tags: str = "default"
    save: bool = False

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
            "--unet-kernel", type=int, default=3, help="Kernel size of U-Net blocks"
        )
        parser.add_argument(
            "--unet-fft",
            action="store_true",
            help="Use Fourier Convolution in U-Net blocks",
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
        parser.add_argument("--save", action="store_true", help="Save the model")
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
        return unet.UNet(
            config.unet_blocks, config.unet_depth, config.unet_kernel, config.unet_fft
        )
    logger.info("Using NoOp model")
    return noop.NoOp()


if __name__ == "__main__":
    config = Config.from_args()

    dm = datamodule.EventDataModule(
        const.TRAIN_VAL_TEST_DIR,
        const.REF_DIR,
        batch_size=config.batch_size,
        frac_used=config.frac_used,
        num_workers=config.num_workers,
    )
    run_logger = logger_from_config(config)
    profiler = profiler_from_config(config)
    callbacks = [image_loggers.ValBatchImageLogger()]
    if dm.ref_paths:
        dm.setup("ref")
        logger.info("Enabling reference image logging")
        ref_img_logger = image_loggers.ReferenceImageLogger(dm.ref_dataloader())
        callbacks.append(ref_img_logger)

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        logger=run_logger,
        profiler=profiler,
        log_every_n_steps=1,
        callbacks=callbacks,
    )
    dataset_sizes = dm.get_dataset_sizes()
    config_dict = dataclasses.asdict(config)
    trainer.logger.log_hyperparams(config_dict)
    trainer.logger.log_hyperparams(dataset_sizes)

    model = model_from_config(config)
    trainer.fit(model, dm)
    if config.unet_blocks:
        trainer.test(model, dm)

    if config.save:
        logger.info("Saving model")
        torch.save(model.state_dict(), "model.pth")
        if isinstance(run_logger, loggers.NeptuneLogger):
            logger.info("Uploading model to Neptune")
            run_logger.experiment["model"].upload("model.pth")
    run_logger.experiment.stop()
