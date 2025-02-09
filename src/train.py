import dataclasses
import logging

import dotenv
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from src import const, utils
from src.callbacks import image_loggers
from src.config import Config
from src.data import datamodule

RUN_IDX = np.random.randint(0, 2**31)

torch.set_float32_matmul_precision("medium")
utils.set_global_seed(0)

dotenv.load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s %(name)s %(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    config = Config.from_args()

    dm = datamodule.EventDataModule(
        const.TRAIN_VAL_TEST_DIR,
        const.REF_DIR,
        batch_size=config.batch_size,
        frac_used=config.frac_used,
        num_workers=config.num_workers,
        ref_threshold=config.diff_intensity,
        sep_event_channel=config.event_channel,
        train_img_glob=config.img_glob,
        progressive_masking=config.progressive_masking,
        soft_masking=config.soft_masking,
        yuv_interpolation=config.yuv_interpolation,
    )
    run_logger = config.get_logger()
    profiler = config.get_profiler()
    callbacks: list[pl.Callback] = [image_loggers.ValBatchImageLogger()]
    if dm.ref_paths:
        dm.setup("ref")
        logger.info("Enabling reference image logging")
        ref_img_logger = image_loggers.ReferenceImageLogger(
            dm.ref_dataloader(), infill_only="infill" in config.module_type
        )
        callbacks.append(ref_img_logger)

    model_chkp = None
    if config.save:
        logger.info("Enabling model checkpointing")
        model_chkp = ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints",
            filename="{run_idx:0>10}-model-{epoch}".format(
                run_idx=RUN_IDX, epoch="{epoch}"
            ),
            save_top_k=1,
            mode="min",
            save_weights_only=True,
        )
        callbacks.append(model_chkp)

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        logger=run_logger,
        profiler=profiler,
        log_every_n_steps=1,
        callbacks=callbacks,
    )
    dataset_sizes = dm.get_dataset_sizes()
    logger.info(f"Dataset sizes: {dataset_sizes}")
    config_dict = dataclasses.asdict(config)
    trainer.logger.log_hyperparams(config_dict)  # type: ignore
    trainer.logger.log_hyperparams(dataset_sizes)  # type: ignore
    trainer.logger.log_hyperparams({"run_idx": RUN_IDX})  # type: ignore

    model = config.get_model()
    trainer.fit(model, dm)

    if model_chkp is not None:
        logger.info(f"Testing best model from checkpoint: {model_chkp.best_model_path}")
        model.load_state_dict(
            torch.load(model_chkp.best_model_path, weights_only=True)["state_dict"]
        )
    utils.set_global_seed(1)
    trainer.test(model, dm)

    if model_chkp is not None:
        best_model_path = model_chkp.best_model_path
        if isinstance(run_logger, loggers.NeptuneLogger):
            logger.info("Uploading model to Neptune")
            run_logger.experiment["model"].upload(best_model_path)
    run_logger.experiment.stop()
