import dataclasses
import logging

import dotenv
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers

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
    )
    run_logger = config.get_logger()
    profiler = config.get_profiler()
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

    model = config.get_model()
    trainer.fit(model, dm)
    if config.unet_blocks:
        utils.set_global_seed(1)
        trainer.test(model, dm)

    if config.save:
        logger.info("Saving model")
        torch.save(model.state_dict(), f"model_{RUN_IDX}.pth")
        if isinstance(run_logger, loggers.NeptuneLogger):
            logger.info("Uploading model to Neptune")
            run_logger.experiment["model"].upload(f"model_{RUN_IDX}.pth")
    run_logger.experiment.stop()
