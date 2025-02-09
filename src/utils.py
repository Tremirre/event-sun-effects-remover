import logging
import random

import numpy
import PIL.Image
import pytorch_lightning.loggers as pl_loggers
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def set_global_seed(seed: int):
    logger.info(f"Setting global seed: {seed}")
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)


def log_image_batch(
    x: torch.Tensor,
    y_hat: torch.Tensor,
    y: torch.Tensor | None,
    logger: pl_loggers.Logger,
    global_step: int,
    tag: str,
):
    x_bgr = x[:, :3]
    if x.shape[1] == 5:
        x_event = x[:, 3:4]
        x_event = torch.cat([x_event, x_event, x_event], dim=1)
        x_mask = x[:, 4:5]
        x_event = x_event.astype(torch.float32) / 255.0
        x_mask = torch.cat([x_mask, x_mask, x_mask], dim=1)
        x_bgr = x_bgr.astype(torch.float32) / 255.0
        x_bgr = (1 - x_mask) * x_bgr * 255.0 + x_mask * x_event * 255.0
        x_bgr = x_bgr.astype(torch.uint8)
    if isinstance(logger, pl_loggers.TensorBoardLogger):
        logger.experiment.add_images(f"{tag}_input", x_bgr, global_step=global_step)
        logger.experiment.add_images(f"{tag}_output", y_hat, global_step=global_step)
        if y is not None:
            logger.experiment.add_images(f"{tag}_target", y, global_step=global_step)
    elif isinstance(logger, pl_loggers.NeptuneLogger):
        x_bgr = torch.dstack(x_bgr.unbind(0))
        x_bgr = x_bgr.permute(1, 2, 0)
        y_hat = torch.dstack(y_hat.unbind(0))
        y_hat = y_hat.permute(1, 2, 0)
        comp_vals = [x_bgr, y_hat]
        if y is not None:
            y = torch.dstack(y.unbind(0))
            y = y.permute(1, 2, 0)
            comp_vals.append(y)

        comparison = torch.cat(comp_vals, dim=0)
        comparison = comparison.cpu().numpy()
        comparison = (comparison * 255).astype("uint8")
        comparison = comparison[:, :, ::-1]
        comparison = PIL.Image.fromarray(comparison)
        logger.experiment[f"{tag}_comparison"].append(comparison)
