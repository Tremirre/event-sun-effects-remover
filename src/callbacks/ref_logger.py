import pathlib

import numpy as np
import PIL.Image
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch


def log_image_batch(
    x: torch.Tensor,
    y_hat: torch.Tensor,
    y: torch.Tensor | None,
    logger: pl_loggers.Logger,
    global_step: int,
    tag: str,
):
    x_bgr = x[:, :3]
    x_event = x[:, 3:4]
    x_event = torch.cat([x_event, x_event, x_event], dim=1)
    x_mask = x[:, 4:5]
    x_mask = torch.cat([x_mask, x_mask, x_mask], dim=1)
    x_bgr = torch.where(x_mask > 0, x_event, x_bgr)
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


class ReferenceImageLogger(pl.Callback):
    def __init__(self, ref_path: pathlib.Path, device: torch.device) -> None:
        super().__init__()
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference image {ref_path} does not exist")
        refs = []
        for ref_obj in ref_path.glob("*.npy"):
            ref = np.load(ref_obj)
            refs.append(ref)
        self.refs = torch.tensor(refs).float()
        self.refs = self.refs.to(device)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        pl_module.eval()
        with torch.no_grad():
            preds = pl_module(self.refs)
            log_image_batch(
                self.refs, preds, None, trainer.logger, trainer.global_step, "ref"
            )
