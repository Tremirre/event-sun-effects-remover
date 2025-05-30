from __future__ import annotations

import logging
import random
import warnings

import cv2
import numpy as np
import optuna
import PIL.Image
import pytorch_lightning.loggers as pl_loggers
import torch
from optuna.storages._cached_storage import _CachedStorage
from optuna.storages._rdb.storage import RDBStorage
from packaging import version

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def set_global_seed(seed: int):
    torch.backends.cudnn.deterministic = True
    logger.info(f"Setting global seed: {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_image_batch(
    event_gt: torch.Tensor,
    bgr_gt: torch.Tensor,
    bgr_with_artifact: torch.Tensor,
    bgr_inpaint: torch.Tensor,
    est_artifact_map: torch.Tensor,
    artifact_map: torch.Tensor,
    logger: pl_loggers.Logger,
    global_step: int,
    tag: str,
):
    """If not Neptune Logger -> skip"""
    if not isinstance(logger, pl_loggers.NeptuneLogger):
        return
    event_gt_np = event_gt.cpu().numpy()
    bgr_gt_np = bgr_gt.cpu().numpy()
    bgr_with_artifact_np = bgr_with_artifact.cpu().numpy()
    bgr_inpaint_np = bgr_inpaint.cpu().numpy()
    est_artifact_map_np = est_artifact_map.cpu().numpy()
    artifact_map_np = artifact_map.cpu().numpy()

    bgr_gt_np = (bgr_gt_np * 255).astype(np.uint8)
    bgr_gt_np = np.transpose(bgr_gt_np, (0, 2, 3, 1))

    bgr_with_artifact_np = (bgr_with_artifact_np * 255).astype(np.uint8)
    bgr_with_artifact_np = np.transpose(bgr_with_artifact_np, (0, 2, 3, 1))

    bgr_inpaint_np = (bgr_inpaint_np * 255).astype(np.uint8)
    bgr_inpaint_np = np.transpose(bgr_inpaint_np, (0, 2, 3, 1))

    event_gt_np = (event_gt_np * 255).astype(np.uint8)
    event_gt_np = np.transpose(event_gt_np, (0, 2, 3, 1))
    event_gt_np = np.repeat(event_gt_np, 3, axis=3)

    est_artifact_map_np = (est_artifact_map_np * 255).astype(np.uint8)
    est_artifact_map_np = np.transpose(est_artifact_map_np, (0, 2, 3, 1))
    est_artifact_map_np = np.repeat(est_artifact_map_np, 3, axis=3)

    artifact_map_np = (artifact_map_np * 255).astype(np.uint8)
    artifact_map_np = np.transpose(artifact_map_np, (0, 2, 3, 1))
    artifact_map_np = np.repeat(artifact_map_np, 3, axis=3)

    detection_result = np.concatenate(
        [bgr_with_artifact_np, est_artifact_map_np, artifact_map_np],
        axis=1,
    )
    detection_result = np.concatenate(detection_result, axis=1)
    detection_result = detection_result[:, :, ::-1]
    detection_result = cv2.resize(
        detection_result,
        (detection_result.shape[1] // 2, detection_result.shape[0] // 2),
        interpolation=cv2.INTER_LINEAR,
    )

    inpainting_result = np.concatenate(
        [bgr_with_artifact_np, bgr_inpaint_np, bgr_gt_np, event_gt_np],
        axis=1,
    )
    inpainting_result = np.concatenate(inpainting_result, axis=1)
    inpainting_result = inpainting_result[:, :, ::-1]
    inpainting_result = cv2.resize(
        inpainting_result,
        (inpainting_result.shape[1] // 2, inpainting_result.shape[0] // 2),
        interpolation=cv2.INTER_LINEAR,
    )

    detection_result_pil = PIL.Image.fromarray(detection_result)
    inpainting_result_pil = PIL.Image.fromarray(inpainting_result)

    # scale down 2 times

    logger.experiment[f"{tag}_det_comparison"].append(detection_result_pil)
    logger.experiment[f"{tag}_rec_comparison"].append(inpainting_result_pil)


# Define key names of `Trial.system_attrs`.
_EPOCH_KEY = "ddp_pl:epoch"
_INTERMEDIATE_VALUE = "ddp_pl:intermediate_value"
_PRUNED_KEY = "ddp_pl:pruned"

with optuna._imports.try_import() as _imports:
    import pytorch_lightning as pl
    from pytorch_lightning import LightningModule, Trainer
    from pytorch_lightning.callbacks import Callback


class PyTorchLightningPruningCallback(Callback):
    """PyTorch Lightning callback to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna-examples/blob/
    main/pytorch/pytorch_lightning_simple.py>`__
    if you want to add a pruning callback which observes accuracy.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``lightning.pytorch.LightningModule.training_step`` or
            ``lightning.pytorch.LightningModule.validation_epoch_end`` and the names thus depend on
            how this dictionary is formatted.


    .. note::
        For the distributed data parallel training, the version of PyTorchLightning needs to be
        higher than or equal to v1.6.0. In addition, :class:`~optuna.study.Study` should be
        instantiated with RDB storage.


    .. note::
        If you would like to use PyTorchLightningPruningCallback in a distributed training
        environment, you need to evoke ``PyTorchLightningPruningCallback.check_pruned()``
        manually so that :class:`~optuna.exceptions.TrialPruned` is properly handled.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        _imports.check()
        super().__init__()

        self._trial = trial
        self.monitor = monitor
        self.is_ddp_backend = False

    def on_fit_start(self, trainer: Trainer, pl_module: "pl.LightningModule") -> None:
        self.is_ddp_backend = trainer._accelerator_connector.is_distributed
        if self.is_ddp_backend:
            if version.parse(pl.__version__) < version.parse(  # type: ignore[attr-defined]
                "1.6.0"
            ):
                raise ValueError("PyTorch Lightning>=1.6.0 is required in DDP.")
            # If it were not for this block, fitting is started even if unsupported storage
            # is used. Note that the ValueError is transformed into ProcessRaisedException inside
            # torch.
            if not (
                isinstance(self._trial.study._storage, _CachedStorage)
                and isinstance(self._trial.study._storage._backend, RDBStorage)
            ):
                raise ValueError(
                    "optuna_integration.PyTorchLightningPruningCallback"
                    " supports only optuna.storages.RDBStorage in DDP."
                )
            # It is necessary to store intermediate values directly in the backend storage because
            # they are not properly propagated to main process due to cached storage.
            # TODO(Shinichi) Remove intermediate_values from system_attr after PR #4431 is merged.
            if trainer.is_global_zero:
                self._trial.storage.set_trial_system_attr(
                    self._trial._trial_id,
                    _INTERMEDIATE_VALUE,
                    dict(),
                )

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Trainer calls `on_validation_end` for sanity check. Therefore, it is necessary to avoid
        # calling `trial.report` multiple times at epoch 0. For more details, see
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                f"The metric '{self.monitor}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name."
            )
            warnings.warn(message)
            return

        epoch = pl_module.current_epoch
        should_stop = False

        # Determine if the trial should be terminated in a single process.
        if not self.is_ddp_backend:
            self._trial.report(current_score.item(), step=epoch)
            if not self._trial.should_prune():
                return
            raise optuna.TrialPruned(f"Trial was pruned at epoch {epoch}.")

        # Determine if the trial should be terminated in a DDP.
        if trainer.is_global_zero:
            self._trial.report(current_score.item(), step=epoch)
            should_stop = self._trial.should_prune()

            # Update intermediate value in the storage.
            _trial_id = self._trial._trial_id
            _study = self._trial.study
            _trial_system_attrs = _study._storage.get_trial_system_attrs(_trial_id)
            intermediate_values = _trial_system_attrs.get(_INTERMEDIATE_VALUE)
            intermediate_values[epoch] = current_score.item()  # type: ignore[index]
            self._trial.storage.set_trial_system_attr(
                self._trial._trial_id, _INTERMEDIATE_VALUE, intermediate_values
            )

        # Terminate every process if any world process decides to stop.
        should_stop = trainer.strategy.broadcast(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if not should_stop:
            return

        if trainer.is_global_zero:
            # Update system_attr from global zero process.
            self._trial.storage.set_trial_system_attr(
                self._trial._trial_id, _PRUNED_KEY, True
            )
            self._trial.storage.set_trial_system_attr(
                self._trial._trial_id, _EPOCH_KEY, epoch
            )

    def check_pruned(self) -> None:
        """Raise :class:`optuna.TrialPruned` manually if pruned.

        Currently, ``intermediate_values`` are not properly propagated between processes due to
        storage cache. Therefore, necessary information is kept in ``trial.system_attrs`` when the
        trial runs in a distributed situation. Please call this method right after calling
        ``lightning.pytorch.Trainer.fit()``.
        If a callback doesn't have any backend storage for DDP, this method does nothing.
        """

        _trial_id = self._trial._trial_id
        _study = self._trial.study
        # Confirm if storage is not InMemory in case this method is called in a non-distributed
        # situation by mistake.
        if not isinstance(_study._storage, _CachedStorage):
            return

        _trial_system_attrs = _study._storage._backend.get_trial_system_attrs(_trial_id)
        is_pruned = _trial_system_attrs.get(_PRUNED_KEY)
        intermediate_values = _trial_system_attrs.get(_INTERMEDIATE_VALUE)

        # Confirm if DDP backend is used in case this method is called from a non-DDP situation by
        # mistake.
        if intermediate_values is None:
            return
        for epoch, score in intermediate_values.items():
            self._trial.report(score, step=int(epoch))
        if is_pruned:
            epoch = _trial_system_attrs.get(_EPOCH_KEY)
            raise optuna.TrialPruned(f"Trial was pruned at epoch {epoch}.")


def tensor_to_numpy_img(tensor: torch.Tensor) -> np.ndarray:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    assert tensor.ndim == 4
    tensor = tensor.permute(0, 2, 3, 1)  # (B, H, W, C)
    np_arr = (tensor.detach().cpu().numpy() * 255.0).astype(np.uint8)
    return np_arr
