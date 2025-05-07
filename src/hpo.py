import argparse

import neptune
import numpy as np
import optuna
import pytorch_lightning as pl
import torch
from neptune.integrations.optuna import NeptuneCallback

from . import const
from .data import datamodule
from .model import combiners, models, modules
from .utils import PyTorchLightningPruningCallback, set_global_seed

FRAC_USED = 0.1
EPOCHS = 50
torch.set_float32_matmul_precision("medium")


def get_data_module(trial: optuna.Trial) -> datamodule.BaseDataModule:
    train_paths = sorted(const.TRAIN_DIR.glob(const.DATA_PATTERN))  # type: ignore
    val_paths = sorted(const.VAL_DIR.glob(const.DATA_PATTERN))  # type: ignore

    np.random.shuffle(train_paths)  # type: ignore
    np.random.shuffle(val_paths)  # type: ignore
    train_paths = train_paths[: int(len(train_paths) * FRAC_USED)]
    val_paths = val_paths[: int(len(val_paths) * FRAC_USED)]

    assert train_paths, "Train paths not found"
    assert val_paths, "Val paths not found"

    return datamodule.JointDataModule(
        train_paths=train_paths,
        val_paths=val_paths,
        test_paths=[],
        ref_paths=[],
        batch_size=trial.suggest_int("batch_size", 1, 16),
        num_workers=4,
        p_sun=trial.suggest_float("p_sun", 0.0, 1.0),
        p_glare=trial.suggest_float("p_glare", 0.0, 1.0),
        p_flare=trial.suggest_float("p_flare", 0.0, 1.0),
        p_hq_flare=trial.suggest_float("p_hq_flare", 0.0, 1.0),
    )


def get_model(trial: optuna.Trial) -> modules.DetectorInpainterModule:
    detector = models.UNet(
        n_blocks=trial.suggest_int("detector_n_blocks", 3, 4),
        block_depth=trial.suggest_int("detector_block_depth", 1, 4),
        in_channels=3,
        kernel_size=trial.suggest_int("detector_kernel_size_half", 1, 3) * 2 + 1,
        activation_func=trial.suggest_categorical(
            "detector_activation_func", ["relu", "leakyrelu", "gelu", "elu", "mish"]
        ),
        batch_norm=trial.suggest_categorical("detector_batch_norm", [True, False]),
        with_fft=trial.suggest_categorical("detector_with_fft", [True, False]),
        out_channels=1,
    )
    combiner_name = trial.suggest_categorical(
        "combiner",
        [
            "masked_removal",
            "simple_concat",
            "convolutional",
        ],
    )
    if combiner_name == "masked_removal":
        combiner = combiners.MaskedRemovalCombiner(
            binarize=trial.suggest_categorical("binarize", [True, False]),  # type: ignore
            yuv_interpolation=trial.suggest_categorical(
                "yuv_interpolation",
                [True, False],  # type: ignore
            ),
            soft_factor=trial.suggest_categorical("soft_factor", [True, False]),  # type: ignore
        )
    elif combiner_name == "convolutional":
        combiner = combiners.ConvolutionalCombiner(
            out_channels=trial.suggest_int("combiner_out_channels", 3, 6),
            depth=trial.suggest_int("combiner_depth", 1, 4),
            kernel_size=trial.suggest_int("combiner_kernel_size_half", 1, 3) * 2 + 1,
        )
    else:
        combiner = combiners.get_combiner(combiner_name)

    inpainter = models.UNet(
        n_blocks=trial.suggest_int("inpainter_n_blocks", 3, 4),
        block_depth=trial.suggest_int("inpainter_block_depth", 1, 4),
        in_channels=combiner.get_output_channels(),
        kernel_size=trial.suggest_int("inpainter_kernel_size_half", 1, 3) * 2 + 1,
        activation_func=trial.suggest_categorical(
            "inpainter_activation_func", ["relu", "leakyrelu", "gelu", "elu", "mish"]
        ),
        batch_norm=trial.suggest_categorical("inpainter_batch_norm", [True, False]),
        with_fft=trial.suggest_categorical("inpainter_with_fft", [True, False]),
        out_channels=const.CHANNELS_OUT,
    )
    return modules.DetectorInpainterModule(
        detector=detector,
        combiner=combiner,
        inpainter=inpainter,
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        apply_non_mask_penalty=trial.suggest_categorical(
            "apply_non_mask_penalty", [True, False]
        ),
    )


def objective(trial: optuna.Trial) -> float:
    set_global_seed(trial.number)
    data_module = get_data_module(trial)
    model = get_model(trial)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=1,
        logger=None,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor="val_inpaint_loss"),
        ],
    )

    trainer.fit(model, data_module)
    val_loss = trainer.callback_metrics["val_loss"].item()
    return val_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization")
    parser.add_argument(
        "--study_name",
        type=str,
        default=".hpo",
        help="Name of the Optuna study",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of trials to run",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run = neptune.init_run()
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1,
        reduction_factor=2,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage="sqlite:///hpo.db",
        load_if_exists=True,
        pruner=pruner,
        direction="minimize",
    )
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=None,
        show_progress_bar=True,
        callbacks=[NeptuneCallback(run)],  # type: ignore
    )
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_trial.value}")
    print(f"Best params: {study.best_trial.params}")
