import argparse
import enum
import functools
import typing

import dotenv
import numpy as np
import optuna
import pytorch_lightning as pl
import torch

from . import const
from .data import datamodule
from .model import combiners, models, modules
from .utils import PyTorchLightningPruningCallback, set_global_seed

FRAC_USED = 0.1
EPOCHS = 50
torch.set_float32_matmul_precision("medium")
dotenv.load_dotenv()


class HPOType(enum.Enum):
    DETECTOR = "detector"
    INPAINTER = "inpainter"
    COMBINER = "combiner"
    OTHER = "other"


def param_from_type(
    cbk: typing.Callable, src_type: HPOType, trgt_type: HPOType, default: typing.Any
) -> typing.Any:
    if src_type == trgt_type:
        return cbk()
    return default


def get_data_module(trial: optuna.Trial, h_type: HPOType) -> datamodule.BaseDataModule:
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
        batch_size=8,
        num_workers=4,
        p_sun=param_from_type(
            lambda: trial.suggest_float("p_sun", 0.0, 1.0), h_type, HPOType.OTHER, 0.4
        ),
        p_glare=param_from_type(
            lambda: trial.suggest_float("p_glare", 0.0, 1.0), h_type, HPOType.OTHER, 0.4
        ),
        p_flare=param_from_type(
            lambda: trial.suggest_float("p_flare", 0.0, 1.0), h_type, HPOType.OTHER, 0.4
        ),
        p_hq_flare=param_from_type(
            lambda: trial.suggest_float("p_hq_flare", 0.0, 1.0),
            h_type,
            HPOType.OTHER,
            0.4,
        ),
        p_overlit=param_from_type(
            lambda: trial.suggest_float("p_overlit", 0.0, 1.0),
            h_type,
            HPOType.OTHER,
            0.4,
        ),
    )


def get_model(trial: optuna.Trial, h_type: HPOType) -> modules.DetectorInpainterModule:
    detector = models.UNet(
        n_blocks=param_from_type(
            lambda: trial.suggest_int("detector_n_blocks", 3, 4),
            h_type,
            HPOType.DETECTOR,
            3,
        ),
        block_depth=param_from_type(
            lambda: trial.suggest_int("detector_block_depth", 1, 4),
            h_type,
            HPOType.DETECTOR,
            2,
        ),
        in_channels=3,
        kernel_size=param_from_type(
            lambda: trial.suggest_int("detector_kernel_size_half", 1, 3) * 2 + 1,
            h_type,
            HPOType.DETECTOR,
            1,
        ),
        activation_func=param_from_type(
            lambda: trial.suggest_categorical(
                "detector_activation_func", ["relu", "leakyrelu", "gelu", "elu", "mish"]
            ),
            h_type,
            HPOType.DETECTOR,
            "relu",
        ),
        batch_norm=param_from_type(
            lambda: trial.suggest_categorical("detector_batch_norm", [True, False]),
            h_type,
            HPOType.DETECTOR,
            True,
        ),
        with_fft=param_from_type(
            lambda: trial.suggest_categorical("detector_with_fft", [True, False]),
            h_type,
            HPOType.DETECTOR,
            False,
        ),
        out_channels=1,
    )
    combiner_name = param_from_type(
        lambda: trial.suggest_categorical(
            "combiner",
            [
                "masked_removal",
                "simple_concat",
                "convolutional",
            ],
        ),
        h_type,
        HPOType.COMBINER,
        "simple_concat",
    )
    if combiner_name == "masked_removal":
        combiner = combiners.MaskedRemovalCombiner(
            binarize=param_from_type(
                lambda: trial.suggest_categorical("binarize", [True, False]),  # type: ignore
                h_type,
                HPOType.COMBINER,
                False,
            ),
            yuv_interpolation=param_from_type(
                lambda: trial.suggest_categorical(
                    "yuv_interpolation",
                    [True, False],  # type: ignore
                ),
                h_type,
                HPOType.COMBINER,
                False,
            ),
            soft_factor=param_from_type(
                lambda: trial.suggest_categorical("soft_factor", [10, 0]),  # type: ignore
                h_type,
                HPOType.COMBINER,
                0,
            ),
        )
    elif combiner_name == "convolutional":
        combiner = combiners.ConvolutionalCombiner(
            out_channels=param_from_type(
                lambda: trial.suggest_int("combiner_out_channels", 3, 6),
                h_type,
                HPOType.COMBINER,
                3,
            ),
            depth=param_from_type(
                lambda: trial.suggest_int("combiner_depth", 1, 4),
                h_type,
                HPOType.COMBINER,
                2,
            ),
            kernel_size=param_from_type(
                lambda: trial.suggest_int("combiner_kernel_size_half", 1, 3) * 2 + 1,
                h_type,
                HPOType.COMBINER,
                1,
            ),
        )
    else:
        combiner = combiners.get_combiner(combiner_name)

    if h_type == HPOType.DETECTOR:
        inpainter = models.NoOp(out_channels=const.CHANNELS_OUT)
    else:
        inpainter = models.UNet(
            n_blocks=param_from_type(
                lambda: trial.suggest_int("inpainter_n_blocks", 3, 4),
                h_type,
                HPOType.INPAINTER,
                3,
            ),
            block_depth=param_from_type(
                lambda: trial.suggest_int("inpainter_block_depth", 1, 4),
                h_type,
                HPOType.INPAINTER,
                2,
            ),
            in_channels=combiner.get_output_channels(),
            kernel_size=param_from_type(
                lambda: trial.suggest_int("inpainter_kernel_size_half", 1, 3) * 2 + 1,
                h_type,
                HPOType.INPAINTER,
                1,
            ),
            activation_func=param_from_type(
                lambda: trial.suggest_categorical(
                    "inpainter_activation_func",
                    ["relu", "leakyrelu", "gelu", "elu", "mish"],
                ),
                h_type,
                HPOType.INPAINTER,
                "relu",
            ),
            batch_norm=param_from_type(
                lambda: trial.suggest_categorical(
                    "inpainter_batch_norm", [True, False]
                ),
                h_type,
                HPOType.INPAINTER,
                True,
            ),
            with_fft=param_from_type(
                lambda: trial.suggest_categorical("inpainter_with_fft", [True, False]),
                h_type,
                HPOType.INPAINTER,
                False,
            ),
            out_channels=const.CHANNELS_OUT,
        )
    return modules.DetectorInpainterModule(
        detector=detector,
        combiner=combiner,
        inpainter=inpainter,
        learning_rate=param_from_type(
            lambda: trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            h_type,
            HPOType.OTHER,
            1e-3,
        ),
        apply_non_mask_penalty=param_from_type(
            lambda: trial.suggest_categorical("apply_non_mask_penalty", [True, False]),
            h_type,
            HPOType.OTHER,
            False,
        ),
    )


def objective(trial: optuna.Trial, h_type: HPOType) -> float:
    set_global_seed(trial.number)
    data_module = get_data_module(trial, h_type)
    model = get_model(trial, h_type)
    target = "val_inpaint_loss" if h_type != HPOType.DETECTOR else "val_det_loss"
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=1,
        logger=None,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor=target),
        ],
    )
    trainer.fit(model, data_module)
    val_loss = trainer.callback_metrics[target].item()
    return val_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization")
    parser.add_argument(
        "--name",
        type=str,
        help="Name of the Optuna study",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of trials to run",
    )
    parser.add_argument(
        "--type",
        type=HPOType,
        choices=list(HPOType),
        help="Type of HPO to run",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1,
        reduction_factor=2,
    )
    study = optuna.create_study(
        study_name=args.name,
        storage=f"sqlite:///{args.name}-{args.type.name}-hpo.db",
        load_if_exists=True,
        pruner=pruner,
        direction="minimize",
    )
    study.optimize(
        functools.partial(objective, h_type=args.type),
        n_trials=args.n_trials,
        timeout=None,
        show_progress_bar=True,
        gc_after_trial=True,
    )
