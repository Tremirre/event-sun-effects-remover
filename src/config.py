import argparse
import dataclasses
import pathlib

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers, profilers
from torch.profiler import tensorboard_trace_handler

from src import const
from src.data import datamodule
from src.model.combiners import get_combiner
from src.model.models import get_model
from src.model.modules import DetectorInpainterModule


@dataclasses.dataclass
class Config:
    batch_size: int
    max_epochs: int
    frac_used: float
    detect_model: str
    combiner_model: str
    inpainter_model: str
    detector_kwargs: dict[str, str | int | bool] = dataclasses.field(
        default_factory=dict
    )
    combiner_kwargs: dict[str, str | int | bool] = dataclasses.field(
        default_factory=dict
    )
    inpainter_kwargs: dict[str, str | int | bool] = dataclasses.field(
        default_factory=dict
    )
    img_glob: str = "**/*.npy"
    num_workers: int = 0
    profile: bool = False
    log_tensorboard: bool = False
    run_tags: str = "default"
    save: bool = False
    ref_dir: pathlib.Path = dataclasses.field(default_factory=lambda: const.REF_DIR)
    train_dir: pathlib.Path = dataclasses.field(default_factory=lambda: const.REF_DIR)
    val_dir: pathlib.Path = dataclasses.field(default_factory=lambda: const.REF_DIR)
    test_dir: pathlib.Path = dataclasses.field(default_factory=lambda: const.REF_DIR)
    p_sun: float = 0.5
    p_glare: float = 0.5
    p_flare: float = 0.5
    p_hq_flare: float = 0.5
    target_binarization: bool = False

    def to_json_dict(self) -> dict[str, object]:
        serialized = dataclasses.asdict(self)
        serialized_clean = {}
        for key, value in serialized.items():
            if isinstance(value, pathlib.Path):
                serialized_clean[key] = str(value)
            else:
                serialized_clean[key] = value
        return serialized_clean

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
        parser.add_argument(
            "--max-epochs", type=int, default=10, help="Number of epochs"
        )
        parser.add_argument(
            "--frac-used", type=float, default=1, help="Fraction of data used"
        )
        parser.add_argument(
            "--detect-model",
            type=str,
            help="Model for the detector",
        )
        parser.add_argument(
            "--combiner-model",
            type=str,
            help="Model for the combiner",
        )
        parser.add_argument(
            "--inpainter-model",
            type=str,
            help="Model for the inpainter",
        )
        parser.add_argument(
            "--detector-kwargs",
            type=lambda x: eval(x),
            default="{}",
            help="Additional kwargs for the detector model",
        )
        parser.add_argument(
            "--combiner-kwargs",
            type=lambda x: eval(x),
            default="{}",
            help="Additional kwargs for the combiner model",
        )
        parser.add_argument(
            "--inpainter-kwargs",
            type=lambda x: eval(x),
            default="{}",
            help="Additional kwargs for the inpainter model",
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
        parser.add_argument(
            "--img-glob",
            type=str,
            default="**/*.npy",
            help="Glob pattern for train image files",
        )
        parser.add_argument(
            "--ref-dir",
            type=pathlib.Path,
            help="Path to reference directory",
        )
        parser.add_argument(
            "--train-dir",
            type=pathlib.Path,
            help="Path to train directory",
        )
        parser.add_argument(
            "--val-dir",
            type=pathlib.Path,
            help="Path to val directory",
        )
        parser.add_argument(
            "--test-dir",
            type=pathlib.Path,
            help="Path to test directory",
        )
        parser.add_argument(
            "--p-sun",
            type=float,
            default=0.5,
            help="Probability of a sun augmentation in artifact detector",
        )
        parser.add_argument(
            "--p-glare",
            type=float,
            default=0.5,
            help="Probability of a glare augmentation in artifact detector",
        )
        parser.add_argument(
            "--p-flare",
            type=float,
            default=0.5,
            help="Probability of a flare augmentation in artifact detector",
        )
        parser.add_argument(
            "--p-hq-flare",
            type=float,
            default=0.5,
            help="Probability of a high quality flare augmentation in artifact detector",
        )
        parser.add_argument(
            "--target-binarization",
            action="store_true",
            help="Use target binarization in artifact detector",
        )
        return cls(**vars(parser.parse_args()))

    def __post_init__(self):
        assert 0 <= self.p_sun <= 1, "Probability of sun augmentation must be in [0, 1]"
        assert 0 <= self.p_glare <= 1, (
            "Probability of glare augmentation must be in [0, 1]"
        )
        assert 0 <= self.p_flare <= 1, (
            "Probability of flare augmentation must be in [0, 1]"
        )
        assert 0 <= self.p_hq_flare <= 1, (
            "Probability of high quality flare augmentation must be in [0, 1]"
        )

    def get_logger(self):
        if self.log_tensorboard:
            return loggers.TensorBoardLogger(
                "lightning_logs",
                name="unet",
            )
        return loggers.NeptuneLogger(
            log_model_checkpoints=False, tags=self.run_tags.split(",")
        )

    def get_profiler(self):
        if self.profile:
            return profilers.PyTorchProfiler(
                on_trace_ready=tensorboard_trace_handler("lightning_logs/profiler0"),
                profile_memory=True,
                record_shapes=True,
                record_functions=True,
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            )
        return None

    def get_model(self) -> pl.LightningModule:
        detector = get_model(
            self.detect_model,
            **self.detector_kwargs,
            in_channels=3,
            out_channels=1,
        )
        combiner = get_combiner(
            self.combiner_model,
            **self.combiner_kwargs,
        )
        inpainter = get_model(
            self.inpainter_model,
            **self.inpainter_kwargs,
            in_channels=combiner.get_output_channels(),
            out_channels=const.CHANNELS_OUT,
        )
        return DetectorInpainterModule(
            detector=detector,
            combiner=combiner,
            inpainter=inpainter,
        )

    def get_data_module(self) -> datamodule.BaseDataModule:
        train_paths = sorted(self.train_dir.glob(const.DATA_PATTERN))  # type: ignore
        train_paths = [p for p in train_paths if p.match(self.img_glob)]
        val_paths = sorted(self.val_dir.glob(const.DATA_PATTERN))  # type: ignore
        test_paths = sorted(self.test_dir.glob(const.DATA_PATTERN))  # type: ignore
        ref_paths = sorted(self.ref_dir.glob(const.DATA_PATTERN))  # type: ignore

        np.random.shuffle(train_paths)  # type: ignore
        np.random.shuffle(val_paths)  # type: ignore
        np.random.shuffle(test_paths)  # type: ignore
        train_paths = train_paths[: int(len(train_paths) * self.frac_used)]
        val_paths = val_paths[: int(len(val_paths) * self.frac_used)]
        test_paths = test_paths[: int(len(test_paths) * self.frac_used)]

        assert train_paths, f"Train paths not found in {self.train_dir}"
        assert val_paths, f"Val paths not found in {self.val_dir}"
        assert test_paths, f"Test paths not found in {self.test_dir}"

        return datamodule.JointDataModule(
            train_paths=train_paths,
            val_paths=val_paths,
            test_paths=test_paths,
            ref_paths=ref_paths,
            batch_size=self.batch_size,
            frac_used=self.frac_used,
            num_workers=self.num_workers,
            p_sun=self.p_sun,
            p_glare=self.p_glare,
            p_flare=self.p_flare,
            p_hq_flare=self.p_hq_flare,
            target_binarization=self.target_binarization,
        )
