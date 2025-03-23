import argparse
import dataclasses
import pathlib

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers, profilers
from torch.profiler import tensorboard_trace_handler

from src import const
from src.data import datamodule
from src.model import modules


@dataclasses.dataclass
class Config:
    batch_size: int
    unet_blocks: int
    unet_depth: int
    unet_kernel: int
    unet_fft: bool
    max_epochs: int
    frac_used: float
    diff_intensity: int
    gan_adv_weight: float = 0.1
    img_glob: str = "**/*.npy"
    module_type: str = "unet"
    event_channel: bool = False
    num_workers: int = 0
    profile: bool = False
    log_tensorboard: bool = False
    yuv_interpolation: bool = False
    mask_blur_factor: int = 0
    sun_aug_prob: float = 0
    grayscale_patch_prob: float = 0
    glare_aug_prob: float = 0
    run_tags: str = "default"
    save: bool = False
    full_pred: bool = False
    weights: pathlib.Path | None = None
    data_dir: pathlib.Path | None = None
    output: pathlib.Path | None = None

    # === artifact detector params ===
    artifact_detector: bool = False
    p_sun: float = 0.5
    p_glare: float = 0.5
    p_flare: float = 0.5
    p_hq_flare: float = 0.5
    target_binarization: bool = False

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
            "--module-type",
            type=str,
            default="unet",
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
        parser.add_argument(
            "--diff-intensity",
            type=int,
            default=100,
            help="Light intensity threshold for reference images",
        )
        parser.add_argument(
            "--gan-adv-weight",
            type=float,
            default=0.1,
            help="Weight for adversarial loss in GAN",
        )
        parser.add_argument(
            "--event-channel",
            action="store_true",
            help="Use separate event channel in dataset (5 channel input), else fill the masked region in bgr with event data.",
        )
        parser.add_argument(
            "--yuv-interpolation",
            action="store_true",
            help="Interpolate Y channel with event data (only if event channel is not separate)",
        )
        parser.add_argument(
            "--mask-blur-factor",
            type=int,
            help="Gaussian blur factor for mask, sigma = mask_blur_factor, kernel_size = 2 * mask_blur_factor + 1",
            default=0,
        )
        parser.add_argument(
            "--sun-aug-prob",
            type=float,
            default=0,
            help="Probability of applying SUN Augmentation",
        )
        parser.add_argument(
            "--grayscale-patch-prob",
            type=float,
            default=0,
            help="Probability of applying Grayscale Patch Augmentation",
        )
        parser.add_argument(
            "--glare-aug-prob",
            type=float,
            default=0,
            help="Probability of applying Glare Augmentation",
        )
        parser.add_argument(
            "--img-glob",
            type=str,
            default="**/*.npy",
            help="Glob pattern for train image files",
        )
        parser.add_argument(
            "--weights",
            type=pathlib.Path,
            help="Path to weights file",
        )
        parser.add_argument(
            "--data-dir",
            type=pathlib.Path,
            help="Path to data directory",
        )
        parser.add_argument(
            "--output",
            type=pathlib.Path,
            help="Output path",
        )
        parser.add_argument(
            "--full-pred",
            action="store_true",
            help="Output full prediction",
        )
        parser.add_argument(
            "--artifact-detector",
            action="store_true",
            help="Use artifact detector",
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
        assert 0 <= self.diff_intensity <= 255, (
            "Diff intensity threshold must be in [0, 255]"
        )
        assert 0 <= self.sun_aug_prob <= 1, (
            "SUN Augmentation probability must be in [0, 1]"
        )

    def prepare_inference(self):
        if self.weights:
            assert self.weights.exists(), f"Weights file {self.weights} does not exist"
        if self.data_dir:
            assert self.data_dir.exists(), (
                f"Data directory {self.data_dir} does not exist"
            )
        if self.output:
            self.output.parent.mkdir(parents=True, exist_ok=True)

    def to_dict(self):
        return dataclasses.asdict(self)

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
        selection_dict = modules.INPAINT_NAMES
        in_channels = 5 if self.event_channel else 4
        if self.artifact_detector:
            selection_dict = modules.DETECTION_NAMES
            in_channels = 3

        model = selection_dict[self.module_type](
            n_blocks=self.unet_blocks,
            block_depth=self.unet_depth,
            kernel_size=self.unet_kernel,
            with_fft=self.unet_fft,
            in_channels=in_channels,
            gan_adv_weight=self.gan_adv_weight,
        )
        if self.weights:
            weights = torch.load(self.weights, weights_only=True)
            if "state_dict" in weights:
                weights = weights["state_dict"]
            model.load_state_dict(weights)
        return model

    def get_data_module(self) -> datamodule.BaseDataModule:
        data_module = datamodule.EventDataModule
        if self.artifact_detector:
            data_module = datamodule.ArtifactDetectionDataModule

        return data_module(
            data_dir=const.TRAIN_VAL_TEST_DIR,
            ref_dir=const.REF_DIR,
            batch_size=self.batch_size,
            frac_used=self.frac_used,
            num_workers=self.num_workers,
            ref_threshold=self.diff_intensity,
            sep_event_channel=self.event_channel,
            train_img_glob=self.img_glob,
            mask_blur_factor=self.mask_blur_factor,
            yuv_interpolation=self.yuv_interpolation,
            sun_aug_prob=self.sun_aug_prob,
            gs_patch_prob=self.grayscale_patch_prob,
            glare_aug_prob=self.glare_aug_prob,
            p_sun=self.p_sun,
            p_glare=self.p_glare,
            p_flare=self.p_flare,
            p_hq_flare=self.p_hq_flare,
            target_binarization=self.target_binarization,
        )
