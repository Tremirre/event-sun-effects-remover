import argparse
import dataclasses
import logging
import pathlib

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
import tqdm

from src import const
from src.data import dataset, transforms
from src.model import noop, unet

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
    weights: pathlib.Path
    data_dir: pathlib.Path
    diff_intensity: int
    output: pathlib.Path

    def __post_init__(self):
        assert self.weights.exists(), f"Weights file {self.weights} does not exist"
        assert self.data_dir.exists(), f"Data directory {self.data_dir} does not exist"
        assert (
            0 <= self.diff_intensity <= 255
        ), "Diff intensity threshold must be in [0, 255]"
        self.output.parent.mkdir(parents=True, exist_ok=True)

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
            "--weights", type=pathlib.Path, help="Path to model weights"
        )
        parser.add_argument(
            "--data-dir",
            type=pathlib.Path,
            help="Path to data directory",
            required=True,
        )
        parser.add_argument(
            "--diff-intensity",
            type=int,
            help="Threshold for diff intensity masker",
            default=110,
        )
        parser.add_argument(
            "--output", type=pathlib.Path, help="Output file path", required=True
        )
        return cls(**vars(parser.parse_args()))


def model_from_config(config: Config) -> pl.LightningModule:
    if config.unet_blocks > 0:
        model = unet.UNet(
            config.unet_blocks, config.unet_depth, config.unet_kernel, config.unet_fft
        )
        model.load_state_dict(torch.load(config.weights, weights_only=True))
        logger.info("Using U-Net model")
        return model
    logger.info("Using NoOp model")
    return noop.NoOp()


def masker_from_config(config: Config) -> dataset.Masker:
    return dataset.DiffIntensityMasker(config.diff_intensity)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    config = Config.from_args()
    model = model_from_config(config).to(DEVICE)
    img_paths = list(config.data_dir.glob("**/*.npy"))
    logger.info(f"Found {len(img_paths)} images in {config.data_dir}")

    infer_dataset = dataset.BGREMDataset(
        img_paths,
        masker=transforms.DiffIntensityMasker(config.diff_intensity),
        bgr_transform=T.Compose([T.ToTensor()]),
    )
    infer_loader = torch.utils.data.DataLoader(
        infer_dataset,
        batch_size=config.batch_size,
        num_workers=4,
        shuffle=False,
        persistent_workers=True,
    )
    num_batches = len(infer_loader)
    logger.info(f"Starting inference on {num_batches} batches")
    iter_batches = tqdm.tqdm(infer_loader, total=num_batches)
    all_bgrem = []
    for batch in iter_batches:
        bgrem, target = batch
        bgr_orig = bgrem[:, :3].detach().cpu().numpy()
        bgr_orig = np.transpose(bgr_orig, (0, 2, 3, 1))
        mask = bgrem[:, 4].unsqueeze(-1).detach().cpu().numpy()
        bgrem = bgrem.to(DEVICE)
        bgr = model(bgrem)
        bgr = bgr.detach().cpu().numpy()

        bgr = np.transpose(bgr, (0, 2, 3, 1))
        bgr = np.where(mask, bgr, bgr_orig)
        bgr = np.clip(bgr * 255, 0, 255).astype(np.uint8)
        all_bgrem.append(bgr)

    all_bgrem = np.concatenate(all_bgrem, axis=0)

    logger.info(f"Saving output to {config.output}")
    out = cv2.VideoWriter(
        str(config.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (const.IMG_WIDTH, const.IMG_HEIGHT),
    )
    for frame in all_bgrem:
        out.write(frame)
    out.release()
