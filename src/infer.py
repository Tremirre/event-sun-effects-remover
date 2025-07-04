from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import tqdm

from src import const
from src.config import Config
from src.data import dataset, transforms
from src.model import combiners
from src.model.modules import DetectorInpainterModule
from src.utils import tensor_to_numpy_img

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s %(name)s %(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclasses.dataclass
class InferArgs:
    config: Config
    weights_path: pathlib.Path
    input_dir: pathlib.Path
    output_dir: pathlib.Path
    batch_size: int = 8

    def __post_init__(self):
        if isinstance(self.config, pathlib.Path):
            self.config = Config.from_json(self.config)
        assert isinstance(self.config, Config), "Config must be a Config object"
        assert self.weights_path.exists(), "Weights path does not exist"
        assert self.input_dir.exists(), "Input directory does not exist"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_args(cls) -> InferArgs:
        parser = argparse.ArgumentParser(
            description="Inference script for background removal"
        )
        parser.add_argument(
            "--config",
            type=pathlib.Path,
            required=True,
            help="Path to the config file",
        )
        parser.add_argument(
            "--weights-path",
            type=pathlib.Path,
            required=True,
            help="Path to the model weights file",
        )
        parser.add_argument(
            "--input-dir",
            type=pathlib.Path,
            required=True,
            help="Path to the input directory containing input frames",
        )
        parser.add_argument(
            "--output-dir",
            type=pathlib.Path,
            required=True,
            help="Path to the output directory to save results",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=8,
            help="Batch size for inference",
        )

        return cls(**vars(parser.parse_args()))

    def get_model(self) -> DetectorInpainterModule:
        model = self.config.get_model()
        model.load_state_dict(
            torch.load(self.weights_path, map_location=DEVICE)["state_dict"]
        )
        return model


if __name__ == "__main__":
    infer_args = InferArgs.from_args()
    model = infer_args.get_model().to(DEVICE)

    combiner_detection = isinstance(model.combiner, combiners.EventConsideringCombiner)
    img_paths = sorted(infer_args.input_dir.glob("**/*.npy"))
    logger.info(f"Found {len(img_paths)} images in {infer_args.input_dir}")
    infer_dataset = dataset.BGREADataset(
        img_paths,
        transform=T.Compose(
            [
                transforms.EventMaskChannelRemover(),
                T.ToTensor(),
            ]
        ),
    )
    infer_loader = torch.utils.data.DataLoader(
        infer_dataset,
        batch_size=infer_args.batch_size,
        num_workers=4,
        shuffle=False,
        persistent_workers=True,
    )
    num_batches = len(infer_loader)
    logger.info(f"Starting inference on {num_batches} batches")
    iter_batches = tqdm.tqdm(infer_loader, total=num_batches)
    all_estimate_maps: list[np.ndarray] = []
    all_rec_frames: list[np.ndarray] = []
    all_post_estimate_maps: list[np.ndarray] = []
    for x, y in iter_batches:
        est_map, rec_frames = model(x.to(DEVICE))
        if combiner_detection:
            est_map = model.combiner(x.to(DEVICE), est_map)[:, 4:5]
        post_est_map = F.sigmoid(model.detector(rec_frames.detach()))
        all_estimate_maps.extend(tensor_to_numpy_img(est_map.cpu()))
        all_rec_frames.extend(tensor_to_numpy_img(rec_frames.cpu()))
        all_post_estimate_maps.extend(tensor_to_numpy_img(post_est_map.cpu()))

    map_out = infer_args.output_dir / const.TEST_EST_MAP_OUT
    post_map_out = infer_args.output_dir / const.TEST_POST_EST_MAP_OUT
    rec_out = infer_args.output_dir / const.TEST_REC_FRAMES_OUT
    logger.info(f"Saving estimate map to {map_out}")
    out = cv2.VideoWriter(
        str(map_out),
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
        30,
        (all_estimate_maps[0].shape[1], all_estimate_maps[0].shape[0]),
        isColor=False,
    )
    for img in all_estimate_maps:
        out.write(img)
    out.release()
    logger.info(f"Saving post estimate map to {post_map_out}")
    out = cv2.VideoWriter(
        str(post_map_out),
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
        30,
        (all_post_estimate_maps[0].shape[1], all_post_estimate_maps[0].shape[0]),
        isColor=False,
    )
    for img in all_post_estimate_maps:
        out.write(img)
    out.release()
    logger.info(f"Saving reconstructed frames to {rec_out}")
    out = cv2.VideoWriter(
        str(rec_out),
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
        30,
        (all_rec_frames[0].shape[1], all_rec_frames[0].shape[0]),
    )
    for img in all_rec_frames:
        out.write(img)
    out.release()
