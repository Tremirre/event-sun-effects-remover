from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib
import typing
from collections import defaultdict

import numpy as np  # type: ignore
import pytorch_msssim as msssim  # type: ignore
import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
import torchvision.transforms as T  # type: ignore
import tqdm  # type: ignore

from src import const, utils
from src.config import Config
from src.data import artifacts, dataset, transforms
from src.model import modules

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s %(name)s %(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_ART_SPLIT = json.loads((const.SPLIT_DIR / "test_artifact_source.json").read_text())
FLARES_TEST = const.DATA_DIR / "detect" / "test"
THRESHOLD = 0.05  # Threshold for artifact detection


@dataclasses.dataclass
class TestArgs:
    config: Config
    weights_path: pathlib.Path
    output_dir: pathlib.Path
    batch_size: int = 8

    def __post_init__(self):
        if isinstance(self.config, pathlib.Path):
            self.config = Config.from_json(self.config)
        assert isinstance(self.config, Config), "Config must be a Config object"
        assert self.weights_path.exists(), "Weights path does not exist"

    @classmethod
    def from_args(cls) -> TestArgs:
        parser = argparse.ArgumentParser(
            description="Test script for lighting artifacts detection and removal"
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
            "--output-dir",
            type=pathlib.Path,
            required=True,
            help="Directory to save test results",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=8,
            help="Batch size for testing",
        )
        return cls(**vars(parser.parse_args()))

    def get_model(self) -> modules.DetectorInpainterModule:
        model = self.config.get_model()
        weights = torch.load(self.weights_path, map_location=DEVICE)
        if "state_dict" in weights:
            # If the weights are saved with a state_dict key
            weights = weights["state_dict"]
        model.load_state_dict(weights)
        return typing.cast(modules.DetectorInpainterModule, model)


def make_test_dataset(test_paths: list[str]) -> dataset.BGREADataset:
    full_paths = [next(const.TEST_DIR.glob(f"**/{p}")) for p in test_paths]
    path_exists = [p.exists() for p in full_paths]
    if not all(path_exists):
        missing_paths = [p for p, exists in zip(full_paths, path_exists) if not exists]
        raise FileNotFoundError(f"Some test paths do not exist: {missing_paths}")
    return dataset.BGREADataset(
        full_paths,
        transform=T.Compose(
            [
                transforms.EventMaskChannelRemover(),
                T.ToTensor(),
            ]
        ),
        artifact_source=artifacts.LightArtifactExtractor(),
    )


COMMON_METRICS = {
    "removal": {
        "mae": lambda x, y: F.l1_loss(x, y).item(),
        "mssim": lambda x, y: msssim.ms_ssim(x, y).item(),
    },
    "detection": {
        "accuracy": lambda preds, targets: (preds == targets).float().mean().item(),
        "iou": lambda preds, targets: (
            (preds & targets).float().sum() / (preds | targets).float().sum()
        ).item(),
    },
}


def main():
    utils.set_global_seed(42)
    args = TestArgs.from_args()
    test_datasets = {
        kind: make_test_dataset(paths) for kind, paths in TEST_ART_SPLIT.items()
    }
    logger.info(f"Loaded test datasets: {list(test_datasets.keys())}")
    logger.info(f"Using device: {DEVICE}")

    logger.info(f"Loading model from {args.weights_path}")
    model = args.get_model().eval()
    model.to(DEVICE)
    COMMON_METRICS["removal"]["vgg"] = model.vgg_loss  # type: ignore

    all_metrics: dict = {"artificial": defaultdict(dict), "real": defaultdict(dict)}  # type: ignore

    logger.info("Testing on artificial datasets...")
    for kind, test_dataset in test_datasets.items():
        logger.info(f"Testing on {kind} test_dataset with {len(test_dataset)} samples")
        dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

        removal_metrics = {kind: [] for kind in COMMON_METRICS["removal"].keys()}
        detection_metrics = {kind: [] for kind in COMMON_METRICS["detection"].keys()}
        num_batches = len(dataloader)
        for x, y in tqdm.tqdm(
            dataloader, total=num_batches, desc=f"Testing {kind} dataset"
        ):
            expected_frames = y[:, :3]
            expected_mask = y[:, 3]
            with torch.no_grad():
                est_map, rec_frames = model(x.to(DEVICE))
                est_map = est_map.cpu()
                rec_frames = rec_frames.cpu()

            for metric_name, metric_fn in COMMON_METRICS["removal"].items():
                if metric_name == "vgg":
                    removal_metrics[metric_name].append(
                        float(
                            metric_fn(rec_frames.to(DEVICE), expected_frames.to(DEVICE))
                        )
                    )
                else:
                    removal_metrics[metric_name].append(
                        float(metric_fn(rec_frames.cpu(), expected_frames.cpu()))
                    )
            for metric_name, metric_fn in COMMON_METRICS["detection"].items():
                preds = est_map > THRESHOLD
                targets = expected_mask > THRESHOLD
                detection_metrics[metric_name].append(
                    float(metric_fn(preds.cpu(), targets.cpu()))
                )

        avg_removal_metrics = {k: sum(v) / len(v) for k, v in removal_metrics.items()}
        avg_detection_metrics = {
            k: sum(v) / len(v) for k, v in detection_metrics.items()
        }
        logger.info(f"{kind} removal metrics: {avg_removal_metrics}")
        logger.info(f"{kind} detection metrics: {avg_detection_metrics}")
        all_metrics["artificial"][kind] = {
            "removal": avg_removal_metrics,
            "detection": avg_detection_metrics,
        }

    real_flare_paths = sorted(FLARES_TEST.glob("real/*.npy"))

    logger.info(f"Found {len(real_flare_paths)} real flare images")
    logger.info("Starting detection on real flare images")
    all_metrics["real"]["detection"]["flare7k"] = defaultdict(list)
    all_metrics["real"]["detection"]["event"] = defaultdict(list)
    for real_path in tqdm.tqdm(real_flare_paths):
        real_img = np.load(real_path)
        bgr_img = real_img[..., :3]
        mask = real_img[..., 3]
        bgr_img_tensor = T.ToTensor()(bgr_img).unsqueeze(0).to(DEVICE)
        mask_tensor = T.ToTensor()(mask).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            est_map = model.detector(bgr_img_tensor).cpu()

        preds = est_map > THRESHOLD
        targets = mask_tensor > THRESHOLD
        for metric_name, metric_fn in COMMON_METRICS["detection"].items():
            all_metrics["real"]["detection"]["flare7k"][metric_name].append(
                float(metric_fn(preds.cpu(), targets.cpu()))
            )
    avg_detection_metrics = {
        k: sum(v) / len(v)
        for k, v in all_metrics["real"]["detection"]["flare7k"].items()
    }
    logger.info(f"Real dataset detection metrics: {avg_detection_metrics}")

    real_event_paths = sorted(FLARES_TEST.glob("masked/*.npy"))
    logger.info(f"Found {len(real_event_paths)} real event images")
    logger.info("Starting removal on real event images")

    for real_path in tqdm.tqdm(real_event_paths):
        real_img = np.load(real_path)
        bgr_img = real_img[..., :3]
        mask = real_img[..., 3]
        bgr_img_tensor = T.ToTensor()(bgr_img).unsqueeze(0).to(DEVICE)
        mask_tensor = T.ToTensor()(mask).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            est_map = model.detector(bgr_img_tensor).cpu()
        preds = est_map > THRESHOLD
        targets = mask_tensor > THRESHOLD
        for metric_name, metric_fn in COMMON_METRICS["detection"].items():
            all_metrics["real"]["detection"]["event"][metric_name].append(
                float(metric_fn(preds.cpu(), targets.cpu()))
            )

    avg_detection_metrics = {
        k: sum(v) / len(v) for k, v in all_metrics["real"]["detection"]["event"].items()
    }
    logger.info(f"Real dataset detection metrics: {avg_detection_metrics}")
    with open(args.output_dir / "test_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)


if __name__ == "__main__":
    main()
