from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib
import typing

import cv2  # type: ignore
import numpy as np  # type: ignore
import pytorch_msssim as msssim  # type: ignore
import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
import torchvision.transforms as T  # type: ignore
import tqdm  # type: ignore

from src import const, utils
from src.config import Config
from src.data import artifacts, dataset, transforms
from src.model import combiners, modules

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s %(name)s %(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_ART_SPLIT = json.loads((const.SPLIT_DIR / "test_artifact_source.json").read_text())
FLARES_TEST = const.DATA_DIR / "detect" / "test"
THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.5]  # Thresholds for artifact detection


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
        self.output_dir.mkdir(parents=True, exist_ok=True)

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


def f1_score(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate F1 score for binary masks."""
    tp = (preds & targets).float().sum()
    fp = (preds & ~targets).float().sum()
    fn = (~preds & targets).float().sum()
    if tp + fp + fn == 0:
        return 0.0
    return (2 * tp) / (2 * tp + fp + fn)


COMMON_METRICS = {
    "removal": {
        "mae": lambda x, y: F.l1_loss(x, y).item(),
        "mse": lambda x, y: F.mse_loss(x, y).item(),
        "mape": lambda x, y: F.l1_loss(x, y) / (y.abs().mean() + 1e-8),
        "psnr": lambda x, y: 10 * torch.log10(1 / F.mse_loss(x, y)).item(),
        "mssim": lambda x, y: msssim.ms_ssim(x, y, data_range=1.0).item(),
    },
    "detection": {
        "accuracy": lambda preds, targets: (preds == targets).float().mean().item(),
        "iou": lambda preds, targets: (
            (preds & targets).float().sum() / (preds | targets).float().sum()
        ).item(),
        "f1_score": f1_score,
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
    combiner_detection = isinstance(model.combiner, combiners.EventConsideringCombiner)
    if combiner_detection:
        logger.info("Using EventConsideringCombiner for detection")
    model.to(DEVICE)

    all_metrics: list = []  # type: ignore

    logger.info("Testing on artificial datasets...")
    for kind, test_dataset in test_datasets.items():
        logger.info(f"Testing on {kind} test_dataset with {len(test_dataset)} samples")
        dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )
        num_batches = len(dataloader)
        pathlib.Path(args.output_dir / "pred" / kind / "rec").mkdir(
            parents=True, exist_ok=True
        )
        pathlib.Path(args.output_dir / "pred" / kind / "est").mkdir(
            parents=True, exist_ok=True
        )
        for bidx, (x, y) in tqdm.tqdm(
            enumerate(dataloader), total=num_batches, desc=f"Testing {kind} dataset"
        ):
            expected_frames = y[:, :3]
            expected_mask = y[:, 3]
            with torch.no_grad():
                model_input = x.to(DEVICE)
                est_map, rec_frames = model(model_input)
                if combiner_detection:
                    est_map = model.combiner(model_input, est_map)[:, 4:5].cpu()
                else:
                    est_map = est_map.cpu()
                rec_frames = rec_frames.cpu()
            rec_frames_np = utils.tensor_to_numpy_img(rec_frames)
            est_map_np = utils.tensor_to_numpy_img(est_map)

            for i in range(len(expected_frames)):
                dataset_idx = bidx * args.batch_size + i
                sample_name = test_dataset.img_paths[dataset_idx].stem
                cv2.imwrite(
                    args.output_dir / "pred" / kind / "rec" / f"{sample_name}.png",
                    rec_frames_np[i],
                )
                cv2.imwrite(
                    args.output_dir / "pred" / kind / "est" / f"{sample_name}.png",
                    est_map_np[i],
                )
                for metric_name, metric_fn in COMMON_METRICS["removal"].items():
                    m_val = float(
                        metric_fn(rec_frames[i : i + 1], expected_frames[i : i + 1])
                    )
                    all_metrics.append(
                        {
                            "kind": kind,
                            "type": "removal",
                            "metric": metric_name,
                            "value": m_val,
                        }
                    )
                for metric_name, metric_fn in COMMON_METRICS["detection"].items():
                    for threshold in THRESHOLDS:
                        preds = est_map[i : i + 1] > threshold
                        targets = expected_mask[i : i + 1] > threshold
                        m_val = float(metric_fn(preds, targets))
                        full_metric_name = f"{metric_name}_th{int(threshold * 100):02d}"
                        all_metrics.append(
                            {
                                "kind": kind,
                                "type": "detection",
                                "metric": full_metric_name,
                                "value": m_val,
                            }
                        )

    real_flare_paths = sorted(FLARES_TEST.glob("real/*.npy"))

    logger.info(f"Found {len(real_flare_paths)} real flare images")
    logger.info("Starting detection on real flare images")
    pathlib.Path(args.output_dir / "pred" / "real_flare" / "est").mkdir(
        parents=True, exist_ok=True
    )
    for real_path in tqdm.tqdm(real_flare_paths):
        real_img = np.load(real_path)
        bgr_img = real_img[..., :3]
        mask = real_img[..., 3]
        bgr_img_tensor = T.ToTensor()(bgr_img).unsqueeze(0).to(DEVICE)
        mask_tensor = T.ToTensor()(mask).unsqueeze(0)
        with torch.no_grad():
            est_map = F.sigmoid(model.detector(bgr_img_tensor)).cpu()

        est_map_np = utils.tensor_to_numpy_img(est_map)
        cv2.imwrite(
            args.output_dir / "pred" / "real_flare" / "est" / f"{real_path.stem}.png",
            est_map_np[0],
        )

        for threshold in THRESHOLDS:
            preds = est_map > threshold
            targets = mask_tensor > threshold
            for metric_name, metric_fn in COMMON_METRICS["detection"].items():
                m_val = float(metric_fn(preds, targets.cpu()))
                full_metric_name = f"{metric_name}_th{int(threshold * 100):02d}"
                all_metrics.append(
                    {
                        "kind": "flare7k",
                        "type": "detection",
                        "metric": full_metric_name,
                        "value": m_val,
                    }
                )

    real_event_paths = sorted(FLARES_TEST.glob("masked/*.npy"))
    logger.info(f"Found {len(real_event_paths)} real event images")
    logger.info("Starting removal on real event images")
    pathlib.Path(args.output_dir / "pred" / "real_event" / "est").mkdir(
        parents=True, exist_ok=True
    )

    for real_path in tqdm.tqdm(real_event_paths):
        real_img = np.load(real_path)
        bgr_img = real_img[..., :4]
        mask = real_img[..., 4]
        bgr_img_tensor = T.ToTensor()(bgr_img).unsqueeze(0).to(DEVICE)
        mask_tensor = T.ToTensor()(mask).unsqueeze(0)
        with torch.no_grad():
            est_map = F.sigmoid(model.detector(bgr_img_tensor[:, :3]))
            if combiner_detection:
                est_map = model.combiner(bgr_img_tensor, est_map)[:, 4:5]
            est_map = est_map.cpu()
        est_map_np = utils.tensor_to_numpy_img(est_map)
        cv2.imwrite(
            args.output_dir / "pred" / "real_event" / "est" / f"{real_path.stem}.png",
            est_map_np[0],
        )

        for threshold in THRESHOLDS:
            preds = est_map > threshold
            targets = mask_tensor > threshold
            for metric_name, metric_fn in COMMON_METRICS["detection"].items():
                full_metric_name = f"{metric_name}_th{int(threshold * 100):02d}"
                all_metrics.append(
                    {
                        "kind": "with_events",
                        "type": "detection",
                        "metric": full_metric_name,
                        "value": float(metric_fn(preds.cpu(), targets.cpu())),
                    }
                )

    logger.info("Saving metrics to output directory")
    with open(args.output_dir / "test_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)


if __name__ == "__main__":
    main()
