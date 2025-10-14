from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib

import cv2
import numpy as np  # type: ignore
import pytorch_msssim as msssim  # type: ignore
import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
import torchvision.transforms as T
import tqdm  # type: ignore

from src import const, utils

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s %(name)s %(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


TEST_ART_SPLIT = json.loads((const.SPLIT_DIR / "test_artifact_source.json").read_text())
FILE_TO_TYPE = {}
for art_type, files in TEST_ART_SPLIT.items():
    for f in files:
        FILE_TO_TYPE[f] = art_type
FLARES_TEST = const.DATA_DIR / "detect" / "test"
THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.5]  # Thresholds for artifact detection


@dataclasses.dataclass
class TestArgs:
    test_path: pathlib.Path
    ref_path: pathlib.Path
    out_file: pathlib.Path

    def __post_init__(self):
        assert self.test_path.exists(), "Test path does not exist"
        assert self.ref_path.exists(), "Reference path does not exist"
        self.out_file.parent.mkdir(parents=True, exist_ok=True)
        assert not self.out_file.exists(), "Output file already exists"

    @classmethod
    def from_args(cls) -> TestArgs:
        parser = argparse.ArgumentParser(
            description="Test script for evaluating the image outputs against a reference set"
        )
        _ = parser.add_argument(
            "test_path", type=pathlib.Path, help="Path to the test set"
        )
        _ = parser.add_argument(
            "ref_path", type=pathlib.Path, help="Path to the reference set"
        )
        _ = parser.add_argument(
            "out_file", type=pathlib.Path, help="Path to the output JSON file"
        )
        return cls(**vars(parser.parse_args()))


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
    logger.info("Starting evaluation...")
    args = TestArgs.from_args()
    ref_paths = sorted(args.ref_path.glob("**/*.npy"))
    test_paths = sorted(args.test_path.glob("**/*.png"))
    logger.info(f"Found {len(ref_paths)} reference images")
    logger.info(f"Found {len(test_paths)} test images")
    assert len(test_paths) == len(ref_paths), "Test and reference paths do not match"
    test_imgs: dict[str, np.ndarray] = {p.stem: cv2.imread(str(p)) for p in test_paths}
    ref_imgs: dict[str, np.ndarray] = {p.stem: np.load(p) for p in ref_paths}
    entries = []
    for img_name, ref_img in tqdm.tqdm(ref_imgs.items()):
        ref_img = ref_img[..., :3]
        test_img = test_imgs[img_name]
        t_h, t_w = test_img.shape[:2]
        r_h, r_w = ref_img.shape[:2]

        if r_h >= t_h:
            y0 = (r_h - t_h) // 2
            y1 = y0 + t_h
        else:
            y0, y1 = 0, r_h

        if r_w >= t_w:
            x0 = (r_w - t_w) // 2
            x1 = x0 + t_w
        else:
            x0, x1 = 0, r_w

        cropped = ref_img[y0:y1, x0:x1]

        # Pad if smaller than target
        pad_h = max(0, t_h - cropped.shape[0])
        pad_w = max(0, t_w - cropped.shape[1])
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        ref_resized = np.pad(
            cropped,
            ((top, bottom), (left, right), (0, 0))
            if ref_img.ndim == 3
            else ((top, bottom), (left, right)),
            mode="constant",
            constant_values=0,
        )

        ref_img_conv = T.ToTensor()(ref_resized)
        test_img_conv = T.ToTensor()(test_img[..., :3])

        assert ref_img_conv.shape == test_img_conv.shape, (
            f"Shape mismatch for {img_name}: {ref_img_conv.shape} vs {test_img_conv.shape}"
        )

        # unsqueeze to add batch dimension
        ref_img_conv = ref_img_conv.unsqueeze(0)
        test_img_conv = test_img_conv.unsqueeze(0)
        for metric_name, metric_fn in COMMON_METRICS["removal"].items():
            m_val = float(metric_fn(test_img_conv, ref_img_conv))
            entries.append(
                {
                    "image": img_name,
                    "type": FILE_TO_TYPE.get(f"{img_name}.npy", "unknown"),
                    "metric": metric_name,
                    "value": float(m_val),
                    "dataset": args.out_file.stem,
                }
            )

    logger.info("Exporting results...")
    results_path = args.out_file
    results_path.write_text(json.dumps(entries, indent=4))
    logger.info(f"Results exported to {results_path}")
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
