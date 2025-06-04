from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib

import brisque
import cv2
import ffmpeg_quality_metrics as fqm
import numpy as np
import pytorch_msssim as msssim
import torch
import torch.nn.functional as F
import tqdm

from src.loss import VGGLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s %(name)s %(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EvalArgs:
    reference_video: pathlib.Path
    test_video: pathlib.Path
    output_path: pathlib.Path

    def __post_init__(self):
        assert self.reference_video.exists(), "Reference video does not exist"
        assert self.test_video.exists(), "Test video does not exist"
        assert self.output_path.parent.exists(), "Output directory does not exist"
        if self.output_path.suffix != ".json":
            self.output_path = self.output_path.with_suffix(".json")

    @classmethod
    def from_args(cls) -> EvalArgs:
        parser = argparse.ArgumentParser(description="Video Quality Evaluation Script")
        parser.add_argument(
            "-r",
            "--reference-video",
            type=pathlib.Path,
            required=True,
            help="Path to the reference video file",
        )
        parser.add_argument(
            "-t",
            "--test-video",
            type=pathlib.Path,
            required=True,
            help="Path to the test video file",
        )
        parser.add_argument(
            "-o",
            "--output-path",
            type=pathlib.Path,
            required=True,
            help="Path to save the evaluation results",
        )
        args = parser.parse_args()
        return cls(**vars(args))


def read_video(
    path: pathlib.Path,
    verbose: bool = True,
    target_width: int = 640,
    target_height: int = 480,
) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    iter_frames = range(num_frames)
    if verbose:
        iter_frames = tqdm.tqdm(iter_frames, total=num_frames, desc=f"Reading {path}")
    frames = []
    for _ in iter_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.shape[1] != target_width or frame.shape[0] != target_height:
            frame = cv2.resize(frame, (target_width, target_height))
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    return frames


BRISQUE = brisque.BRISQUE()
VGG = VGGLoss().to(DEVICE)


def eval_frames(ref_frame: np.ndarray, test_frame: np.ndarray) -> dict[str, float]:
    ref_brisque = BRISQUE.score(ref_frame)
    test_brisque = BRISQUE.score(test_frame)
    diff_brisque = ref_brisque - test_brisque
    ref_torch = (
        torch.from_numpy(ref_frame).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        / 255.0
    )
    test_torch = (
        torch.from_numpy(test_frame).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        / 255.0
    )
    vgg_loss = VGG(ref_torch, test_torch).item()
    ssim = msssim.ms_ssim(ref_torch, test_torch).item()
    mae = F.l1_loss(ref_torch, test_torch).item()
    res = {
        "ref_brisque": ref_brisque,
        "test_brisque": test_brisque,
        "diff_brisque": diff_brisque,
        "vgg_loss": vgg_loss,
        "ssim": ssim,
        "mae": mae,
    }
    return {k: float(v) for k, v in res.items()}


def main():
    args = EvalArgs.from_args()
    ref_video = read_video(args.reference_video)
    test_video = read_video(args.test_video)
    logger.info(
        f"Reference video shape: {ref_video.shape}, Test video shape: {test_video.shape}"
    )
    assert ref_video.shape == test_video.shape, (
        "Reference and test videos must have the same shape",
        ref_video.shape,
        test_video.shape,
    )
    logger.info("Evaluating VMAF")

    fqm_evaluator = fqm.FfmpegQualityMetrics(
        str(args.reference_video),
        str(args.test_video),
    )
    fpm_scores = fqm_evaluator.calculate(["vmaf"])

    logger.info("Calculating frame-wise scores")
    iter_frames = zip(ref_video, test_video, fpm_scores["vmaf"], strict=True)
    pbar = tqdm.tqdm(iter_frames, total=len(ref_video), desc="Calculating scores")
    all_scores: list[dict[str, float]] = []
    for ref_frame, test_frame, vmaf in pbar:
        scores = eval_frames(ref_frame, test_frame)
        scores["vmaf"] = vmaf["vmaf"]
        all_scores.append(scores)

    avg_scores = {
        k: float(np.mean([s[k] for s in all_scores])) for k in all_scores[0].keys()
    }
    logger.info(f"Average scores: {avg_scores}")
    with open(args.output_path, "w") as f:
        json.dump(all_scores, f, indent=4)


if __name__ == "__main__":
    main()
