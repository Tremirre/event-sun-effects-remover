from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib

import brisque
import cv2
import numpy as np
import pytorch_msssim as msssim
import torch
import torch.nn.functional as F
import tqdm

from src import const

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s %(name)s %(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EvalArgs:
    reference_video: pathlib.Path
    test_res_dir: pathlib.Path
    output_path: pathlib.Path

    def __post_init__(self):
        assert self.reference_video.exists(), "Reference video does not exist"
        assert self.test_res_dir.exists(), "Test directory does not exist"
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
            "--test-res-dir",
            type=pathlib.Path,
            required=True,
            help="Path to the test dir with video files",
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
THRESHOLD = 0.5


def eval_frames(ref_frame: np.ndarray, test_frame: np.ndarray) -> dict[str, float]:
    ref_rgb = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
    test_rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
    ref_brisque = BRISQUE.score(ref_rgb)
    test_brisque = BRISQUE.score(test_rgb)
    diff_brisque = ref_brisque - test_brisque
    ref_torch = (
        torch.from_numpy(ref_frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    )
    test_torch = (
        torch.from_numpy(test_frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    )
    ssim = msssim.ms_ssim(ref_torch, test_torch, data_range=1.0).item()
    mae = F.l1_loss(ref_torch, test_torch).item()
    ref_mean_intensity = np.mean(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY))
    test_mean_intensity = np.mean(cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY))
    res = {
        "ref_brisque": ref_brisque,
        "test_brisque": test_brisque,
        "diff_brisque": diff_brisque,
        "ssim": ssim,
        "mae": mae,
        "ref_mean_intensity": ref_mean_intensity,
        "test_mean_intensity": test_mean_intensity,
    }
    return {k: float(v) for k, v in res.items()}


def eval_est_frames(est_frame: np.ndarray, post_frame: np.ndarray) -> dict[str, float]:
    est_frame = est_frame.astype(np.float32) / 255.0
    post_frame = post_frame.astype(np.float32) / 255.0
    est_mean = np.mean(est_frame)
    post_mean = np.mean(post_frame)
    est_over_threshold = np.mean(est_frame > THRESHOLD)
    post_over_threshold = np.mean(post_frame > THRESHOLD)

    return {
        "est_mean": float(est_mean),
        "post_mean": float(post_mean),
        "est_over_threshold": float(est_over_threshold),
        "post_over_threshold": float(post_over_threshold),
    }


def main():
    args = EvalArgs.from_args()
    test_video_path = args.test_res_dir / const.TEST_REC_FRAMES_OUT
    est_map_path = args.test_res_dir / const.TEST_EST_MAP_OUT
    post_map_path = args.test_res_dir / const.TEST_POST_EST_MAP_OUT

    logger.info(f"Reference video path: {args.reference_video}")
    logger.info(f"Test video path: {test_video_path}")
    logger.info(f"Estimated map path: {est_map_path}")
    logger.info(f"Post-processed map path: {post_map_path}")
    ref_video = read_video(args.reference_video)
    test_video = read_video(test_video_path)
    est_map = read_video(est_map_path)
    post_map = read_video(post_map_path)
    logger.info(
        f"Reference video shape: {ref_video.shape}, Test video shape: {test_video.shape}"
    )
    assert ref_video.shape == test_video.shape, (
        "Reference and test videos must have the same shape",
        ref_video.shape,
        test_video.shape,
    )
    logger.info("Evaluating VMAF")

    logger.info("Calculating frame-wise scores")
    iter_frames = zip(ref_video, test_video, strict=True)
    pbar = tqdm.tqdm(iter_frames, total=len(ref_video), desc="Calculating scores")
    all_scores: list[dict[str, float]] = []
    for ref_frame, test_frame in pbar:
        scores = eval_frames(ref_frame, test_frame)
        all_scores.append(scores)

    iter_est_frames = enumerate(zip(est_map, post_map, strict=True))
    pbar = tqdm.tqdm(
        iter_est_frames, total=len(ref_video), desc="Calculating maps scores"
    )
    for i, (est_frame, post_frame) in pbar:
        map_scores = eval_est_frames(est_frame, post_frame)
        all_scores[i].update(map_scores)

    avg_scores = {
        k: float(np.mean([s[k] for s in all_scores])) for k in all_scores[0].keys()
    }
    logger.info(f"Average scores: {avg_scores}")
    with open(args.output_path, "w") as f:
        json.dump(all_scores, f, indent=4)


if __name__ == "__main__":
    main()
