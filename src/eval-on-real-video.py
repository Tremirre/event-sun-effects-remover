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

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s %(name)s %(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EvalArgs:
    reference_videos: pathlib.Path
    test_res_dir: pathlib.Path
    output_path: pathlib.Path

    def __post_init__(self):
        assert self.reference_videos.exists(), "Reference video does not exist"
        assert self.test_res_dir.exists(), "Test directory does not exist"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        assert not self.output_path.exists(), "Output file already exists"
        if self.output_path.suffix != ".json":
            self.output_path = self.output_path.with_suffix(".json")

    @classmethod
    def from_args(cls) -> EvalArgs:
        parser = argparse.ArgumentParser(description="Video Quality Evaluation Script")
        parser.add_argument(
            "-r",
            "--reference-videos",
            type=pathlib.Path,
            required=True,
            help="Path to the reference videos directory",
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


def main():
    args = EvalArgs.from_args()
    logger.info(f"Reference videos path: {args.reference_videos}")
    logger.info(f"Test frames path: {args.test_res_dir}")

    videos = sorted(args.reference_videos.glob("*.mp4"))
    logger.info(f"Found {len(videos)} reference videos")
    all_scores = []
    for video_path in videos:
        logger.info(f"Evaluating video: {video_path.name}")
        test_frames = sorted(args.test_res_dir.glob(f"**/{video_path.stem}*.png"))
        logger.info(f"Found {len(test_frames)} test frames")

        ref_video = read_video(video_path)
        test_video = np.stack([cv2.imread(str(p)) for p in test_frames])

        logger.info(
            f"Reference video shape: {ref_video.shape}, Test video shape: {test_video.shape}"
        )
        assert ref_video.shape[0] == test_video.shape[0], (
            "Reference and test videos must have the same length",
            ref_video.shape,
            test_video.shape,
        )
        logger.info("Evaluating VMAF")

        logger.info("Calculating frame-wise scores")
        iter_frames = zip(ref_video, test_video, strict=True)
        pbar: tqdm.tqdm[tuple[np.ndarray, np.ndarray]] = tqdm.tqdm(
            iter_frames, total=len(ref_video), desc="Calculating scores"
        )
        vid_scores: list[dict[str, float]] = []
        for ref_img, test_img in pbar:
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
            scores = eval_frames(ref_resized, test_img)
            scores["video"] = video_path.name
            vid_scores.append(scores)

        avg_scores = {
            k: float(np.mean([s[k] for s in vid_scores]))
            for k in vid_scores[0].keys()
            if k != "video"
        }
        logger.info(f"Average scores: {avg_scores}")
        all_scores.extend(vid_scores)
    with open(args.output_path, "w") as f:
        json.dump(all_scores, f, indent=4)


if __name__ == "__main__":
    main()
