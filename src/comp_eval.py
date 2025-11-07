from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib
import subprocess
import tempfile
import warnings
from enum import Enum

import cv2
import decord
import ffmpeg_quality_metrics as fqm
import numpy as np
import pytorch_msssim as msssim
import torch
import torch.nn.functional as F
import torchvision.transforms as T  # type: ignore
import tqdm
import yaml
from fastvqa.datasets import FragmentSampleFrames, SampleFrames, get_spatial_fragments
from fastvqa.models import DiViDeAddEvaluator

from src.utils import set_global_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BRISQUE_CONFIG_PATH = pathlib.Path("data/brisque")
REF_WIDTH, REF_HEIGHT = 640, 480


def brisque(img: np.ndarray) -> float:
    return cv2.quality.QualityBRISQUE_compute(
        img,
        str(BRISQUE_CONFIG_PATH / "brisque_model_live.yml"),
        str(BRISQUE_CONFIG_PATH / "brisque_range_live.yml"),
    )[0]


logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s %(name)s %(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def sigmoid_rescale(score):
    mean, std = (-0.110198185, 0.04178565)
    x = (score - mean) / std
    score = 1 / (1 + np.exp(-x))
    return score


@dataclasses.dataclass
class EvalArgs:
    class Part(str, Enum):
        BASE = "base"
        EXTRA = "extra"
        BOTH = "both"

    vid_dir: pathlib.Path
    comp_results_dir: pathlib.Path
    split_dir: pathlib.Path
    vqa_model_path: pathlib.Path
    vqa_opt_path: pathlib.Path
    part: Part

    def __post_init__(self):
        assert self.vid_dir.exists(), "Video directory does not exist"
        assert self.comp_results_dir.exists(), "Comp results directory does not exist"
        assert self.split_dir.exists(), "Split directory does not exist"
        assert self.vqa_model_path.exists(), "VQA model does not exist"
        assert self.vqa_opt_path.exists(), "VQA opt file does not exist"
        self.part = self.Part(self.part)

    @classmethod
    def from_args(cls) -> EvalArgs:
        parser = argparse.ArgumentParser(
            description="Video Quality Evaluation Script for DeLux"
        )
        _ = parser.add_argument("-v", "--vid-dir", type=pathlib.Path, required=True)
        _ = parser.add_argument(
            "-c",
            "--comp-results-dir",
            type=pathlib.Path,
            required=True,
            help="Path to the comp results directory",
        )
        _ = parser.add_argument(
            "-s",
            "--split-dir",
            type=pathlib.Path,
            required=True,
            help="Path to the split data directory",
        )
        _ = parser.add_argument(
            "-m",
            "--vqa-model-path",
            type=pathlib.Path,
            required=True,
            help="Path to the VQA model",
        )
        _ = parser.add_argument(
            "-o",
            "--vqa-opt-path",
            type=pathlib.Path,
            required=True,
            help="Path to the VQA opt file",
        )
        parser.add_argument(
            "-p",
            "--part",
            type=str,
            required=True,
            help="Part to evaluate [base/extra/both]",
        )
        return cls(**vars(parser.parse_args()))

    def print(self):
        print(f"Vid dir: {self.vid_dir}")
        print(f"Comp results directory: {self.comp_results_dir}")
        print(f"Test data directory: {self.split_dir}")
        print(f"VQA model path: {self.vqa_model_path}")
        print(f"VQA opt file: {self.vqa_opt_path}")
        print(f"Part: {self.part}")


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


THRESHOLD = 0.5

DETECT_THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.5]


def f1_score(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate F1 score for binary masks."""
    tp = (preds & targets).float().sum()
    fp = (preds & ~targets).float().sum()
    fn = (~preds & targets).float().sum()
    if tp + fp + fn == 0:
        return 0.0
    return ((2 * tp) / (2 * tp + fp + fn)).item()


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


def resize_to_shape(ref_img: np.ndarray, t_h: int, t_w: int) -> np.ndarray:
    r_h, r_w = ref_img.shape[:2]

    if r_h == t_h and r_w == t_w:
        return ref_img

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
    return ref_resized


def get_video_dimensions(path: pathlib.Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0",
        str(path),
    ]
    output = subprocess.check_output(cmd, text=True)
    width, height = output.strip().split(",")
    return int(width), int(height)


def normalize_video(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    target_width: int,
    target_height: int,
):
    vf = (
        f"pad='if(gt({target_width},iw),{target_width},iw)':'if(gt({target_height},ih),{target_height},ih)':"
        f"({target_width}-iw)/2:({target_height}-ih)/2:black,"
        f"crop={target_width}:{target_height}"
    )
    cmd = [
        "ffmpeg",
        "-i",
        str(input_path.resolve()),
        "-vf",
        vf,
        "-y",
        str(output_path.resolve()),
    ]
    subprocess.run(cmd, check=True)


def compare_synthetic_reconstruction(
    ref_frames: np.ndarray,
    comp_dir: pathlib.Path,
    competitor: str,
    frame_to_type: dict[str, str],
) -> list[dict[str, float | str]]:
    img_paths = sorted(
        (comp_dir / "preds" / "synth" / "img" / competitor).glob("*.png"),
        key=lambda x: x.stem,
    )
    tested_frames = np.stack([cv2.imread(str(p)) for p in img_paths])

    assert ref_frames.shape[0] == tested_frames.shape[0], (
        f"Number of reference frames ({ref_frames.shape[0]}) does not match number of tested frames ({tested_frames.shape[0]})"
    )
    target_width = min(tested_frames.shape[2], REF_WIDTH)
    target_height = min(tested_frames.shape[1], REF_HEIGHT)
    ref_frames = np.stack(
        [
            resize_to_shape(ref, target_height, target_width)
            for ref, tested in zip(ref_frames[..., :3], tested_frames)
        ]
    )
    tested_frames = np.stack(
        [
            resize_to_shape(tested, target_height, target_width)
            for tested in tested_frames
        ]
    )
    res = []
    for i, (ref, tested) in tqdm.tqdm(
        enumerate(zip(ref_frames, tested_frames)),
        total=len(ref_frames),
        desc="Comparing synthetic reconstruction",
    ):
        ref_torch = T.ToTensor()(ref).unsqueeze(0)
        tested_torch = T.ToTensor()(tested).unsqueeze(0)
        for k, v in COMMON_METRICS["removal"].items():
            value = float(v(ref_torch, tested_torch))
            res.append(
                {
                    "competitor": competitor,
                    "image": img_paths[i].stem,
                    "type": "synthetic",
                    "metric": k,
                    "value": value,
                    "division": frame_to_type[img_paths[i].stem + ".npy"],
                }
            )
    return res


def compare_synthetic_detection(
    ref_frames: np.ndarray,
    comp_dir: pathlib.Path,
    competitor: str,
    frame_to_type: dict[str, str],
) -> list[dict[str, float | str]]:
    competitor_art_path = comp_dir / "preds" / "synth" / "artifact" / competitor
    if not competitor_art_path.exists():
        logger.info(f"No artifacts for competitor {competitor}")
        return []
    img_paths = sorted(competitor_art_path.glob("*.png"), key=lambda x: x.stem)
    tested_frames = np.stack([cv2.imread(str(p)) for p in img_paths])
    assert ref_frames.shape[0] == tested_frames.shape[0], (
        f"Number of reference frames ({ref_frames.shape[0]}) does not match number of tested frames ({tested_frames.shape[0]})"
    )
    target_width = min(tested_frames.shape[2], REF_WIDTH)
    target_height = min(tested_frames.shape[1], REF_HEIGHT)
    ref_frames = np.stack(
        [
            resize_to_shape(ref, target_height, target_width)
            for ref, tested in zip(ref_frames[..., :3], tested_frames)
        ]
    )
    tested_frames = np.stack(
        [
            resize_to_shape(tested, target_height, target_width)
            for tested in tested_frames
        ]
    )
    res = []
    for i, (ref, tested) in tqdm.tqdm(
        enumerate(zip(ref_frames, tested_frames)),
        total=len(ref_frames),
        desc="Comparing synthetic detection",
    ):
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        tested = cv2.cvtColor(tested, cv2.COLOR_BGR2GRAY)
        ref_torch = T.ToTensor()(ref).unsqueeze(0)
        tested_torch = T.ToTensor()(tested).unsqueeze(0)
        for k, v in COMMON_METRICS["detection"].items():
            for threshold in DETECT_THRESHOLDS:
                ref_torch_thresholded = ref_torch > threshold
                tested_torch_thresholded = tested_torch > threshold
                value = v(ref_torch_thresholded, tested_torch_thresholded)
                full_metric_name = f"{k}_th{int(threshold * 100):02d}"
                res.append(
                    {
                        "competitor": competitor,
                        "image": img_paths[i].stem,
                        "type": "synthetic",
                        "metric": full_metric_name,
                        "value": value,
                        "division": frame_to_type[img_paths[i].stem + ".npy"],
                    }
                )
    return res


def eval_frames(ref_frame: np.ndarray, test_frame: np.ndarray) -> dict[str, float]:
    ref_rgb = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
    test_rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
    ref_brisque = brisque(ref_rgb)
    test_brisque = brisque(test_rgb)
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
        "ssim": ssim,
        "mae": mae,
        "ref_mean_intensity": ref_mean_intensity,
        "test_mean_intensity": test_mean_intensity,
    }
    return {k: float(v) for k, v in res.items()}


def compare_real_base_metrics(
    recordings: dict[str, np.ndarray],
    comp_dir: pathlib.Path,
    competitor: str,
) -> list[dict[str, float]]:
    res = []
    for video_name, ref_imgs in recordings.items():
        img_paths = sorted(
            (comp_dir / "preds" / "real" / "img" / competitor).glob(
                f"{video_name}*.png"
            ),
            key=lambda x: x.stem,
        )
        tested_imgs = np.stack([cv2.imread(str(p)) for p in img_paths])
        assert ref_imgs.shape[0] == tested_imgs.shape[0], (
            f"Number of reference frames ({ref_imgs.shape[0]}) does not match number of tested frames ({tested_imgs.shape[0]})"
        )
        target_width = min(tested_imgs.shape[2], REF_WIDTH)
        target_height = min(tested_imgs.shape[1], REF_HEIGHT)
        ref_imgs = np.stack(
            [
                resize_to_shape(ref, target_height, target_width)
                for ref, tested in zip(ref_imgs[..., :3], tested_imgs)
            ]
        )
        tested_imgs = np.stack(
            [
                resize_to_shape(tested, target_height, target_width)
                for tested in tested_imgs
            ]
        )
        for i, (ref, tested) in tqdm.tqdm(
            enumerate(zip(ref_imgs, tested_imgs)),
            total=len(ref_imgs),
            desc=f"Comparing real base metrics for {video_name}",
        ):
            metrics = eval_frames(ref, tested)
            for k, v in metrics.items():
                res.append(
                    {
                        "competitor": competitor,
                        "image": img_paths[i].stem,
                        "type": "real",
                        "metric": k,
                        "value": v,
                        "division": video_name,
                    }
                )
    return res


def compare_real_ffqm_metrics(
    recording_paths: list[pathlib.Path],
    comp_dir: pathlib.Path,
    competitor: str,
) -> list[dict[str, float]]:
    res = []
    for ref_video_path in tqdm.tqdm(recording_paths, desc="Comparing videos with FFQM"):
        video_name = ref_video_path.stem
        img_paths = sorted(
            (comp_dir / "preds" / "real" / "img" / competitor).glob(
                f"{video_name}*.png"
            ),
            key=lambda x: x.stem,
        )
        tested_video_path = (
            comp_dir / "preds" / "vids" / competitor / f"{video_name}.mp4"
        )
        test_width, test_height = get_video_dimensions(tested_video_path)
        target_width = min(test_width, REF_WIDTH)
        target_height = min(test_height, REF_HEIGHT)
        if target_width != REF_WIDTH or target_height != REF_HEIGHT:
            tempdir = tempfile.TemporaryDirectory()
            new_ref_video_path = pathlib.Path(tempdir.name) / ref_video_path.name
            normalize_video(
                ref_video_path, new_ref_video_path, target_width, target_height
            )
            ref_video_path = new_ref_video_path
            new_tested_video_path = pathlib.Path(tempdir.name) / tested_video_path.name
            normalize_video(
                tested_video_path, new_tested_video_path, target_width, target_height
            )
            tested_video_path = new_tested_video_path
        evaluator = fqm.FfmpegQualityMetrics(
            str(ref_video_path), str(tested_video_path)
        )
        fqm_scores = evaluator.calculate(["vmaf", "psnr"])
        assert len(fqm_scores["psnr"]) == len(img_paths)
        for i, (psnr, vmaf) in enumerate(zip(fqm_scores["psnr"], fqm_scores["vmaf"])):
            res.append(
                {
                    "competitor": competitor,
                    "image": img_paths[i].stem,
                    "type": "real",
                    "metric": "vmaf",
                    "value": vmaf["vmaf"],
                    "division": video_name,
                }
            )
            res.append(
                {
                    "competitor": competitor,
                    "image": img_paths[i].stem,
                    "type": "real",
                    "metric": "psnr",
                    "value": psnr["psnr_avg"],
                    "division": video_name,
                }
            )
            res.append(
                {
                    "competitor": competitor,
                    "image": img_paths[i].stem,
                    "type": "real",
                    "metric": "mse",
                    "value": psnr["mse_avg"],
                    "division": video_name,
                }
            )
    return res


def evaluate_video_fastvqa(
    video_path: pathlib.Path,
    opt: dict,
    evaluator: DiViDeAddEvaluator,
) -> float:
    video_reader = decord.VideoReader(str(video_path))
    vsamples = {}
    t_data_opt = opt["data"]["val-kv1k"]["args"]
    s_data_opt = opt["data"]["val-kv1k"]["args"]["sample_types"]
    for sample_type, sample_args in s_data_opt.items():
        ## Sample Temporally
        if t_data_opt.get("t_frag", 1) > 1:
            sampler = FragmentSampleFrames(
                fsize_t=sample_args["clip_len"] // sample_args.get("t_frag", 1),
                fragments_t=sample_args.get("t_frag", 1),
                num_clips=sample_args.get("num_clips", 1),
            )
        else:
            sampler = SampleFrames(
                clip_len=sample_args["clip_len"], num_clips=sample_args["num_clips"]
            )

        num_clips = sample_args.get("num_clips", 1)
        frames = sampler(len(video_reader))
        frame_dict = {idx: video_reader[idx] for idx in np.unique(frames)}
        imgs = [frame_dict[idx] for idx in frames]
        video = torch.stack(imgs, 0)
        video = video.permute(3, 0, 1, 2)

        ## Sample Spatially
        sampled_video = get_spatial_fragments(video, **sample_args)
        mean, std = (
            torch.FloatTensor([123.675, 116.28, 103.53]),
            torch.FloatTensor([58.395, 57.12, 57.375]),
        )
        sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(
            3, 0, 1, 2
        )

        sampled_video = sampled_video.reshape(
            sampled_video.shape[0], num_clips, -1, *sampled_video.shape[2:]
        ).transpose(0, 1)
        vsamples[sample_type] = sampled_video.to(DEVICE)
    result = evaluator(vsamples)
    score = sigmoid_rescale(result.mean().item())
    return score


def compare_fastvqa_metrics(
    recording_scores: dict[str, float],
    comp_dir: pathlib.Path,
    vid_dir: pathlib.Path,
    competitor: str,
    evaluator: DiViDeAddEvaluator,
    opts: dict,
):
    res = []
    for video_name, base_score in tqdm.tqdm(
        recording_scores.items(), desc="Comparing videos with FAST-VQA-M"
    ):
        tested_video_path = (
            comp_dir / "preds" / "vids" / competitor / f"{video_name}.mp4"
        )
        tested_width, tested_height = get_video_dimensions(tested_video_path)
        target_width = min(tested_width, REF_WIDTH)
        target_height = min(tested_height, REF_HEIGHT)

        ref_video_path = vid_dir / f"{video_name}.mp4"
        if target_width != REF_WIDTH or target_height != REF_HEIGHT:
            tempdir = tempfile.TemporaryDirectory()
            new_ref_video_path = pathlib.Path(tempdir.name) / ref_video_path.name
            normalize_video(
                ref_video_path, new_ref_video_path, target_width, target_height
            )
            ref_video_path = new_ref_video_path
            new_tested_video_path = pathlib.Path(tempdir.name) / tested_video_path.name
            normalize_video(
                tested_video_path, new_tested_video_path, target_width, target_height
            )
            tested_video_path = new_tested_video_path
            base_score = evaluate_video_fastvqa(ref_video_path, opts, evaluator)
        else:
            base_score = recording_scores[video_name]
        score = evaluate_video_fastvqa(tested_video_path, opts, evaluator)
        res.append(
            {
                "competitor": competitor,
                "image": tested_video_path.stem,
                "type": "real",
                "metric": "ref_fastvqa",
                "value": score,
                "division": video_name,
            }
        )
        res.append(
            {
                "competitor": competitor,
                "image": tested_video_path.stem,
                "type": "real",
                "metric": "test_fastvqa",
                "value": base_score,
                "division": video_name,
            }
        )
    return res


def main():
    set_global_seed(42)
    warnings.filterwarnings("ignore", category=UserWarning)
    args = EvalArgs.from_args()
    args.print()

    recording_paths = sorted((args.vid_dir).glob("*.mp4"), key=lambda x: x.stem)
    logger.info(f"Found {len(recording_paths)} recordings")
    recordings = {p.stem: read_video(p) for p in recording_paths}
    with open(args.vqa_opt_path, "r") as f:
        opt = yaml.safe_load(f)

    recording_scores = {}
    if args.part != EvalArgs.Part.EXTRA:
        evaluator = DiViDeAddEvaluator(**opt["model"]["args"]).to(DEVICE)
        evaluator.load_state_dict(
            torch.load(args.vqa_model_path, map_location=DEVICE)["state_dict"]
        )
        for recording_path in tqdm.tqdm(
            recording_paths, desc="Evaluating original recordings with FAST-VQA"
        ):
            video_name = recording_path.stem
            evaluated_score = evaluate_video_fastvqa(recording_path, opt, evaluator)
            recording_scores[video_name] = evaluated_score

    competitors = sorted(
        [p.stem for p in (args.comp_results_dir / "preds" / "vids").glob("*")]
    )
    logger.info(f"Found {len(competitors)} competitors")

    test_frames_paths = sorted(
        (args.split_dir / "test").glob("**/*.npy"), key=lambda x: x.stem
    )
    logger.info(f"Found {len(test_frames_paths)} test frames")

    test_frames = np.stack([np.load(p) for p in test_frames_paths])

    test_frames_split = json.loads(
        (args.split_dir / "test_artifact_source.json").read_text()
    )
    frame_to_type = {}
    for frame_type, frames in test_frames_split.items():
        for frame in frames:
            frame_to_type[frame] = frame_type

    for comp in competitors:
        scores_file = args.comp_results_dir / "scores" / f"{comp}.json"
        if not scores_file.exists() and args.part != EvalArgs.Part.EXTRA:
            results = []
            logger.info(f"Running evaluation for {comp}")
            sr = compare_synthetic_reconstruction(
                test_frames,
                args.comp_results_dir,
                comp,
                frame_to_type,
            )
            results.extend(sr)
            sd = compare_synthetic_detection(
                test_frames,
                args.comp_results_dir,
                comp,
                frame_to_type,
            )
            results.extend(sd)

            rf = compare_real_ffqm_metrics(recording_paths, args.comp_results_dir, comp)
            results.extend(rf)

            fb = compare_fastvqa_metrics(
                recording_scores,
                args.comp_results_dir,
                args.vid_dir,
                comp,
                evaluator,
                opt,
            )
            results.extend(fb)

            with open(scores_file, "w") as f:
                json.dump(results, f, indent=4)

            logger.info(f"Saved results to {scores_file}")
        else:
            logger.info(f"Skipping {comp} as scores already exist")

        extra_scores_file = args.comp_results_dir / "scores" / f"{comp}_extra.json"
        if args.part != EvalArgs.Part.BASE and not extra_scores_file.exists():
            logger.info("Extra metrics")
            rb = compare_real_base_metrics(recordings, args.comp_results_dir, comp)
            with open(extra_scores_file, "w") as f:
                json.dump(rb, f, indent=4)
            logger.info(f"Saved extra metrics to {extra_scores_file}")
        else:
            logger.info(f"Skipping extra metrics for {comp}")


if __name__ == "__main__":
    main()
