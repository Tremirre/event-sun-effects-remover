import json
import pathlib

import cv2
import ffmpeg_quality_metrics as fqm
import numpy as np
import tqdm

TEST_RES_PATH = pathlib.Path("data/test-res/")
VID_DIR = pathlib.Path("data/videos/")


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


def main():
    parsed_recordings = sorted(TEST_RES_PATH.glob("**/refscores.json"))
    for rec in tqdm.tqdm(parsed_recordings):
        reconstructed = rec.parent / "reconstructed.mp4"
        recording = rec.parent.stem
        reference = VID_DIR / f"{recording}.mp4"
        evaluator = fqm.FfmpegQualityMetrics(str(reference), str(reconstructed))
        fqm_scores = evaluator.calculate(["vmaf", "psnr"])
        real_scores = json.loads(rec.read_text())
        assert len(fqm_scores["psnr"]) == len(real_scores)
        for i, (psnr, vmaf) in enumerate(zip(fqm_scores["psnr"], fqm_scores["vmaf"])):
            real_scores[i]["psnr"] = psnr["psnr_avg"]
            real_scores[i]["mse"] = psnr["mse_avg"]
            real_scores[i]["vmaf"] = vmaf["vmaf"]
        rec.write_text(json.dumps(real_scores, indent=4))


if __name__ == "__main__":
    main()
