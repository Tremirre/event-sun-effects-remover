from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib

import cv2
import numpy as np
import torch
import tqdm
import utils
from rpg_e2vid.utils.inference_utils import events_to_voxel_grid
from rpg_e2vid.utils.loading_utils import load_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PRETRAINED_DIR = pathlib.Path("pretrained")
META_DIR = pathlib.Path("experiments/meta")
VIDEOS_DIR = pathlib.Path("experiments/videos")
IMAGES_DIR = pathlib.Path("experiments/images")
META_DIR.mkdir(exist_ok=True)
VIDEOS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)


@dataclasses.dataclass
class Args:
    input_events: pathlib.Path
    input_video: pathlib.Path
    input_meta: pathlib.Path

    @classmethod
    def from_cli(cls) -> Args:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--input_events",
            required=True,
            type=pathlib.Path,
            help="Path to the input events .bin file",
        )
        parser.add_argument(
            "--input_video",
            required=True,
            type=pathlib.Path,
            help="Path to the input video .mp4 file",
        )
        parser.add_argument(
            "--input_meta",
            required=True,
            type=pathlib.Path,
            help="Path to the meta file",
        )
        return cls(**vars(parser.parse_args()))

    def __post_init__(self):
        assert self.input_events.exists(), f"{self.input_events} does not exist"
        assert self.input_video.exists(), f"{self.input_video} does not exist"
        assert self.input_meta.exists(), f"{self.input_meta} does not exist"


@dataclasses.dataclass
class AlignMeta:
    homography: np.ndarray
    offset_ms: int

    @classmethod
    def from_json(cls, path: pathlib.Path) -> AlignMeta:
        data = json.loads(path.read_text())
        homography = np.array(data["homography"])
        offset_ms = data["temporal_offset"]
        return cls(homography, offset_ms)


def overlay_events_on_video(
    video: np.ndarray,
    v_timestamps: np.ndarray,
    events: np.ndarray,
    ts_counts: np.ndarray,
    homography: np.ndarray,
    model: torch.nn.Module,
) -> np.ndarray:
    assert video.shape[0] == len(v_timestamps), "Video and timestamps mismatch"
    width, height = video.shape[2], video.shape[1]
    count_index = 0
    event_index = 0
    overlayed = np.zeros_like(video)
    prev = None
    for i in tqdm.tqdm(range(1, video.shape[0])):
        if v_timestamps[i] == 0:
            continue
        window_start = v_timestamps[i - 1]
        window_end = v_timestamps[i]
        events_in_window = ts_counts[window_start:window_end].sum()
        events_end = event_index + events_in_window
        window = events[event_index:events_end]
        if not len(window):
            break
        count_index += v_timestamps[i]
        event_index = events_end
        voxel_grid = events_to_voxel_grid(window, 5, width, height)
        voxel_grid = torch.from_numpy(voxel_grid).to(DEVICE).unsqueeze(0).float()
        with torch.no_grad():
            pred, prev = model(voxel_grid, prev)
            pred = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
        pred_edges = (cv2.Canny(pred, 50, 200) > 0).astype(np.uint8) * 255
        pred_edges = cv2.dilate(pred_edges, np.ones((3, 3), np.uint8), iterations=1)
        pred_edges = cv2.warpPerspective(pred_edges, homography, (width, height))
        pred_edges = cv2.cvtColor(pred_edges, cv2.COLOR_GRAY2BGR)
        overlayed_frame = cv2.addWeighted(video[i], 0.7, pred_edges, 0.3, 0)
        overlayed[i] = overlayed_frame
    overlayed = overlayed[:i]
    return overlayed


if __name__ == "__main__":
    args = Args.from_cli()
    logging.info(f"Input events: {args.input_events}")
    logging.info(f"Input video: {args.input_video}")
    logging.info(f"Input meta: {args.input_meta}")

    alignment_meta = AlignMeta.from_json(args.input_meta)

    logging.info("Loading model")
    model = load_model(PRETRAINED_DIR / "E2VID_lightweight.pth.tar").to(DEVICE)
    model.eval()

    logging.info(f"Loading events from {args.input_events}")
    events = utils.EventsData.from_path(args.input_events)

    first_timestamp = events.array[0, 0]
    logging.info(f"First timestamp: {first_timestamp}")
    logging.info(f"Width: {events.width}, Height: {events.height}")
    logging.info(f"Number of events: {len(events.array)}")
    video, v_timestamps = utils.read_video(args.input_video)

    logging.info(f"Resizing video to {events.width}x{events.height}")
    video = utils.crop_to_size(video, events.width, events.height)

    _, ts_counts = np.unique(events.array[:, 0], return_counts=True)

    logging.info(f"Applying offset {alignment_meta.offset_ms}ms")
    skip_events = ts_counts[: alignment_meta.offset_ms].sum()
    events = events.array[skip_events:]
    ts_counts = ts_counts[alignment_meta.offset_ms :]
    assert len(events) == ts_counts.sum(), "Events and counts mismatch"
    hom_inv = np.linalg.inv(alignment_meta.homography)
    overlayed = overlay_events_on_video(
        video, v_timestamps, events, ts_counts, hom_inv, model
    )

    out_video = IMAGES_DIR / f"overlayed-{args.input_video.stem}.mp4"
    logging.info(f"Saving overlayed video to {out_video}")
    out = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (video.shape[2], video.shape[1]),
    )
    for frame in overlayed:
        out.write(frame)
    out.release()
