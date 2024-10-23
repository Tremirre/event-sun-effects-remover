from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib

import const
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import utils
from rpg_e2vid.utils.inference_utils import events_to_voxel_grid
from rpg_e2vid.utils.loading_utils import load_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)


@dataclasses.dataclass
class Args:
    input_events: pathlib.Path
    input_video: pathlib.Path

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
        return cls(**vars(parser.parse_args()))

    def __post_init__(self):
        assert self.input_events.exists(), f"{self.input_events} does not exist"
        assert self.input_video.exists(), f"{self.input_video} does not exist"


class ORBMeasurement:
    def __init__(self, reference_img: np.ndarray):
        self.orb = cv2.ORB.create()
        self.kp_ref, self.des_ref = self.orb.detectAndCompute(reference_img, None)
        self.kp_checked = []
        self.des_checked = []
        self.matches = []

    def measure(self, img: np.ndarray) -> None:
        kp2, des2 = self.orb.detectAndCompute(img, None)
        self.kp_checked.append(kp2)
        self.des_checked.append(des2)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(self.des_ref, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        self.matches.append(len(good))


@dataclasses.dataclass
class MatchResult:
    rec_frames: np.ndarray
    measurement: ORBMeasurement

    @property
    def best_idx(self) -> int:
        return np.argmax(self.measurement.matches)

    def resolve_homography(self) -> np.ndarray:
        kp1 = self.measurement.kp_ref
        kp2 = self.measurement.kp_checked[self.best_idx]
        matches = cv2.BFMatcher().knnMatch(
            self.measurement.des_ref, self.measurement.des_checked[self.best_idx], k=2
        )
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H

    def resolve_temporal_offset(self, window_length: int) -> int:
        return int(self.best_idx * window_length)


def match_events_with_frame(
    events: np.ndarray,
    ts_counts: np.ndarray,
    reference_frame: np.ndarray,
    width: int,
    height: int,
    window_length: int,
    model: torch.nn.Module,
    verbose: bool = True,
) -> MatchResult:
    event_it = utils.EventWindowIterator(
        events, ts_counts, window_length, stride=window_length
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        "experiments/videos/debug-match.mp4", fourcc, 20.0, (width, height)
    )

    ref_frame_gs = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    ref_frame_edged = (cv2.Canny(ref_frame_gs, 300, 400) > 0).astype(np.uint8) * 255
    ref_frame_edged = cv2.dilate(ref_frame_edged, np.ones((3, 3), np.uint8))
    orb_measure = ORBMeasurement(ref_frame_gs)
    rec = []
    prev = None
    if verbose:
        event_it = tqdm.tqdm(event_it, total=len(event_it), desc="Matching events")
    for window in event_it:
        window = window.astype(int)
        vg = events_to_voxel_grid(window, 5, width, height)
        vg = torch.from_numpy(vg).unsqueeze(0).float().to(const.DEVICE)
        with torch.no_grad():
            pred, prev = model(vg, prev)
            pred = pred.squeeze().cpu().numpy()
            pred *= 255
            pred = pred.astype(np.uint8)
            pred = cv2.undistort(pred, const.EVENT_MTX, const.EVENT_DIST)

        orb_measure.measure(pred)
        out.write(cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR))
        rec.append(pred)

    out.release()
    return MatchResult(np.array(rec), orb_measure)


ABC = 1
if __name__ == "__main__":
    args = Args.from_cli()

    model = load_model(const.PRETRAINED_DIR / "E2VID_lightweight.pth.tar").to(
        const.DEVICE
    )
    logging.info(f"Loading events from {args.input_events}")
    events = utils.EventsData.from_path(args.input_events)
    first_timestamp = events.array[0, 0]
    logging.info(f"First timestamp: {first_timestamp}")
    logging.info(f"Width: {events.width}, Height: {events.height}")
    logging.info(f"Number of events: {len(events.array)}")
    video, v_ts = utils.read_video(args.input_video)
    logging.info(f"Resizing video to {events.width}x{events.height}")
    video = utils.crop_to_size(video, events.width, events.height)
    _, ts_counts = np.unique(events.array[:, 0], return_counts=True)

    logging.info("Matching events with frame")
    checked_time_ms = 3000
    window_length = 35
    ref_frame_idx = 5
    checked_counts = ts_counts[:checked_time_ms]
    checked_events = events.array[: checked_counts.sum()]

    result = match_events_with_frame(
        checked_events,
        checked_counts,
        video[ref_frame_idx],
        events.width,
        events.height,
        window_length=window_length,
        model=model,
    )
    logging.info(f"Best match index: {result.best_idx}")
    logging.info(f"Best match homography: {result.resolve_homography()}")
    temporal_offset = int(
        result.resolve_temporal_offset(window_length) - v_ts[ref_frame_idx]
    )
    logging.info(f"Best match temporal offset: {temporal_offset}ms")

    out_img = const.IMAGES_DIR / f"matches-{window_length}-{args.input_video.stem}.png"
    logging.info(f"Saving matches by frame to {out_img}")
    plt.plot(result.measurement.matches)
    plt.savefig(out_img)

    out_meta = const.META_DIR / f"match-{args.input_video.stem}.json"
    logging.info(f"Saving metadata to {out_meta}")

    with open(out_meta, "w") as f:
        json.dump(
            {
                "homography": result.resolve_homography().astype(float).tolist(),
                "temporal_offset": temporal_offset,
            },
            f,
        )
