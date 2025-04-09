from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib

import cv2
import numpy as np
import torch
import tqdm
from rpg_e2vid.utils.inference_utils import events_to_voxel_grid
from rpg_e2vid.utils.loading_utils import load_model

import const
import utils

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)


@dataclasses.dataclass
class Args:
    input_events: pathlib.Path
    input_frames: pathlib.Path
    input_timestamps: pathlib.Path
    series_name: str
    skip_every: int
    output_folder: pathlib.Path
    event_width: int = 640
    event_height: int = 480

    @classmethod
    def from_cli(cls) -> Args:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--input_events",
            required=True,
            type=pathlib.Path,
            help="Path to the input events .npy file",
        )
        parser.add_argument(
            "--input_frames",
            required=True,
            type=pathlib.Path,
            help="Path to the input RGB .npy file",
        )
        parser.add_argument(
            "--input_timestamps",
            required=True,
            type=pathlib.Path,
            help="Path to the frames timestamps .npy file",
        )
        parser.add_argument(
            "--skip_every",
            required=False,
            type=int,
            default=3,
            help="Skip every n frames",
        )
        parser.add_argument(
            "--output_folder",
            required=True,
            type=pathlib.Path,
            help="Path to the output folder",
        )
        parser.add_argument(
            "--series_name",
            required=True,
            type=str,
            help="Name of the series for the output folder",
        )
        parser.add_argument(
            "--event_width",
            required=False,
            type=int,
            default=640,
            help="Width of the event frames",
        )
        parser.add_argument(
            "--event_height",
            required=False,
            type=int,
            default=480,
            help="Height of the event frames",
        )
        return cls(**vars(parser.parse_args()))

    def __post_init__(self):
        assert self.input_events.exists(), f"{self.input_events} does not exist"
        assert self.input_frames.exists(), f"{self.input_frames} does not exist"
        assert self.input_timestamps.exists(), f"{self.input_timestamps} does not exist"
        self.output_folder.mkdir(parents=True, exist_ok=True)


def export_frames(
    src_frames: utils.Frames,
    events: np.ndarray,
    ts_counts: np.ndarray,
    model: torch.nn.Module,
    output_folder: pathlib.Path,
    skip_every: int,
    prefix: str,
):
    count_index = 0
    event_index = 0
    prev = None
    for i in tqdm.tqdm(range(1, src_frames.num_frames), desc="Overlaying frames"):
        if src_frames.timestamps[i] == 0:
            continue
        window_start = src_frames.timestamps[i - 1]
        window_end = src_frames.timestamps[i]
        events_in_window = ts_counts[window_start:window_end].sum()
        events_end = event_index + events_in_window
        window = events[event_index:events_end].copy()
        count_index += src_frames.timestamps[i]
        event_index = events_end
        if not len(window):
            break

        voxel_grid = events_to_voxel_grid(
            window, 5, src_frames.width, src_frames.height
        )
        voxel_grid = torch.from_numpy(voxel_grid).to(const.DEVICE).unsqueeze(0).float()
        with torch.no_grad():
            pred, prev = model(voxel_grid, prev)
            pred = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)

        if skip_every > 0 and i % skip_every != 0:
            continue

        pred = pred[..., np.newaxis]
        pred_gb = cv2.GaussianBlur(pred, (0, 0), 3)
        pred = cv2.addWeighted(pred, 1.5, pred_gb, -0.5, 0)[..., np.newaxis]

        src_frame = src_frames.array[i]
        frame = np.concatenate([src_frame, pred], axis=-1)
        output_path = output_folder / f"{prefix}_{i:>05}.npy"
        np.save(output_path, frame)


if __name__ == "__main__":
    args = Args.from_cli()
    logging.info(f"Input events: {args.input_events}")
    logging.info(f"Input video: {args.input_frames}")
    logging.info(f"Input timestamps: {args.input_timestamps}")

    logging.info("Loading model")
    model = load_model(const.PRETRAINED_DIR / "E2VID_lightweight.pth.tar").to(
        const.DEVICE
    )
    # model.eval()

    logging.info(f"Loading events from {args.input_events}")
    events_orig = np.load(args.input_events)
    events = np.stack(
        [events_orig["t"], events_orig["x"], events_orig["y"], events_orig["p"]]
    ).T.astype(np.int64)
    event_timestamps = (events[:, 0] / 1e6).astype(np.int64)
    _, ts_counts = np.unique(event_timestamps, return_counts=True)

    first_timestamp = event_timestamps[0]
    logging.info(f"First timestamp: {first_timestamp}")
    logging.info(f"Width: {args.event_width}, Height: {args.event_height}")
    logging.info(f"Number of events: {len(events)}")
    logging.info(f"Loading frames from {args.input_frames}")

    src_frames = np.load(args.input_frames)
    src_frames = np.stack(
        [
            cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_RGB2BGR)
            for frame in src_frames
        ]
    )
    logging.info(f"Number of frames: {len(src_frames)}")

    frame_timestamps = (np.load(args.input_timestamps) * 1e3).astype(np.int64)
    frame_timestamps -= frame_timestamps.min()
    src_frames = utils.Frames(src_frames, frame_timestamps)

    export_frames(
        src_frames,
        events,
        ts_counts,
        model,
        args.output_folder,
        args.skip_every,
        args.series_name,
    )
