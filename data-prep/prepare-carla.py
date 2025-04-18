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

EVENT_FORMAT = "all_events_{}.npy"
FRAME_FORMAT = "bgr_frames_{}.npy"
FRAME_TS_FORMAT = "bgr_timestamps_{}.npy"
SUFFIX_LENGTH = 5


@dataclasses.dataclass
class Args:
    input: pathlib.Path
    skip_every: int
    output_folder: pathlib.Path
    event_width: int = 640
    event_height: int = 480

    @classmethod
    def from_cli(cls) -> Args:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--input",
            required=True,
            type=pathlib.Path,
            help="Path to the input directory containing events, frames, and timestamps",
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
        assert self.input.is_dir(), f"Input path {self.input} is not a directory"
        n_events = len(list(self.input.glob(EVENT_FORMAT.format("*"))))
        n_frames = len(list(self.input.glob(FRAME_FORMAT.format("*"))))
        n_frame_ts = len(list(self.input.glob(FRAME_TS_FORMAT.format("*"))))
        assert n_events > 0, f"No event files found in {self.input}"
        assert n_frames > 0, f"No frame files found in {self.input}"
        assert n_frame_ts > 0, f"No frame timestamp files found in {self.input}"

        assert (
            n_events == n_frames
        ), f"Number of events ({n_events}) does not match number of frames ({n_frames})"
        assert (
            n_events == n_frame_ts
        ), f"Number of events ({n_events}) does not match number of frame timestamps ({n_frame_ts})"

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
    logging.info(f"Input directory: {args.input}")
    logging.info(f"Output directory: {args.output_folder}")

    logging.info("Loading model")
    model = load_model(const.PRETRAINED_DIR / "E2VID_lightweight.pth.tar").to(
        const.DEVICE
    )
    suffixes = [
        path.stem[-SUFFIX_LENGTH:] for path in args.input.glob(EVENT_FORMAT.format("*"))
    ]

    suffixes = sorted(set(suffixes))

    logging.info(f"Found {len(suffixes)} suffixes: {suffixes}")

    for suffix in suffixes:
        logging.info(f"Processing suffix: {suffix}")
        input_events = args.input / EVENT_FORMAT.format(suffix)
        input_frames = args.input / FRAME_FORMAT.format(suffix)
        input_timestamps = args.input / FRAME_TS_FORMAT.format(suffix)

        logging.info(f"Input events: {input_events}")
        logging.info(f"Input video: {input_frames}")
        logging.info(f"Input timestamps: {input_timestamps}")

        # model.eval()

        logging.info(f"Loading events from {input_events}")
        events_orig = np.load(input_events)
        events = np.stack(
            [events_orig["t"], events_orig["x"], events_orig["y"], events_orig["p"]]
        ).T.astype(np.int64)
        event_timestamps = (events[:, 0] / 1e6).astype(np.int64)
        _, ts_counts = np.unique(event_timestamps, return_counts=True)

        first_timestamp = event_timestamps[0]
        logging.info(f"First timestamp: {first_timestamp}")
        logging.info(f"Width: {args.event_width}, Height: {args.event_height}")
        logging.info(f"Number of events: {len(events)}")
        logging.info(f"Loading frames from {input_frames}")

        src_frames = np.load(input_frames)
        src_frames = np.stack(
            [
                cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_RGB2BGR)
                for frame in src_frames
            ]
        )
        logging.info(f"Number of frames: {len(src_frames)}")

        frame_timestamps = (np.load(input_timestamps) * 1e3).astype(np.int64)
        frame_timestamps -= frame_timestamps.min()
        src_frames = utils.Frames(src_frames, frame_timestamps)

        export_frames(
            src_frames,
            events,
            ts_counts,
            model,
            args.output_folder,
            args.skip_every,
            args.input.stem + "_" + suffix,
        )
