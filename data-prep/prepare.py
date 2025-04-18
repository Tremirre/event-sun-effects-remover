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
    input_video: pathlib.Path
    input_meta: pathlib.Path
    skip_every: int
    output_folder: pathlib.Path

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
        return cls(**vars(parser.parse_args()))

    def __post_init__(self):
        assert self.input_events.exists(), f"{self.input_events} does not exist"
        assert self.input_video.exists(), f"{self.input_video} does not exist"
        assert self.input_meta.exists(), f"{self.input_meta} does not exist"
        self.output_folder.mkdir(parents=True, exist_ok=True)


def export_frames(
    src_frames: utils.Frames,
    events: np.ndarray,
    ts_counts: np.ndarray,
    alignment: utils.AlignMeta,
    model: torch.nn.Module,
    output_folder: pathlib.Path,
    skip_every: int,
    prefix: str,
):
    count_index = 0
    event_index = 0
    prev = None
    masks = utils.make_horiz_masks(
        len(alignment.homographies), src_frames.width, src_frames.height
    )
    hom_mask = alignment.get_common_mask(src_frames.width, src_frames.height) > 0
    hom_mask = np.expand_dims(hom_mask, axis=-1).astype(np.uint8) * 255
    for i in tqdm.tqdm(range(1, src_frames.num_frames), desc="Overlaying frames"):
        if src_frames.timestamps[i] == 0:
            continue
        window_start = src_frames.timestamps[i - 1]
        window_end = src_frames.timestamps[i]
        events_in_window = ts_counts[window_start:window_end].sum()
        events_end = event_index + events_in_window
        window = events[event_index:events_end]
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

        pred = cv2.undistort(pred, const.EVENT_MTX, const.EVENT_DIST)
        pred_gb = cv2.GaussianBlur(pred, (0, 0), 3)
        pred = cv2.addWeighted(pred, 1.5, pred_gb, -0.5, 0)

        full_pred = np.zeros((src_frames.height, src_frames.width), np.uint8)
        for mask, hom in zip(masks, alignment.homographies):
            hom_warped = cv2.warpPerspective(
                pred.copy(), hom, (src_frames.width, src_frames.height)
            )
            full_pred[mask > 0] = hom_warped[mask > 0]

        src_frame = src_frames.array[i]
        full_pred = np.expand_dims(full_pred, axis=-1)
        frame = np.concatenate([src_frame, full_pred, hom_mask], axis=-1)
        output_path = output_folder / f"{prefix}_{i:>05}.npy"
        np.save(output_path, frame)


if __name__ == "__main__":
    args = Args.from_cli()
    logging.info(f"Input events: {args.input_events}")
    logging.info(f"Input video: {args.input_video}")
    logging.info(f"Input meta: {args.input_meta}")

    alignment_meta = utils.AlignMeta.from_json(args.input_meta)
    logging.info("Loading model")
    model = load_model(const.PRETRAINED_DIR / "E2VID_lightweight.pth.tar").to(
        const.DEVICE
    )
    model.eval()

    logging.info(f"Loading events from {args.input_events}")
    events = utils.EventsData.from_path(args.input_events)

    first_timestamp = events.array[0, 0]
    logging.info(f"First timestamp: {first_timestamp}")
    logging.info(f"Width: {events.width}, Height: {events.height}")
    logging.info(f"Number of events: {len(events.array)}")
    src_frames = utils.read_video(args.input_video)

    logging.info(f"Resizing video to {events.width}x{events.height}")
    src_frames.array = utils.crop_vid_to_size(
        src_frames.array, events.width, events.height
    )

    _, ts_counts = np.unique(events.array[:, 0], return_counts=True)

    if alignment_meta.offset_ms > 0:
        logging.info(f"Applying offset {alignment_meta.offset_ms}ms to events")
        skip_events = ts_counts[: alignment_meta.offset_ms].sum()
        events.array = events.array[skip_events:]
        ts_counts = ts_counts[alignment_meta.offset_ms :]
    else:
        logging.info(f"Applying offset {-alignment_meta.offset_ms}ms to video")
        skip_frames_ms = -alignment_meta.offset_ms
        skip_frames = (src_frames.timestamps < skip_frames_ms).sum()
        logging.info(f"Skipping {skip_frames} frames")
        src_frames.array = src_frames.array[skip_frames:]
        src_frames.timestamps = (
            src_frames.timestamps[skip_frames:] - src_frames.timestamps[skip_frames]
        )
    assert len(events.array) == ts_counts.sum(), "Events and counts mismatch"
    export_frames(
        src_frames,
        events.array,
        ts_counts,
        alignment_meta,
        model,
        args.output_folder,
        args.skip_every,
        args.input_video.stem,
    )
