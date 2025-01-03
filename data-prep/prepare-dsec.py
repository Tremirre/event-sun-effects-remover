from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib

import cv2
import h5py
import hdf5plugin
import numpy as np
import torch
import tqdm
from rpg_e2vid.utils.inference_utils import events_to_voxel_grid

import const
import utils

assert hdf5plugin is not None, "hdf5plugin is not available"
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)


@dataclasses.dataclass
class Args:
    input_dir: pathlib.Path
    skip_every: int
    output_folder: pathlib.Path
    save_debug: bool

    @classmethod
    def from_cli(cls) -> Args:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--input_dir",
            required=True,
            type=pathlib.Path,
            help="Path to the input events .bin file",
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
            "--save-debug",
            required=False,
            action="store_true",
            help="Save debug recording of the reconstruction",
        )
        return cls(**vars(parser.parse_args()))

    @property
    def events_path(self) -> pathlib.Path:
        return self.input_dir / "events.h5"

    @property
    def rectify_path(self) -> pathlib.Path:
        return self.input_dir / "rectify_map.h5"

    @property
    def timestamps_path(self) -> pathlib.Path:
        return self.input_dir / "timestamps.txt"

    @property
    def calib_path(self) -> pathlib.Path:
        return self.input_dir / "cam_to_cam.yaml"

    @property
    def img_paths(self) -> list[pathlib.Path]:
        return list((self.input_dir / "imgs").glob("*.png"))

    def __post_init__(self):
        assert self.input_dir.is_dir(), "Input directory does not exist"
        assert self.events_path.is_file(), "Events file does not exist"
        assert self.rectify_path.is_file(), "Rectify file does not exist"
        assert self.timestamps_path.is_file(), "Timestamps file does not exist"
        assert self.calib_path.is_file(), "Calibration file does not exist"
        assert self.img_paths, "No images found in the input directory"

        self.output_folder.mkdir(parents=True, exist_ok=True)


def read_rect_map(rectify_path: pathlib.Path) -> np.ndarray:
    with h5py.File(rectify_path, "r") as f:
        return f["rectify_map"][:]  # type: ignore


def read_events_h5(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        offset = f["t_offset"][()]
        x = f["events"]["x"][()].astype(np.int32)
        y = f["events"]["y"][()].astype(np.int32)
        ts = (f["events"]["t"][()].astype(np.int64) + offset) // 1_000
        pol = f["events"]["p"][()].astype(np.int32)
        events = np.array([ts, x, y, pol]).T
        ms_to_idx = f["ms_to_idx"][()].astype(np.int32)
        return events, ms_to_idx


def export_frames(
    src_frames: utils.Frames,
    events: np.ndarray,
    ms_to_idx: np.ndarray,
    rect_map: np.ndarray,
    model: torch.nn.Module,
    output_folder: pathlib.Path,
    skip_every: int,
    prefix: str,
    t_height: int,
    t_width: int,
    save_debug: bool = False,
):
    freq = (src_frames.timestamps[1:] - src_frames.timestamps[:-1]).mean() / 1_000
    freq = int(round(freq))
    ev_freq = freq // 2
    logging.info(f"Overlaying frames with a frequency of {freq} ms")

    artifact_mask = utils.get_rect_artifact_mask(rect_map)
    degrid_kernel = np.ones((3, 3), np.float32)
    degrid_kernel[1, 1] = 0
    degrid_kernel /= degrid_kernel.sum()
    out = None
    if save_debug:
        debug_path = output_folder / f"{prefix}_debug.mp4"
        logging.info(f"Saving debug recording to {debug_path}")
        out = cv2.VideoWriter(
            str(debug_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            40,
            (t_width * 2, t_height),
        )

    pbar = tqdm.tqdm(range(1, len(ms_to_idx), ev_freq), desc="Overlaying frames")
    for i in pbar:
        from_idx = ms_to_idx[i - ev_freq]
        to_idx = ms_to_idx[i]
        window = events[from_idx:to_idx]
        window[:, [1, 2]] = rect_map[window[:, 2], window[:, 1]]
        window = window[
            (window[:, 1] >= 0)
            & (window[:, 1] < rect_map.shape[1])
            & (window[:, 2] >= 0)
            & (window[:, 2] < rect_map.shape[0])
        ]

        window = window.astype(np.float32)
        if not len(window):
            continue

        voxel_grid = events_to_voxel_grid(window, 5, t_width, t_height)
        for j in range(len(voxel_grid)):
            conved = cv2.filter2D(voxel_grid[j], -1, degrid_kernel)
            voxel_grid[j] = np.where(artifact_mask, conved, voxel_grid[j])
        voxel_grid = torch.from_numpy(voxel_grid).to(const.DEVICE).unsqueeze(0).float()

        with torch.no_grad():
            pred = model(voxel_grid)["image"]
            pred = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)

        frame_idx = i // freq

        frame = src_frames.array[frame_idx]
        if out:
            pred_bgr = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
            frame_out = np.concatenate([frame, pred_bgr], axis=1)
            out.write(frame_out)

        if (frame_idx > 0 and frame_idx % skip_every != 0) or (i - 1) % freq != 0:
            continue

        full_pred = np.expand_dims(pred, axis=-1)
        frame = np.concatenate([frame, full_pred], axis=-1)
        output_path = output_folder / f"{prefix}_{frame_idx:>05}.npy"
        np.save(output_path, frame)
    if out:
        out.release()


if __name__ == "__main__":
    args = Args.from_cli()
    logging.info(f"Loading events from {args.events_path}")
    events, ms_to_idx = read_events_h5(args.events_path)
    width = events[:, 1].max() + 1
    height = events[:, 2].max() + 1
    logging.info(f"Loading rectify map from {args.rectify_path}")
    rect_map = read_rect_map(args.rectify_path)

    logging.info(
        f"Loading images from {args.input_dir}, timestamps from {args.timestamps_path} and calibration from {args.calib_path}"
    )
    src_frames = utils.Frames.from_paths(
        args.img_paths, args.timestamps_path, args.calib_path
    )

    logging.info("Loading model")
    model = utils.load_model_2(const.PRETRAINED_DIR / "better_e2vid_weights_v5.pth").to(
        const.DEVICE
    )
    model.reset_states()
    model.eval()

    export_frames(
        src_frames,
        events,
        ms_to_idx,
        rect_map,
        model,
        args.output_folder,
        args.skip_every,
        args.input_dir.name,
        height,
        width,
        save_debug=args.save_debug,
    )
