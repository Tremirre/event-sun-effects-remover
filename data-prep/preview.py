from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib

import const
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


def overlay_events_on_video(
    src_frames: utils.Frames,
    events: np.ndarray,
    ts_counts: np.ndarray,
    alignment: utils.AlignMeta,
    model: torch.nn.Module,
) -> tuple[np.ndarray, np.ndarray]:
    count_index = 0
    event_index = 0
    overlayed = np.zeros_like(src_frames.array)
    prev = None
    masks = utils.make_horiz_masks(
        len(alignment.homographies), src_frames.width, src_frames.height
    )
    hom_mask = alignment.get_common_mask(src_frames.width, src_frames.height) > 0
    metrics = []
    for i in tqdm.tqdm(range(1, src_frames.num_frames), desc="Overlaying frames"):
        if src_frames.timestamps[i] == 0:
            continue
        window_start = src_frames.timestamps[i - 1]
        window_end = src_frames.timestamps[i]
        events_in_window = ts_counts[window_start:window_end].sum()
        events_end = event_index + events_in_window
        window = events[event_index:events_end]
        if not len(window):
            break
        count_index += src_frames.timestamps[i]
        event_index = events_end
        voxel_grid = events_to_voxel_grid(
            window, 5, src_frames.width, src_frames.height
        )
        voxel_grid = torch.from_numpy(voxel_grid).to(const.DEVICE).unsqueeze(0).float()
        with torch.no_grad():
            pred, prev = model(voxel_grid, prev)
            pred = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
        pred = cv2.undistort(pred, const.EVENT_MTX, const.EVENT_DIST)
        pred_gb = cv2.GaussianBlur(pred, (0, 0), 3)
        pred = cv2.addWeighted(pred, 1.5, pred_gb, -0.5, 0)
        pred_edges = (cv2.Canny(pred, 50, 100) > 0).astype(np.uint8) * 255
        pred_edges = cv2.dilate(pred_edges, np.ones((3, 3), np.uint8), iterations=1)

        full_pred_edges = np.zeros((src_frames.height, src_frames.width), np.uint8)
        for mask, hom in zip(masks, alignment.homographies):
            hom_warped = cv2.warpPerspective(
                pred_edges.copy(), hom, (src_frames.width, src_frames.height)
            )
            full_pred_edges[mask > 0] = hom_warped[mask > 0]
        pred_edges = full_pred_edges
        overlay_gs = cv2.cvtColor(src_frames.array[i], cv2.COLOR_BGR2GRAY)
        overlay_edges = cv2.Canny(overlay_gs, 100, 200) > 0
        overlay_edges = cv2.dilate(
            overlay_edges.astype(np.uint8), np.ones((3, 3), np.uint8)
        )
        overlay_edges[~hom_mask] = 0
        tp_mask = np.logical_and(overlay_edges, pred_edges > 0)
        fp_mask = np.logical_and(overlay_edges == 0, pred_edges > 0)
        fn_mask = np.logical_and(overlay_edges, pred_edges == 0)
        true_positives = tp_mask.sum()
        false_positives = fp_mask.sum()
        false_negatives = fn_mask.sum()
        accuracy = true_positives / (true_positives + false_positives + false_negatives)
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * precision * recall / (precision + recall)
        if np.isnan(f1):
            f1 = 0

        overlay_quality = np.zeros((src_frames.height, src_frames.width, 3), np.uint8)
        overlay_quality[tp_mask] = [0, 255, 0]
        overlay_quality[fp_mask] = [0, 0, 255]
        overlay_quality[fn_mask] = [255, 0, 0]
        overlayed[i] = overlay_quality
        metrics.append((accuracy, precision, recall, f1))

    metrics = np.array(metrics)
    overlayed = overlayed[:i]
    return overlayed, metrics


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

    cv2.imwrite(
        const.IMAGES_DIR / "common_mask.png",
        alignment_meta.get_common_mask(events.width, events.height),
    )
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
        events = events.array[skip_events:]
        ts_counts = ts_counts[alignment_meta.offset_ms :]
    else:
        logging.info(f"Applying offset {-alignment_meta.offset_ms}ms to video")
        skip_frames_ms = -alignment_meta.offset_ms
        skip_frames = (src_frames.timestamps < skip_frames_ms).sum()
        src_frames.array = src_frames.array[skip_frames:]
        src_frames.timestamps = (
            src_frames.timestamps[skip_frames:] - src_frames.timestamps[skip_frames]
        )
    assert len(events) == ts_counts.sum(), "Events and counts mismatch"
    overlayed, metrics = overlay_events_on_video(
        src_frames, events, ts_counts, alignment_meta, model
    )

    logging.info(f"Accuracy: {metrics[:, 0].mean()}")
    logging.info(f"Precision: {metrics[:, 1].mean()}")
    logging.info(f"Recall: {metrics[:, 2].mean()}")
    logging.info(f"F1: {metrics[:, 3].mean()}")

    out_video = const.VIDEOS_DIR / f"overlayed-{args.input_video.stem}.mp4"
    logging.info(f"Saving overlayed video to {out_video}")
    out = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (src_frames.width, src_frames.height),
    )
    for frame in overlayed:
        out.write(frame)
    out.release()
