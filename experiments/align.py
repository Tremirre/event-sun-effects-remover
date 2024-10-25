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
    window_length: int
    ref_frame_idx: int
    check_first_ms: int
    n_homography_strips: int = 4

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
            "--window_length",
            required=False,
            type=int,
            default=50,
            help="Window length for event matching",
        )
        parser.add_argument(
            "--ref_frame_idx",
            required=False,
            type=int,
            default=5,
            help="Reference frame index",
        )
        parser.add_argument(
            "--check_first_ms",
            required=False,
            type=int,
            default=3000,
            help="Resolve spatio-temporal alignment for the first N milliseconds",
        )
        parser.add_argument(
            "--n_homography_strips",
            required=False,
            type=int,
            default=4,
            help="Number of horizontal homography strips, used to refine the initial homography",
        )
        return cls(**vars(parser.parse_args()))

    def __post_init__(self):
        assert self.input_events.exists(), f"{self.input_events} does not exist"
        assert self.input_video.exists(), f"{self.input_video} does not exist"


class FeatureMeasurement:
    def __init__(self, reference_img: np.ndarray):
        self.alg = cv2.AffineFeature.create(cv2.SIFT.create())
        self.kp_ref, self.des_ref = self.alg.detectAndCompute(reference_img, None)
        self.kp_checked = []
        self.des_checked = []
        self.matches = []

    def measure(self, img: np.ndarray) -> None:
        kp2, des2 = self.alg.detectAndCompute(img, None)
        self.kp_checked.append(kp2)
        self.des_checked.append(des2)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(self.des_ref, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append([m])
        self.matches.append(len(good))


@dataclasses.dataclass
class MatchResult:
    rec_frames: np.ndarray
    measurement: FeatureMeasurement

    @property
    def best_idx(self) -> int:
        return np.argmax(self.measurement.matches)

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
    fm = FeatureMeasurement(ref_frame_gs)
    rec = []
    prev = None
    if verbose:
        event_it = tqdm.tqdm(event_it, total=len(event_it), desc="Matching events")
    for window in event_it:
        # window = window.astype(int)
        vg = events_to_voxel_grid(window, 5, width, height)
        vg = torch.from_numpy(vg).unsqueeze(0).float().to(const.DEVICE)
        with torch.no_grad():
            pred, prev = model(vg, prev)
            pred = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
            pred = cv2.undistort(pred, const.EVENT_MTX, const.EVENT_DIST)
            # sharpen
            pred_gb = cv2.GaussianBlur(pred, (0, 0), 3)
            pred = cv2.addWeighted(pred, 1.5, pred_gb, -0.5, 0)

        fm.measure(pred)
        out.write(cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR))
        rec.append(pred)

    out.release()
    return MatchResult(np.array(object=rec), fm)


def make_horiz_masks(
    n: int, width: int, height: int, overlap: int = 0
) -> list[np.ndarray]:
    masks = []
    for i in range(n):
        mask = np.zeros((height, width), dtype=np.uint8)
        window_start = max(i * width // n - overlap, 0)
        window_end = min((i + 1) * width // n + overlap, width)
        mask[:, window_start:window_end] = 255
        masks.append(mask)
    return masks


@dataclasses.dataclass
class StripSpec:
    mask: np.ndarray
    homography: np.ndarray
    matches: int

    @property
    def inv_homography(self) -> np.ndarray:
        return np.linalg.inv(self.homography)


def refine_homography(
    src_frame: np.ndarray, event_frame_gs: np.ndarray, num_strips: int
) -> tuple[StripSpec]:
    overlap_masks = make_horiz_masks(
        num_strips, src_frame.shape[1], src_frame.shape[0], 120
    )
    masks = make_horiz_masks(num_strips, src_frame.shape[1], src_frame.shape[0])
    src_frame_gs = cv2.cvtColor(src_frame, cv2.COLOR_BGR2GRAY)
    event_frame_gs_gb = cv2.GaussianBlur(event_frame_gs, (0, 0), 3)
    event_frame_gs = cv2.addWeighted(event_frame_gs, 1.5, event_frame_gs_gb, -0.5, 0)

    alg = cv2.AffineFeature.create(cv2.SIFT.create())
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    specs = []
    for o_mask, mask in zip(overlap_masks, masks):
        kp1, des1 = alg.detectAndCompute(src_frame_gs, o_mask)
        kp2, des2 = alg.detectAndCompute(event_frame_gs, o_mask)
        matches = matcher.match(des1, des2)
        assert len(matches) > 0, "No matches found"
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        specs.append(StripSpec(mask, H, len(matches)))
    return specs


def save_refinement_result(
    event_frame_gs: np.ndarray, specs: list[StripSpec], path: pathlib.Path
):
    full = np.zeros_like(event_frame_gs)
    fig, ax = plt.subplots(1, len(specs) + 2, figsize=((len(specs) + 2) * 5, 5))
    ax[0].imshow(event_frame_gs, cmap="gray")
    for i, spec in enumerate(specs):
        warped = cv2.warpPerspective(
            event_frame_gs, spec.inv_homography, (full.shape[1], full.shape[0])
        )
        full[spec.mask > 0] = warped[spec.mask > 0]
        ax[i + 1].imshow(warped, cmap="gray")
    ax[-1].imshow(full, cmap="gray")
    plt.savefig(path)


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
    checked_counts = ts_counts[: args.check_first_ms]
    checked_events = events.array[: checked_counts.sum()]

    result = match_events_with_frame(
        checked_events,
        checked_counts,
        video[args.ref_frame_idx],
        events.width,
        events.height,
        window_length=args.window_length,
        model=model,
    )
    logging.info(f"Best match index: {result.best_idx}")
    temporal_offset = int(
        result.resolve_temporal_offset(args.window_length) - v_ts[args.ref_frame_idx]
    )
    logging.info(f"Best match temporal offset: {temporal_offset}ms")
    out_img = (
        const.IMAGES_DIR / f"matches-{args.window_length}-{args.input_video.stem}.png"
    )
    logging.info(f"Saving matches by frame to {out_img}")
    plt.plot(result.measurement.matches)
    plt.savefig(out_img)

    logging.info("Refining homography")
    specs = refine_homography(
        video[args.ref_frame_idx],
        result.rec_frames[result.best_idx],
        args.n_homography_strips,
    )
    logging.info(
        f"Matches by strip: {', '.join([str(spec.matches) for spec in specs])}"
    )
    logging.info(f"Total matches: {sum([spec.matches for spec in specs])}")
    ref_path = pathlib.Path(const.IMAGES_DIR) / "refinement.png"
    logging.info(f"Saving refinement result to {ref_path}")
    save_refinement_result(result.rec_frames[result.best_idx], specs, ref_path)

    out_meta = const.META_DIR / f"match-{args.input_video.stem}.json"
    logging.info(f"Saving metadata to {out_meta}")

    with open(out_meta, "w") as f:
        json.dump(
            {
                "homographies": [spec.inv_homography.tolist() for spec in specs],
                "temporal_offset": temporal_offset,
            },
            f,
        )
