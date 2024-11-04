from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib
import typing

import const
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import utils
from rpg_e2vid.utils.loading_utils import load_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)


class FeatureAlg(typing.Protocol):
    def detectAndCompute(self, img: np.ndarray, mask: np.ndarray) -> tuple: ...


class MatchingAlg(typing.Protocol):
    def knnMatch(self, des1: np.ndarray, des2: np.ndarray, k: int) -> list: ...

    def match(self, des1: np.ndarray, des2: np.ndarray) -> list: ...


@dataclasses.dataclass
class Args:
    input_events: pathlib.Path
    input_video: pathlib.Path
    window_length: int
    check_offsets: int
    check_every: int
    skip_first_frames: int
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
            "--check_offsets",
            required=False,
            type=int,
            default=200,
            help="Number of offsets to check for temporal alignment",
        )
        parser.add_argument(
            "--check_every",
            required=False,
            type=int,
            default=50,
            help="Check every nth frame for spatio-temporal alignment",
        )
        parser.add_argument(
            "--skip_first_frames",
            required=False,
            type=int,
            default=100,
            help="Skip first n frames of the source video",
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


@dataclasses.dataclass
class TemporalMatchResult:
    matches: list[list[int]] = dataclasses.field(default_factory=list)
    align_indices: list[list[tuple[int, int]]] = dataclasses.field(default_factory=list)
    reiterated_matches: list[int] = dataclasses.field(default_factory=list)

    @property
    def avg_matches_by_offset(self) -> np.ndarray:
        return np.mean(self.matches, axis=1)

    def resolve_final_offset_ms(
        self,
        src_frames: utils.Frames,
        rec_frames: utils.Frames,
        window_length: int,
        skipped_ms: int,
    ) -> tuple[int, int]:
        assert self.matches, "No matches to improve"
        asift = cv2.AffineFeature.create(cv2.SIFT.create())
        bf = cv2.BFMatcher()
        best_matches = np.argsort(self.avg_matches_by_offset)[-10:]
        best_asift_matches = []
        logging.info(f"Current optimal index: {best_matches[-1]}")
        for idx in tqdm.tqdm(best_matches, desc="Refining matches"):
            align_matches = []
            align_indices = self.align_indices[idx]
            for j, (src_idx, rec_idx) in enumerate(align_indices):
                src_frame = src_frames.array[src_idx]
                rec_frame = rec_frames.array[rec_idx]
                src_frame_gs = cv2.cvtColor(src_frame, cv2.COLOR_BGR2GRAY)
                _, src_desc = asift.detectAndCompute(src_frame_gs, None)
                _, rec_desc = asift.detectAndCompute(rec_frame, None)
                cur_matches = bf.knnMatch(src_desc, rec_desc, k=2)
                good = []
                for m, n in cur_matches:
                    if m.distance < 0.7 * n.distance:
                        good.append(m)
                align_matches.append(len(good))
            best_asift_matches.append(align_matches)
        best_mean_matches = np.mean(best_asift_matches, axis=1)
        self.reiterated_matches = best_mean_matches
        logging.info(f"Best mean matches: {best_mean_matches}")
        best_reiterated = np.argmax(best_mean_matches)
        best_offset = best_matches[best_reiterated]
        logging.info(f"Reiterated best offset: {best_offset}")
        rem = 0
        if best_offset == 0 or best_offset == len(self.avg_matches_by_offset) - 1:
            logging.warning("Optimal index is at the edge of the range")
        else:
            rem = (
                -1
                if self.avg_matches_by_offset[best_offset - 1]
                > self.avg_matches_by_offset[best_offset + 1]
                else 1
            )
        rem *= window_length / 2
        return best_offset, best_offset * window_length - skipped_ms + rem

    def save_plot(self, path: pathlib.Path) -> None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(self.avg_matches_by_offset)
        ax[0].set_title("Average matches by offset")
        ax[1].plot(self.reiterated_matches)
        ax[1].set_title("Reiterated matches")
        plt.savefig(path)

    def update(self, matches: list[int], align_indices: list[tuple[int, int]]) -> None:
        self.matches.append(matches)
        self.align_indices.append(align_indices)


def resolve_temporal_offset(
    src_frames: utils.Frames,
    rec_frames: utils.Frames,
    window_length: int,
    feature_alg: FeatureAlg,
    matching_alg: MatchingAlg,
    check_offsets: int = 100,
    check_every: int = 50,
    nn_thresh: float = 0.7,
    verbose: bool = True,
) -> TemporalMatchResult:
    res = TemporalMatchResult()
    src_descs = []
    check_indices = range(0, len(src_frames.array), check_every)
    for i in check_indices:
        src_frame = src_frames.array[i]
        src_frame_gs = cv2.cvtColor(src_frame, cv2.COLOR_BGR2GRAY)
        _, des = feature_alg.detectAndCompute(src_frame_gs, None)
        src_descs.append(des)

    offsets = range(check_offsets)
    if verbose:
        offsets = tqdm.tqdm(offsets, desc="Checking offsets")

    for offset in offsets:
        matches = []
        align_indices = []
        for i, src_desc in zip(check_indices, src_descs):
            aligned_rec_ts = offset * window_length + src_frames.timestamps[i]
            rec_idx = np.argmin(np.abs(rec_frames.timestamps - aligned_rec_ts))
            align_indices.append((i, rec_idx))
            rec_frame = rec_frames.array[rec_idx]
            _, rec_desc = feature_alg.detectAndCompute(rec_frame, None)
            cur_matches = matching_alg.knnMatch(src_desc, rec_desc, k=2)
            good = []
            for m, n in cur_matches:
                if m.distance < nn_thresh * n.distance:
                    good.append(m)
            num_matches = len(good)
            matches.append(num_matches)
        res.update(matches, align_indices)
    return res


@dataclasses.dataclass
class StripSpec:
    mask: np.ndarray
    homography: np.ndarray
    matches: int

    @property
    def inv_homography(self) -> np.ndarray:
        return np.linalg.inv(self.homography)


def resolve_homography(
    src_frame: np.ndarray,
    event_frame_gs: np.ndarray,
    num_strips: int,
    feature_alg: FeatureAlg,
    matching_alg: MatchingAlg,
) -> tuple[StripSpec]:
    overlap_masks = utils.make_horiz_masks(
        num_strips, src_frame.shape[1], src_frame.shape[0], 120
    )
    masks = utils.make_horiz_masks(num_strips, src_frame.shape[1], src_frame.shape[0])
    src_frame_gs = cv2.cvtColor(src_frame, cv2.COLOR_BGR2GRAY)

    specs = []
    for o_mask, mask in zip(overlap_masks, masks):
        kp1, des1 = feature_alg.detectAndCompute(src_frame_gs, o_mask)
        kp2, des2 = feature_alg.detectAndCompute(event_frame_gs, o_mask)
        matches = matching_alg.match(des1, des2)
        assert len(matches) > 0, "No matches found"
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        specs.append(StripSpec(mask, H, len(matches)))
    return specs


def save_homography_result(
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
    src_frames = utils.read_video(args.input_video)
    logging.info(f"Skipping first {args.skip_first_frames} frames")
    skipped_ms = src_frames.timestamps[args.skip_first_frames]
    src_frames.array = src_frames.array[args.skip_first_frames :]
    src_frames.timestamps = src_frames.timestamps[args.skip_first_frames :] - skipped_ms

    logging.info(f"Resizing video to {events.width}x{events.height}")
    src_frames.array = utils.crop_vid_to_size(
        src_frames.array, events.width, events.height
    )
    _, ts_counts = np.unique(events.array[:, 0], return_counts=True)

    event_it = utils.EventWindowIterator(
        events.array, ts_counts, args.window_length, stride=args.window_length
    )
    logging.info("Reconstructing video from events")
    rec_frames_arr = utils.reconstruct_video(
        event_it, events.width, events.height, model
    )
    rec_frames_ts = np.arange(
        0,
        len(rec_frames_arr) * args.window_length,
        args.window_length,
    )
    rec_frames = utils.Frames(rec_frames_arr, rec_frames_ts)

    temporal_feature_alg = cv2.ORB.create()
    temporal_matching_alg = cv2.BFMatcher(cv2.NORM_HAMMING)

    logging.info("Resolving temporal offset")
    result = resolve_temporal_offset(
        src_frames,
        rec_frames,
        args.window_length,
        temporal_feature_alg,
        temporal_matching_alg,
        check_offsets=args.check_offsets,
        check_every=args.check_every,
    )
    matching_img_path = (
        const.IMAGES_DIR / f"matches-{args.window_length}-{args.input_video.stem}.png"
    )
    optimal_idx, temporal_offset = result.resolve_final_offset_ms(
        src_frames, rec_frames, args.window_length, skipped_ms
    )
    logging.info(f"Temporal offset: {temporal_offset}ms")
    logging.info(f"Saving matching plot to {matching_img_path}")
    result.save_plot(matching_img_path)

    logging.info("Refining homography")
    spatial_feature_alg = cv2.AffineFeature.create(cv2.SIFT.create())
    spatial_matching_alg = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    all_specs = []
    for src_idx, rec_idx in tqdm.tqdm(result.align_indices[optimal_idx]):
        specs = resolve_homography(
            src_frames.array[src_idx],
            rec_frames.array[rec_idx],
            args.n_homography_strips,
            spatial_feature_alg,
            spatial_matching_alg,
        )
        all_specs.append(specs)

    best_specs = all_specs[
        np.argmax([sum(s.matches for s in spec) for spec in all_specs])
    ]

    logging.info(
        f"Matches by strip: {', '.join([str(spec.matches) for spec in best_specs])}"
    )
    logging.info(f"Total matches: {sum([spec.matches for spec in best_specs])}")
    ref_path = pathlib.Path(const.IMAGES_DIR) / "refinement.png"
    logging.info(f"Saving refinement result to {ref_path}")
    save_homography_result(rec_frames.array[optimal_idx], best_specs, ref_path)

    out_meta = const.META_DIR / f"match-{args.input_video.stem}.json"
    logging.info(f"Saving metadata to {out_meta}")

    with open(out_meta, "w") as f:
        json.dump(
            {
                "homographies": [spec.inv_homography.tolist() for spec in best_specs],
                "temporal_offset": int(temporal_offset),
            },
            f,
        )
