from __future__ import annotations

import dataclasses
import json
import pathlib
import typing

import cv2
import numpy as np
import rpg_e2vid.utils.inference_utils as iu
import torch
import tqdm
import yaml
from scipy.spatial.transform import Rotation as Rot

import const


class Transform:
    def __init__(self, translation: np.ndarray, rotation: Rot):
        if translation.ndim > 1:
            self._translation = translation.flatten()
        else:
            self._translation = translation
        assert self._translation.size == 3
        self._rotation = rotation

    @staticmethod
    def from_transform_matrix(transform_matrix: np.ndarray):
        translation = transform_matrix[:3, 3]
        rotation = Rot.from_matrix(transform_matrix[:3, :3])
        return Transform(translation, rotation)

    @staticmethod
    def from_rotation(rotation: Rot):
        return Transform(np.zeros(3), rotation)

    def R_matrix(self):
        return self._rotation.as_matrix()

    def R(self):
        return self._rotation

    def t(self):
        return self._translation

    def T_matrix(self) -> np.ndarray:
        return self._T_matrix_from_tR(self._translation, self._rotation.as_matrix())

    def q(self):
        # returns (x, y, z, w)
        return self._rotation.as_quat()

    def euler(self):
        return self._rotation.as_euler("xyz", degrees=True)

    def __matmul__(self, other):
        # a (self), b (other)
        # returns a @ b
        #
        # R_A | t_A   R_B | t_B   R_A @ R_B | R_A @ t_B + t_A
        # --------- @ --------- = ---------------------------
        # 0   | 1     0   | 1     0         | 1
        #
        rotation = self._rotation * other._rotation
        translation = self._rotation.apply(other._translation) + self._translation
        return Transform(translation, rotation)

    def inverse(self):
        #           R_AB  | A_t_AB
        # T_AB =    ------|-------
        #           0     | 1
        #
        # to be converted to
        #
        #           R_BA  | B_t_BA    R_AB.T | -R_AB.T @ A_t_AB
        # T_BA =    ------|------- =  -------|-----------------
        #           0     | 1         0      | 1
        #
        # This is numerically more stable than matrix inversion of T_AB
        rotation = self._rotation.inv()
        translation = -rotation.apply(self._translation)
        return Transform(translation, rotation)


def get_mapping_from_calib(calib: dict) -> np.ndarray:
    K_r0 = np.eye(3)
    K_r0[[0, 1, 0, 1], [0, 1, 2, 2]] = calib["intrinsics"]["camRect0"]["camera_matrix"]
    K_r1 = np.eye(3)
    K_r1[[0, 1, 0, 1], [0, 1, 2, 2]] = calib["intrinsics"]["camRect1"]["camera_matrix"]

    R_r0_0 = Rot.from_matrix(np.array(calib["extrinsics"]["R_rect0"]))
    R_r1_1 = Rot.from_matrix(np.array(calib["extrinsics"]["R_rect1"]))

    T_r0_0 = Transform.from_rotation(R_r0_0)
    T_r1_1 = Transform.from_rotation(R_r1_1)
    T_1_0 = Transform.from_transform_matrix(np.array(calib["extrinsics"]["T_10"]))

    T_r1_r0 = T_r1_1 @ T_1_0 @ T_r0_0.inverse()
    R_r1_r0_matrix = T_r1_r0.R().as_matrix()
    P_r1_r0 = K_r1 @ R_r1_r0_matrix @ np.linalg.inv(K_r0)

    ht = 480
    wd = 640
    # coords: ht, wd, 2
    coords = np.stack(np.meshgrid(np.arange(wd), np.arange(ht)), axis=-1)
    # coords_hom: ht, wd, 3
    coords_hom = np.concatenate((coords, np.ones((ht, wd, 1))), axis=-1)
    # mapping: ht, wd, 3
    mapping = (P_r1_r0 @ coords_hom[..., None]).squeeze()
    # mapping: ht, wd, 2
    mapping = (mapping / mapping[..., -1][..., None])[..., :2]
    mapping = mapping.astype("float32")
    return mapping


@dataclasses.dataclass
class EventsData:
    array: np.ndarray
    width: int
    height: int

    @classmethod
    def from_path(cls, path: pathlib.Path | str) -> EventsData:
        with open(path, "rb") as f:
            events = np.fromfile(f, dtype=np.uint8).reshape(-1, 8)

        meta_entry = events[-1]
        events = events[:-1]

        width = meta_entry[:2].view(dtype=np.uint16)[0]
        height = meta_entry[2:4].view(dtype=np.uint16)[0]
        n_events = meta_entry[4:].view(dtype=np.uint32)[0]
        assert n_events == len(events), f"Expected {n_events} events, got {len(events)}"

        ts_data = np.hstack(
            [events[:, :3], np.zeros((n_events, 1), dtype=np.uint8)]
        ).view(dtype=np.uint32)
        xs_data = events[:, 3:5].flatten().view(dtype=np.uint16).reshape(-1, 1)
        ys_data = events[:, 5:7].flatten().view(dtype=np.uint16).reshape(-1, 1)

        events = np.hstack(
            [ts_data, xs_data, ys_data, events[:, 7].reshape(-1, 1)]
        ).astype(np.float32)
        return cls(events, width, height)

    @property
    def num_events(self) -> int:
        return self.array.shape[0]


class EventWindowIterator:
    def __init__(
        self,
        events: np.ndarray,
        counts: np.ndarray,
        window_length: int,
        stride: int = 1,
        offset: int = 0,
    ) -> None:
        self.events = events
        self.counts = counts
        self.event_index = counts[:offset].sum()
        self.count_index = offset
        self.window_length = window_length
        self.stride = stride
        self.offset = offset

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        window_start = self.offset + self.count_index
        if window_start >= self.counts.shape[0]:
            raise StopIteration

        window_end = window_start + self.window_length
        total_counts = self.counts[window_start:window_end].sum()
        window = self.events[self.event_index : self.event_index + total_counts]
        stride_counts = self.counts[window_start : window_start + self.stride].sum()
        self.event_index += stride_counts
        self.count_index += self.stride
        return window

    def __len__(self) -> int:
        res, rem = divmod(self.counts.shape[0] - self.offset, self.stride)
        return res + bool(rem)


@dataclasses.dataclass
class Frames:
    array: np.ndarray
    timestamps: np.ndarray

    def __post_init__(self):
        assert (
            self.array.shape[0] == self.timestamps.shape[0]
        ), "Array and timestamps should have the same length"

    @property
    def num_frames(self) -> int:
        return self.array.shape[0]

    @property
    def height(self) -> int:
        return self.array.shape[1]

    @property
    def width(self) -> int:
        return self.array.shape[2]

    @classmethod
    def from_paths(
        cls,
        img_paths: list[pathlib.Path],
        ts_path: pathlib.Path,
        calib_path: pathlib.Path,
    ) -> Frames:
        timestamps = np.loadtxt(ts_path).astype(int)

        with open(calib_path) as f:
            calib = yaml.safe_load(f)
        mapping = get_mapping_from_calib(calib)
        frames = []
        for path in tqdm.tqdm(img_paths, desc="Reading images"):
            assert path.exists(), f"Image {path} does not exist"
            frame = cv2.imread(str(path))
            frame = cv2.remap(frame, mapping, None, cv2.INTER_CUBIC)
            frames.append(frame)
        return cls(np.array(frames), timestamps)


@dataclasses.dataclass
class AlignMeta:
    homographies: np.ndarray
    offset_ms: int

    @classmethod
    def from_json(cls, path: pathlib.Path) -> AlignMeta:
        data = json.loads(path.read_text())
        homographies = np.array(data["homographies"])
        assert len(homographies.shape) == 3, "There should be a list of homographies"
        assert (
            homographies.shape[1] == 3 and homographies.shape[2] == 3
        ), "Invalid shape"
        offset_ms = data["temporal_offset"]
        return cls(homographies, offset_ms)

    def get_common_mask(self, width: int, height: int) -> np.ndarray:
        common_mask = np.zeros((height, width), np.uint8)
        masks = make_horiz_masks(len(self.homographies), width, height)
        for mask, hom in zip(masks, self.homographies):
            full_img = np.ones((height, width), np.uint8) * 255
            full_img = cv2.warpPerspective(full_img, hom, (width, height))
            common_mask[mask > 0] = full_img[mask > 0]
        return common_mask


def read_video(path: pathlib.Path, verbose: bool = True) -> Frames:
    cap = cv2.VideoCapture(str(path))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    iter_frames = range(num_frames)
    timestamps = []
    if verbose:
        iter_frames = tqdm.tqdm(iter_frames, total=num_frames, desc=f"Reading {path}")
    frames = []
    for _ in iter_frames:
        ret, frame = cap.read()
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        if not ret:
            break
        frames.append(frame)
    cap.release()
    timestamps = np.array(timestamps).astype(int)
    frames = np.array(frames)
    return Frames(frames, timestamps)


def crop_vid_to_size(video: np.ndarray, width: int, height: int) -> np.ndarray:
    v_height, v_width = video.shape[1:3]
    assert v_width >= width, f"Video width {v_width} < {width}"
    assert v_height >= height, f"Video height {v_height} < {height}"

    target_prop = width / height
    assert target_prop > 1, "Width should be greater than height"
    target_width = int(target_prop * v_height)

    width_margin = int(v_width - target_width) // 2
    video = video[:, :, width_margin : width_margin + target_width]

    resized = [
        cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        for frame in video
    ]
    return np.array(resized)


def crop_img_to_size(img: np.ndarray, width: int, height: int) -> np.ndarray:
    v_height, v_width = img.shape[:2]
    assert v_width >= width, f"Image width {v_width} < {width}"
    assert v_height >= height, f"Image height {v_height} < {height}"

    target_prop = width / height
    assert target_prop > 1, "Width should be greater than height"
    target_width = int(target_prop * v_height)

    width_margin = int(v_width - target_width) // 2
    img = img[:, width_margin : width_margin + target_width]

    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


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


def reconstruct_video(
    event_it: typing.Iterator[np.ndarray],
    width: int,
    height: int,
    model: torch.nn.Module,
    sharpen: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    if verbose:
        event_it = tqdm.tqdm(event_it, total=len(event_it), desc="Reconstructing video")
    rec = []
    prev = None
    for window in event_it:
        vg = iu.events_to_voxel_grid(window, 5, width, height)
        vg = torch.from_numpy(vg).unsqueeze(0).float().to(const.DEVICE)
        with torch.no_grad():
            pred, prev = model(vg, prev)
            pred = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
            pred = cv2.undistort(pred, const.EVENT_MTX, const.EVENT_DIST)
            # sharpen
            if sharpen:
                pred_gb = cv2.GaussianBlur(pred, (0, 0), 3)
                pred = cv2.addWeighted(pred, 1.5, pred_gb, -0.5, 0)

        rec.append(pred)
    return np.array(rec)
