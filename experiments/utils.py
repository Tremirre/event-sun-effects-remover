from __future__ import annotations

import dataclasses
import pathlib

import cv2
import numpy as np
import tqdm


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


def read_video(
    path: pathlib.Path, verbose: bool = True
) -> tuple[np.ndarray, np.ndarray]:
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
    return np.array(frames), timestamps


def crop_to_size(video: np.ndarray, width: int, height: int) -> np.ndarray:
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
        self.event_index = 0
        self.count_index = 0
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
