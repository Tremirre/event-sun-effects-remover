import contextlib

import cv2
import numpy as np

from . import mask


def custom_seed_context(seed: int):
    @contextlib.contextmanager
    def seed_context():
        state = np.random.get_state()
        np.random.seed(seed)
        yield
        np.random.set_state(state)

    return seed_context()


def passthrough_context(*args, **kwargs):
    @contextlib.contextmanager
    def passthrough():
        yield

    return passthrough()


class RandomizedBrightnessScaler:
    def __init__(self, min_factor: float, max_factor: float):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, img_gs: np.ndarray) -> np.ndarray:
        factor = np.random.uniform(self.min_factor, self.max_factor)
        return np.clip(255 * (img_gs / 255) ** factor, 0, 255).astype(np.uint8)


class RandomizedContrastScaler:
    def __init__(self, min_factor: float, max_factor: float):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, img_gs: np.ndarray) -> np.ndarray:
        factor = np.random.uniform(self.min_factor, self.max_factor)
        min_gs, max_gs = img_gs.min(), img_gs.max()
        mean_gs = min_gs / 2 + max_gs / 2
        img_gs = mean_gs + (img_gs - mean_gs) * factor
        return np.clip(img_gs, 0, 255).astype(np.uint8)


class RandomizedMasker:
    def __init__(self, min_width: int, max_width: int, fix_by_idx: bool = False):
        self.min_width = min_width
        self.max_width = max_width
        self.generator = mask.DilatingMaskGenerator(5)
        self.fix_by_idx = fix_by_idx

    def __call__(
        self, idx: int, bgr: np.ndarray, event_mask: np.ndarray, _
    ) -> np.ndarray:
        height, width = bgr.shape[:2]
        ctx = custom_seed_context if not self.fix_by_idx else passthrough_context
        with ctx(idx):
            mask = self.generator(height, width)
            mask[event_mask == 0] = 0
            return mask


class DiffIntensityMasker:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(
        self, idx: int, bgr: np.ndarray, event_mask: np.ndarray, event: np.ndarray
    ) -> np.ndarray:
        hsl = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
        brightness = hsl[:, :, 1]
        diff_signed = brightness.astype(np.int16) - event.astype(np.int16)
        diff_signed[event_mask == 0] = 0
        diff_mask = (diff_signed > self.threshold).astype(np.uint8) * 255
        # close diff mask
        kernel = np.ones((5, 5), np.uint8)
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)
        # erode 2 times
        diff_mask = cv2.erode(diff_mask, kernel, iterations=2)
        # dilate 5 times
        diff_mask = cv2.dilate(diff_mask, kernel, iterations=25)
        diff_mask[event_mask == 0] = 0
        return diff_mask
        return diff_mask
