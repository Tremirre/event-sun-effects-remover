import cv2
import numpy as np

from . import mask


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
    def __init__(self, min_width: int, max_width: int):
        self.min_width = min_width
        self.max_width = max_width
        self.generator = mask.DilatingMaskGenerator(5)

    def __call__(self, bgr: np.ndarray, event_mask: np.ndarray, _) -> np.ndarray:
        height, width = bgr.shape[:2]
        mask = self.generator(height, width)
        mask[event_mask == 0] = 0
        return mask


class FullMasker:
    def __call__(self, bgr: np.ndarray, event_mask: np.ndarray, _) -> np.ndarray:
        return event_mask


class DiffIntensityMasker:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(
        self, bgr: np.ndarray, event_mask: np.ndarray, event: np.ndarray
    ) -> np.ndarray:
        hsl = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
        brightness = hsl[:, :, 1]
        diff = cv2.absdiff(brightness, event)
        diff[event_mask == 0] = 0
        diff_mask = (diff > self.threshold).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)
        diff_mask = cv2.erode(diff_mask, kernel, iterations=2)
        diff_mask = cv2.dilate(diff_mask, kernel, iterations=25)
        diff_mask[event_mask == 0] = 0
        return diff_mask
