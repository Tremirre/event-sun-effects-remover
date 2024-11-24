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
        self.generators = [
            mask.MaskGenerator(
                target_channels=1,
                degree=degree,
                min_width=min_width,
                max_width=max_width,
            )
            for degree in mask.Degree
        ]

    def __call__(self, bgr: np.ndarray, event_mask: np.ndarray) -> np.ndarray:
        height, width = bgr.shape[:2]
        generator = np.random.choice(self.generators)  # type: ignore
        mask = generator(height, width)[0]
        mask[event_mask == 0] = 0
        return mask


class FullMasker:
    def __call__(self, bgr: np.ndarray, event_mask: np.ndarray) -> np.ndarray:
        return event_mask
