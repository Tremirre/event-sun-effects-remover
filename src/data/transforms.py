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


class RandomizedSunAdder:
    def __init__(self, prob: float, min_radius: int = 3, max_radius: int = 5):
        self.prob = prob
        self.min_radius = min_radius
        self.max_radius = max_radius
        assert 0 <= prob <= 1, "Probability must be between 0 and 1"

    def __call__(self, gs_with_mask: np.ndarray) -> np.ndarray:
        img_gs = gs_with_mask[:, :, 0]
        if np.random.rand() > self.prob:
            return img_gs[:, :, np.newaxis]

        mask = gs_with_mask[:, :, 1]
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        candidates = np.argwhere(mask > 0)
        if len(candidates) == 0:
            return img_gs[:, :, np.newaxis]

        back_strength = np.random.randint(20, 51)
        center = tuple(candidates[np.random.randint(len(candidates))])[::-1]
        radius = np.random.randint(self.min_radius, self.max_radius + 1)
        dec_canvas = np.zeros_like(img_gs)
        dec_canvas = cv2.circle(dec_canvas, center, 16 * radius, back_strength, -1)  # type: ignore
        dec_canvas = cv2.circle(dec_canvas, center, 4 * radius, 0, -1)  # type: ignore
        dec_canvas = cv2.GaussianBlur(dec_canvas, (31, 31), 15)

        add_canvas = np.zeros_like(img_gs)
        add_canvas = cv2.circle(add_canvas, center, radius, 255, -1)  # type: ignore
        add_canvas = cv2.GaussianBlur(add_canvas, (5, 5), 5)

        img_gs = cv2.subtract(img_gs, dec_canvas)
        img_gs = cv2.add(img_gs, add_canvas)
        return img_gs[:, :, np.newaxis]


class RadnomizedGaussianBlur:
    def __init__(self, min_sigma: float, max_sigma: float):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, img: np.ndarray) -> np.ndarray:
        sigma = np.random.uniform(self.min_sigma, self.max_sigma)
        res = cv2.GaussianBlur(img, (5, 5), sigma)
        return res[:, :, np.newaxis]


class RandomizedMasker:
    def __init__(
        self,
        gparams: list[dict] | None = None,
        fix_by_idx: bool = False,
    ):
        if gparams is None:
            gparams = [{"max_centers": 5}]
        if len(gparams) > 1:
            assert not fix_by_idx, "Cannot fix by idx with multiple mask generators"
        self.generator = mask.DilatingMaskGenerator(**gparams.pop(0))
        self.fix_by_idx = fix_by_idx
        self.gparams = gparams
        self.cache = {}

    def progress(self) -> None:
        if len(self.gparams) == 0:
            return
        self.generator = mask.DilatingMaskGenerator(**self.gparams.pop(0))

    def __call__(
        self, idx: int, bgr: np.ndarray, event_mask: np.ndarray, event: np.ndarray
    ) -> np.ndarray:
        height, width = bgr.shape[:2]
        if self.fix_by_idx and idx in self.cache:
            return self.cache[idx]
        mask = self.generator(height, width)
        mask[event_mask == 0] = 0
        if self.fix_by_idx:
            self.cache[idx] = mask
        return mask


class DiffIntensityMasker:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(
        self, idx: int, bgr: np.ndarray, event_mask: np.ndarray, event: np.ndarray
    ) -> np.ndarray:
        hsl = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
        brightness = hsl[:, :, 1:2]
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

    def progress(self) -> None:
        pass
