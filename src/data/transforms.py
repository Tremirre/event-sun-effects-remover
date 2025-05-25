import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class EventTransform:
    def process_event(
        self, event_img: np.ndarray, event_mask: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if img.shape[-1] < 5:
            return img
        event_img = self.process_event(img[:, :, 3], img[:, :, 4])
        img[:, :, 3] = event_img
        return img


class RandomizedEventBrightnessScaler(EventTransform):
    def __init__(self, min_factor: float, max_factor: float):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def process_event(
        self, event_img: np.ndarray, event_mask: np.ndarray
    ) -> np.ndarray:
        factor = np.random.uniform(self.min_factor, self.max_factor)
        return np.clip(255 * (event_img / 255) ** factor, 0, 255).astype(np.uint8)


class RandomizedEventContrastScaler(EventTransform):
    def __init__(self, min_factor: float, max_factor: float):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def process_event(
        self, event_img: np.ndarray, event_mask: np.ndarray
    ) -> np.ndarray:
        factor = np.random.uniform(self.min_factor, self.max_factor)
        min_gs, max_gs = event_img.min(), event_img.max()
        mean_gs = min_gs / 2 + max_gs / 2
        event_img = mean_gs + (event_img - mean_gs) * factor
        return np.clip(event_img, 0, 255).astype(np.uint8)


class RandomizedEventSunAdder(EventTransform):
    def __init__(self, prob: float, min_radius: int = 3, max_radius: int = 5):
        self.prob = prob
        self.min_radius = min_radius
        self.max_radius = max_radius
        assert 0 <= prob <= 1, "Probability must be between 0 and 1"

    def process_event(
        self, event_img: np.ndarray, event_mask: np.ndarray
    ) -> np.ndarray:
        if np.random.rand() > self.prob:
            return event_img

        mask = event_mask.copy()
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        candidates = np.argwhere(mask > 0)
        if len(candidates) == 0:
            return event_img

        back_strength = np.random.randint(20, 51)
        center = tuple(candidates[np.random.randint(len(candidates))])[::-1]
        radius = np.random.randint(self.min_radius, self.max_radius + 1)
        dec_canvas = np.zeros_like(event_img)
        dec_canvas = cv2.circle(dec_canvas, center, 16 * radius, back_strength, -1)  # type: ignore
        dec_canvas = cv2.circle(dec_canvas, center, 4 * radius, 0, -1)  # type: ignore
        dec_canvas = cv2.GaussianBlur(dec_canvas, (31, 31), 15)

        add_canvas = np.zeros_like(event_img)
        add_canvas = cv2.circle(add_canvas, center, radius, 255, -1)  # type: ignore
        add_canvas = cv2.GaussianBlur(add_canvas, (5, 5), 5)

        event_img = cv2.subtract(event_img, dec_canvas)
        event_img = cv2.add(event_img, add_canvas)
        event_img = np.clip(event_img, 0, 255).astype(np.uint8)
        return event_img


class EventMaskChannelRemover:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if img.shape[-1] < 5:
            return img
        return img[:, :, :4]


class RadnomizedGaussianBlur:
    def __init__(self, min_sigma: float, max_sigma: float):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if img.shape[-1] == 3:
            return img
        sigma = np.random.uniform(self.min_sigma, self.max_sigma)
        bgr_blurred = cv2.GaussianBlur(img[:, :, :3], (5, 5), sigma)
        img[:, :, :3] = bgr_blurred
        return img


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


class BGRColorJitter:
    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

        assert all(0 <= factor <= 1 for factor in (brightness, contrast, saturation)), (
            "Factors must be between 0 and 1"
        )

        self.brightness_factor = 1.0
        self.contrast_factor = 1.0
        self.saturation_factor = 1.0

    def is_enabled(self) -> bool:
        return any(
            factor > 0 for factor in (self.brightness, self.contrast, self.saturation)
        )

    def reset(self) -> None:
        self.brightness_factor = 1.0 + np.random.uniform(
            -self.brightness, self.brightness
        )
        self.contrast_factor = 1.0 + np.random.uniform(-self.contrast, self.contrast)
        self.saturation_factor = 1.0 + np.random.uniform(
            -self.saturation, self.saturation
        )

    def __call__(self, img: np.ndarray) -> np.ndarray:
        assert isinstance(img, np.ndarray), "Input must be a NumPy array"
        assert img.ndim >= 3 and img.shape[-1] >= 3, (
            "Image must have at least 3 channels"
        )
        if not self.is_enabled():
            return img

        img = img.astype(np.float32)

        bgr = img[..., :3]
        other = img[..., 3:] if img.shape[-1] > 3 else None

        # Brightness
        if self.brightness > 0:
            bgr = bgr * self.brightness_factor

        # Contrast
        if self.contrast > 0:
            mean = bgr.mean(axis=(0, 1), keepdims=True)
            bgr = (bgr - mean) * self.contrast_factor + mean

        # Saturation (via conversion to HSV)
        if self.saturation > 0:
            bgr_uint8 = np.clip(bgr, 0, 255).astype(np.uint8)
            hsv = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] *= self.saturation_factor
            hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
            bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(
                np.float32
            )

        bgr = np.clip(bgr, 0, 255)

        if other is not None:
            img = np.concatenate([bgr, other], axis=-1)
        else:
            img = bgr

        return img.astype(np.uint8)
