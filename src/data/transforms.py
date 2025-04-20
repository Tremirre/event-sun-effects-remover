import logging
import pathlib

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class EventTransform:
    def process_event(
        self, event_img: np.ndarray, event_mask: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if img.shape[-1] == 3:
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
        if img.shape[-1] == 3:
            return img
        assert img.shape[-1] == 6, (
            "Expected 6 channels (BGR + event + event mask + artifact mask)"
        )
        return np.concatenate([img[:, :, :4], img[:, :, 5:6]], axis=-1)


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


def rand_light_color() -> tuple[int, ...]:
    return tuple(int(v) for v in np.random.randint(150, 256, 3))


def blur(img: np.ndarray, strength: int) -> np.ndarray:
    return cv2.GaussianBlur(img, (strength * 2 + 1, strength * 2 + 1), strength)


class BaseLightArtifactAugmenter:
    def generate_map(self, shape: tuple[int, int, int]) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, img: np.ndarray) -> np.ndarray:
        aug_map = self.generate_map(img[:, :, :3].shape)
        img[:, :, :3] = cv2.add(img[:, :, :3], aug_map)  # type: ignore
        aug_map = cv2.cvtColor(aug_map, cv2.COLOR_BGR2GRAY)
        img[:, :, -1] = cv2.add(img[:, :, -1], aug_map)
        return img


class VeilingGlareAdder(BaseLightArtifactAugmenter):
    def __init__(
        self,
        min_radius: int,
        max_radius: int,
        min_blur: int,
        max_blur: int,
        p_round: float,
    ):
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_blur = min_blur
        self.max_blur = max_blur
        self.p_round = p_round

    def generate_map(self, shape: tuple[int, int, int]) -> np.ndarray:
        g_map = np.zeros(shape=shape, dtype=np.uint8)
        # add ellipse
        pos = (np.random.randint(0, shape[1]), np.random.randint(0, shape[0]))
        if np.random.rand() < self.p_round:
            radius = np.random.randint(self.min_radius, self.max_radius)
            axes = (radius, radius)
        else:
            axes = np.random.randint(self.min_radius, self.max_radius, 2)
        angle = np.random.randint(0, 360)
        color = rand_light_color()
        cv2.ellipse(g_map, pos, axes, angle, 0, 360, color, -1)  # type: ignore
        blur_strength = np.random.randint(self.min_blur, self.max_blur)
        g_map = blur(g_map, blur_strength)
        return g_map


class LensFlareAdder(BaseLightArtifactAugmenter):
    def __init__(
        self,
        min_objects: int,
        max_objects: int,
        min_size: int,
        max_size: int,
        min_strength: float,
        max_strength: float,
        p_add_line: float,
    ):
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.p_add_line = p_add_line
        self.min_size = min_size
        self.max_size = max_size
        self.min_strength = min_strength
        self.max_strength = max_strength

    def _add_donut(
        self, img: np.ndarray, pos: tuple[int, int], size: int
    ) -> np.ndarray:
        cv2.circle(
            img,
            pos,
            size,
            rand_light_color(),
            np.random.randint(1, 5),
        )
        return img

    def _add_circle(
        self, img: np.ndarray, pos: tuple[int, int], size: int
    ) -> np.ndarray:
        cv2.circle(img, pos, size, rand_light_color(), -1)
        return img

    def _add_poly(self, img: np.ndarray, pos: tuple[int, int], size: int) -> np.ndarray:
        num_sides = np.random.randint(4, 8)
        points = []
        for j in range(num_sides):
            angle = j * (2 * np.pi / num_sides)
            x = pos[0] + int(size * np.cos(angle))
            y = pos[1] + int(size * np.sin(angle))
            points.append((x, y))

        cv2.fillPoly(img, [np.array(points)], rand_light_color())
        return img

    def _add_line(
        self, img: np.ndarray, line_start: tuple[int, int], line_end: tuple[int, int]
    ) -> np.ndarray:
        cv2.line(
            img, line_start, line_end, rand_light_color(), np.random.randint(1, 10)
        )
        img = blur(img, np.random.randint(5, 10))

        return img

    def generate_map(self, shape: tuple[int, int, int]) -> np.ndarray:
        flare_map = np.zeros(shape=shape, dtype=np.uint8)
        line_start = (np.random.randint(0, shape[1]), np.random.randint(0, shape[0]))
        line_end = (np.random.randint(0, shape[1]), np.random.randint(0, shape[0]))
        if np.random.rand() < self.p_add_line:
            flare_map = self._add_line(flare_map, line_start, line_end)

        candidate_points_on_line = np.linspace(line_start, line_end, 200)
        num_points = np.random.randint(self.min_objects, self.max_objects)
        np.random.shuffle(candidate_points_on_line)
        candidate_points_on_line = candidate_points_on_line[:num_points].astype(
            np.int32
        )
        candidate_funcs = [self._add_donut, self._add_circle, self._add_poly]
        for point in candidate_points_on_line:
            size = np.random.randint(self.min_size, self.max_size)
            func = np.random.choice(candidate_funcs)  # type: ignore
            flare_map = func(flare_map, point, size)
        flare_map = (blur(flare_map, np.random.randint(1, 5)) * 0.6).astype(np.uint8)
        return flare_map


class SunAdder(BaseLightArtifactAugmenter):
    def __init__(
        self,
        min_size: int,
        max_size: int,
        min_outer_strength: float,
        max_outer_strength: float,
        p_rays: float,
        p_outer: float,
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.min_outer_strength = min_outer_strength
        self.max_outer_strength = max_outer_strength
        self.p_rays = p_rays
        self.p_outer = p_outer

    def _add_rays(
        self, img: np.ndarray, pos: tuple[int, int], color: tuple[int, int, int]
    ) -> np.ndarray:
        ray_map = np.zeros_like(img)
        num_rays = np.random.randint(2, 30)
        for _ in range(num_rays):
            angle = np.random.uniform(0, 2 * np.pi)
            ray_length = np.random.randint(50, 250)
            ray_end = (
                pos[0] + int(ray_length * np.cos(angle)),
                pos[1] + int(ray_length * np.sin(angle)),
            )
            cv2.line(ray_map, pos, ray_end, color, np.random.randint(1, 5))

        ray_map = blur(ray_map, 5)
        strong_ray_radius = np.random.randint(50, 200)
        multiplier = np.zeros_like(img, dtype=np.float32)
        cv2.circle(multiplier, pos, strong_ray_radius, (1.0, 1.0, 1.0), -1)
        multiplier = blur(multiplier, 50)
        ray_map = (ray_map * multiplier).astype(np.uint8)
        img = cv2.add(img, ray_map)
        return img

    def generate_map(self, shape: tuple[int, int, int]) -> np.ndarray:
        s_map = np.zeros(shape=shape, dtype=np.uint8)
        pos = (np.random.randint(0, shape[1]), np.random.randint(0, shape[0]))
        size = np.random.randint(self.min_size, self.max_size)
        outer_radius = np.random.randint(10 * size, 16 * size)
        inner_radius = np.random.randint(2 * size, 4 * size)
        color = rand_light_color()
        if np.random.rand() < self.p_outer:
            cv2.circle(s_map, pos, outer_radius, color, -1)
            outer_strength = np.random.uniform(
                self.min_outer_strength, self.max_outer_strength
            )
            s_map = (s_map * outer_strength).astype(np.uint8)
            s_map = blur(s_map, 50)

        cv2.circle(s_map, pos, inner_radius, color, -1)
        s_map = blur(s_map, 20)

        cv2.circle(s_map, pos, size, (255, 255, 255), -1)

        if np.random.rand() < self.p_rays:
            s_map = self._add_rays(s_map, pos, color)
        s_map = blur(s_map, 5)

        return s_map


class HQFlareBasedAugmenter(BaseLightArtifactAugmenter):
    def __init__(self, flare_imgs: list[pathlib.Path]):
        super().__init__()
        self.flare_imgs = flare_imgs

    def _get_flare(self, shape: tuple[int, int, int]) -> np.ndarray:
        idx = np.random.randint(0, len(self.flare_imgs))
        flare = cv2.imread(str(self.flare_imgs[idx]), cv2.IMREAD_UNCHANGED)
        flare = cv2.resize(flare, (shape[1], shape[0]))
        # TODO: maybe scale?

        width, height = shape[1], shape[0]

        dx, dy = (
            np.random.randint(-width // 4, width // 4),
            np.random.randint(-height // 4, height // 4),
        )
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        flare = cv2.warpAffine(flare, translation_matrix, (width, height))

        angle = np.random.randint(0, 360)
        rot_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        flare = cv2.warpAffine(flare, rot_matrix, (width, height))

        return flare

    def generate_map(self, shape):
        n_flares = np.random.randint(1, 5)
        flare_map = np.zeros(shape=shape, dtype=np.uint8)
        for _ in range(n_flares):
            flare = self._get_flare(shape)
            flare_map = cv2.add(flare_map, flare)
        return flare_map


class CompositeLightArtifactAugmenter:
    def __init__(
        self,
        augmenters: list[BaseLightArtifactAugmenter],
        probs: list[float],
        fix_by_idx: bool = False,
        target_binarization: bool = False,
    ):
        assert len(augmenters) == len(probs), (
            "Augmenters and probs must have same length"
        )
        self.augmenters = augmenters
        self.probs = probs
        self.target_binarization = target_binarization
        self.fix_by_idx = fix_by_idx
        self.cache = {}

    def __call__(self, img: np.ndarray, idx: int) -> np.ndarray:
        if self.fix_by_idx and idx in self.cache:
            return self.cache[idx]

        np.random.seed(idx)
        for augmenter, prob in zip(self.augmenters, self.probs, strict=True):
            if np.random.rand() < prob:
                img = augmenter(img)
        if self.target_binarization:
            img[:, :, -1] = np.where(img[:, :, -1] > 0, 255, 0)
        return img
