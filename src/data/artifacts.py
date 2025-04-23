import pathlib

import cv2
import numpy as np


def rand_light_color() -> tuple[int, ...]:
    return tuple(int(v) for v in np.random.randint(150, 256, 3))


def blur(img: np.ndarray, strength: int) -> np.ndarray:
    return cv2.GaussianBlur(img, (strength * 2 + 1, strength * 2 + 1), strength)


class BaseLightArtifactGenerator:
    def __call__(self, shape: tuple[int, int, int]) -> np.ndarray:
        raise NotImplementedError


class VeilingGlareAdder(BaseLightArtifactGenerator):
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

    def __call__(self, shape: tuple[int, int, int]) -> np.ndarray:
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


class LensFlareAdder(BaseLightArtifactGenerator):
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

    def __call__(self, shape: tuple[int, int, int]) -> np.ndarray:
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


class SunAdder(BaseLightArtifactGenerator):
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

    def __call__(self, shape: tuple[int, int, int]) -> np.ndarray:
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


class HQFlareBasedAugmenter(BaseLightArtifactGenerator):
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

    def __call__(self, shape):
        n_flares = np.random.randint(1, 5)
        flare_map = np.zeros(shape=shape, dtype=np.uint8)
        for _ in range(n_flares):
            flare = self._get_flare(shape)
            flare_map = cv2.add(flare_map, flare)
        return flare_map


class NoOpAugmenter(BaseLightArtifactGenerator):
    def __call__(self, shape: tuple[int, int, int]) -> np.ndarray:
        return np.zeros(shape=shape, dtype=np.uint8)


class ArtifactSource:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CompositeLightArtifactGenerator(ArtifactSource):
    def __init__(
        self,
        augmenters: list[BaseLightArtifactGenerator],
        probs: list[float],
        exclusive: bool = False,
    ):
        assert len(augmenters) == len(
            probs
        ), "Augmenters and probs must have same length"
        self.augmenters = augmenters
        self.probs = probs

    def __call__(self, img: np.ndarray) -> np.ndarray:
        artifact_map = np.zeros_like(img[:, :, :3], dtype=np.uint8)
        for augmenter, prob in zip(self.augmenters, self.probs, strict=True):
            if np.random.rand() < prob:
                artifact_map = cv2.add(augmenter(artifact_map.shape), artifact_map)

        return artifact_map


class SingleChoiceArtifactSource(ArtifactSource):
    def __init__(
        self, augmenters: list[BaseLightArtifactGenerator], probs: list[float]
    ):
        assert len(augmenters) == len(
            probs
        ), "Augmenters and probs must have same length"
        self.augmenters = augmenters
        self.probs = probs

    def __call__(self, img: np.ndarray) -> np.ndarray:
        idx = np.random.choice(len(self.augmenters), p=self.probs)
        augmenter = self.augmenters[idx]
        target_shape = img.shape[:2] + (3,)
        return augmenter(target_shape)


class LightArtifactExtractor(ArtifactSource):
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img[:, :, -3:]
