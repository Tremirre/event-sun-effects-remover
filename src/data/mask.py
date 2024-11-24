import cv2
import numpy as np


class DilatingMaskGenerator:
    def __init__(
        self,
        max_centers: int,
        min_points: int = 20,
        max_points: int = 50,
        min_dil: int = 20,
        max_dil: int = 30,
        padding: int = 100,
    ) -> None:
        self.max_centers = max_centers
        self.min_points = min_points
        self.max_points = max_points
        self.min_dil = min_dil
        self.max_dil = max_dil
        self.padding = padding

    def __call__(self, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        for _ in range(np.random.randint(1, self.max_centers + 1)):
            x, y = np.random.randint(self.padding, width - self.padding, size=2)
            c1, c2 = np.random.randint(100, 1000, size=2)
            cov = np.array([[c1, 0], [0, c2]])
            points = np.random.randint(self.min_points, self.max_points)
            mvnorm = np.random.multivariate_normal([x, y], cov, points)
            mvnorm = mvnorm.astype(np.int32)
            mvnorm = np.clip(mvnorm, 0, [width - 1, height - 1])
            mask[mvnorm[:, 1], mvnorm[:, 0]] = 255

        num_dils = np.random.randint(self.min_dil, self.max_dil)
        for _ in range(num_dils):
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        return mask
