import logging
import pathlib
import typing

import cv2
import numpy as np
import skimage
import torch
import torch.utils.data

from .artifacts import ArtifactSource

logger = logging.getLogger(__name__)


class Transform(typing.Protocol):
    def __call__(
        self,
        img: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor: ...


class BGREADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_paths: list[pathlib.Path],
        transform: Transform,
        artifact_source: ArtifactSource | None = None,
    ) -> None:
        super().__init__()
        self.img_paths = img_paths
        self.transform = transform
        self.artifact_source = artifact_source or self.no_source

    @staticmethod
    def no_source(img: np.ndarray) -> np.ndarray:
        target_shape = img.shape[:2] + (3,)
        return np.zeros(target_shape, dtype=np.uint8)

    @staticmethod
    def fill_event(img: np.ndarray) -> np.ndarray:
        if img.shape[-1] < 5:  # no event mask
            return img
        mask = img[:, :, 4]
        if (mask > 0).all():
            return img
        rec = img[:, :, 3]
        bgr = img[:, :, :3]
        gs = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        height, width = gs.shape[:2]
        gs_equated = (
            skimage.exposure.match_histograms(gs.flatten(), rec[mask > 0])
            .astype(np.uint8)
            .reshape(height, width)
        )
        mask_eroded = cv2.erode(
            mask.copy(), np.ones((3, 3), dtype=np.uint8), iterations=8
        )
        mask_softened = (
            cv2.GaussianBlur(mask_eroded, (0, 0), 4).astype(np.float32) / 255.0
        )

        img[:, :, 3] = mask_softened * rec + (1 - mask_softened) * gs_equated
        return img

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # x -> BGR + Event Reconstruction
        # y -> BGR + Artifact Map
        img = np.load(self.img_paths[idx])
        img = self.fill_event(img)
        out = img[:, :, :3].copy()
        art_map = self.artifact_source(img)
        img[:, :, :3] = cv2.add(img[:, :, :3], art_map)
        art_map_gs = cv2.cvtColor(art_map, cv2.COLOR_BGR2GRAY)
        out = np.concatenate([out, art_map_gs[:, :, np.newaxis]], axis=-1)

        img = img[:, :, :5]
        x = self.transform(img)
        y = self.transform(out)
        x = typing.cast(torch.Tensor, x)
        y = typing.cast(torch.Tensor, y)
        assert x.shape[0] == 4, "x should have 4 channels"
        assert y.shape[0] == 4, "y should have 4 channels"
        return x, y
