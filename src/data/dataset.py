import logging
import pathlib
import typing

import numpy as np
import torch
import torch.utils.data

logger = logging.getLogger(__name__)


class Transform(typing.Protocol):
    def __call__(
        self,
        img: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor: ...


class Augmenter(typing.Protocol):
    def __call__(
        self,
        img: np.ndarray,
        idx: int,
    ) -> np.ndarray: ...


class BGREADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_paths: list[pathlib.Path],
        transform: Transform,
        augmenter: Augmenter | None = None,
    ) -> None:
        super().__init__()
        self.img_paths = img_paths
        self.transform = transform
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # x -> BGR + Event Reconstruction + Light Augmentation Mask
        # y -> artifact free image
        img = np.load(self.img_paths[idx])

        # img will be 4 channel (if event mask is implicitly the whole image) or 5 channel
        if img.shape[2] == 4:
            img = np.concatenate([img, np.ones_like(img[:, :, :1]) * 255], axis=-1)

        assert img.shape[2] == 5, "Expected 5 channels"
        bgr = img[:, :, :3].copy()
        if self.augmenter is not None:
            img = self.augmenter(img, idx)

        x = self.transform(img)
        y = self.transform(bgr)
        x = typing.cast(torch.Tensor, x)
        y = typing.cast(torch.Tensor, y)
        return x, y
