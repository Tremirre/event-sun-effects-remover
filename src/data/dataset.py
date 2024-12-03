import pathlib
import typing

import numpy as np
import torch
import torch.utils.data


class Masker(typing.Protocol):
    def __call__(
        self, idx: int, bgr: np.ndarray, event_mask: np.ndarray, event: np.ndarray
    ) -> np.ndarray: ...


class Transform(typing.Protocol):
    def __call__(self, img: np.ndarray | torch.Tensor) -> torch.Tensor: ...


class BGREMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_paths: list[pathlib.Path],
        masker: Masker,
        bgr_transform: None | Transform = None,
        event_transform: None | Transform = None,
    ) -> None:
        self.img_paths = img_paths
        self.masker = masker
        self.bgr_transform = bgr_transform
        self.event_transform = event_transform

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = np.load(self.img_paths[idx])
        assert img.shape[2] == 5, "Expected 5 channels"
        bgr = img[:, :, :3]
        event = img[:, :, 3:4]
        event_mask = img[:, :, 4]
        mask = self.masker(idx, bgr, event_mask, event)
        target = bgr.copy()
        mask = np.expand_dims(mask, axis=-1)
        mask_expanded = np.repeat(mask, 3, axis=-1)
        bgr = np.where(mask_expanded, 0, bgr)

        if self.event_transform:
            event = self.event_transform(event)

        if self.bgr_transform:
            bgr = self.bgr_transform(bgr)
            target = self.bgr_transform(target)
            mask = self.bgr_transform(mask)
            event = self.bgr_transform(event)

        bgrem = torch.concatenate([bgr, event, mask], dim=0)
        return bgrem, target
