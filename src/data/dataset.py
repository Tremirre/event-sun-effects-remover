import logging
import pathlib
import typing

import numpy as np
import torch
import torch.utils.data

from src import const

logger = logging.getLogger(__name__)


class Masker(typing.Protocol):
    def __call__(
        self, idx: int, bgr: np.ndarray, event_mask: np.ndarray, event: np.ndarray
    ) -> np.ndarray: ...

    def progress(self) -> None: ...


class Transform(typing.Protocol):
    def __call__(self, img: np.ndarray | torch.Tensor) -> torch.Tensor: ...


class BGREMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_paths: list[pathlib.Path],
        masker: Masker,
        bgr_transform: None | Transform = None,
        event_transform: None | Transform = None,
        separate_event_channel: bool = True,
        mask_progression: bool = False,
    ) -> None:
        self.img_paths = img_paths
        self.masker = masker
        self.bgr_transform = bgr_transform
        self.event_transform = event_transform
        self.separate_event_channel = separate_event_channel
        self.mask_progression = mask_progression
        self._retrieve_count = 0

    def __len__(self) -> int:
        return len(self.img_paths)

    def on_retrieve(self) -> None:
        if not self.mask_progression:
            return
        self._retrieve_count += 1
        if self._retrieve_count % (const.MASK_GROWTH_EVERY_N_EPOCHS * len(self)) == 0:
            logger.info("Progressing mask")
            self.masker.progress()
            self._retrieve_count = 0

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.on_retrieve()
        img = np.load(self.img_paths[idx])
        if img.shape[2] == 4:
            img = np.concatenate([img, np.ones_like(img[:, :, :1]) * 255], axis=-1)

        assert img.shape[2] == 5, "Expected 5 channels"
        bgr = img[:, :, :3]
        event = img[:, :, 3:4]
        event_mask = img[:, :, 4]
        mask = self.masker(idx, bgr, event_mask, event)
        target = bgr.copy()
        mask = np.expand_dims(mask, axis=-1)

        if self.event_transform:
            event = self.event_transform(event)

        mask_expanded = np.repeat(mask, 3, axis=-1)
        event_expanded = np.repeat(event, 3, axis=-1)
        bgr = np.where(
            mask_expanded, 0 if self.separate_event_channel else event_expanded, bgr
        )

        if self.bgr_transform:
            bgr = self.bgr_transform(bgr)
            target = self.bgr_transform(target)
            mask = self.bgr_transform(mask)
            if self.separate_event_channel:
                event = self.bgr_transform(event)
        if self.separate_event_channel:
            bgrem = torch.cat([bgr, event, mask], dim=0)
        else:
            bgrem = torch.concatenate([bgr, mask], dim=0)
        return bgrem, target
