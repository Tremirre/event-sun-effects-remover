import logging
import pathlib
import typing

import cv2
import numpy as np
import torch
import torch.utils.data

logger = logging.getLogger(__name__)


class Masker(typing.Protocol):
    def __call__(
        self, idx: int, bgr: np.ndarray, event_mask: np.ndarray, event: np.ndarray
    ) -> np.ndarray: ...

    def progress(self) -> None: ...


class Transform(typing.Protocol):
    def __call__(
        self,
        img: np.ndarray | torch.Tensor,
    ) -> torch.Tensor: ...


class LightAugmenter(typing.Protocol):
    def __call__(
        self,
        img: np.ndarray,
        idx: int,
    ) -> np.ndarray: ...


class BGREMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_paths: list[pathlib.Path],
        masker: Masker,
        bgr_transform: None | Transform = None,
        event_transform: None | Transform = None,
        masked_bgr_transform: None | Transform = None,
        separate_event_channel: bool = True,
        mask_progression: bool = False,
        blur_factor: int = 0,
        yuv_interpolation: bool = False,
    ) -> None:
        self.img_paths = img_paths
        self.masker = masker
        self.bgr_transform = bgr_transform
        self.event_transform = event_transform
        self.masked_bgr_transform = masked_bgr_transform
        self.separate_event_channel = separate_event_channel
        self.mask_progression = mask_progression
        self.blur_factor = blur_factor
        self.yuv_interpolation = yuv_interpolation
        self.blur_kernel = (2 * blur_factor + 1, 2 * blur_factor + 1)

        # if self.yuv_interpolation:
        #     assert not self.separate_event_channel, (
        #         "Cannot interpolate YUV and have separate event channel"
        #     )

    def __len__(self) -> int:
        return len(self.img_paths)

    def mask_out_image(
        self, mask_expanded: np.ndarray, bgr: np.ndarray, event_expanded: np.ndarray
    ) -> np.ndarray:
        if self.separate_event_channel and not self.yuv_interpolation:
            bgr = bgr.astype(np.float32) / 255.0
            return (1 - mask_expanded) * bgr * 255.0
        if not self.yuv_interpolation:
            bgr = bgr.astype(np.float32) / 255.0
            return ((1 - mask_expanded) * bgr + mask_expanded * event_expanded) * 255.0
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
        yuv = yuv.astype(np.float32) / 255.0
        if self.separate_event_channel:
            yuv[:, :, 0] = (1 - mask_expanded[:, :, 0]) * yuv[:, :, 0] + mask_expanded[
                :, :, 0
            ] * np.zeros_like(yuv[:, :, 0])
        else:
            yuv[:, :, 0] = (1 - mask_expanded[:, :, 0]) * yuv[:, :, 0] + mask_expanded[
                :, :, 0
            ] * event_expanded[:, :, 0]
        return cv2.cvtColor((yuv * 255.0).astype(np.uint8), cv2.COLOR_YUV2BGR)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = np.load(self.img_paths[idx])
        if img.shape[2] == 4:
            img = np.concatenate([img, np.ones_like(img[:, :, :1]) * 255], axis=-1)

        assert img.shape[2] == 5, "Expected 5 channels"
        bgr = img[:, :, :3]
        target = bgr.copy()
        event = img[:, :, 3:4]
        event_mask = img[:, :, 4]
        if self.blur_factor > 0:
            # erode event mask to prevent masking out too much
            event_mask = cv2.erode(event_mask, np.ones(self.blur_kernel, np.uint8))
        mask = self.masker(idx, bgr, event_mask, event)
        if self.event_transform:
            event = np.dstack([event, mask])
            event = self.event_transform(event)
            event = event[:, :, 0:1]

        if self.masked_bgr_transform:
            bgrm = np.dstack([bgr, mask])
            bgrm = self.masked_bgr_transform(bgrm)
            bgr = bgrm[:, :, :3]

        mask = mask.astype(np.float32) / 255.0
        if self.blur_factor > 0:
            # blur mask
            mask = cv2.GaussianBlur(mask, self.blur_kernel, self.blur_factor)

        mask = np.expand_dims(mask, axis=-1)

        mask_expanded = np.repeat(mask, 3, axis=-1)
        event_expanded = np.repeat(event, 3, axis=-1).astype(np.float32) / 255.0

        bgr = self.mask_out_image(mask_expanded, bgr, event_expanded).astype(np.uint8)
        event_expanded = (event_expanded * 255.0).astype(np.uint8)

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


class BGRArtifcatDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_paths: list[pathlib.Path],
        transform: Transform,
        augmenter: LightAugmenter,
        binarize: bool = False,
        test_mode: bool = False,
    ) -> None:
        self.img_paths = img_paths
        self.transform = transform
        self.augmenter = augmenter
        self.test_mode = test_mode
        self.binarize = binarize

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = np.load(self.img_paths[idx])
        if self.test_mode:
            assert img.shape[2] == 4, "Expected 4 channels for test mode"
            bgr = img[:, :, :3]
            target = img[:, :, 3]
            bgr = self.transform(bgr)
            target = self.transform(target)
            return bgr, target

        if img.shape[2] > 3:
            img = img[:, :, :3]

        assert img.shape[2] == 3, "Expected 3 channels"
        none_mask = np.zeros_like(img[:, :, 0])
        img = np.dstack([img, none_mask])
        img = self.augmenter(img, idx)
        bgr = img[:, :, :3]
        target = img[:, :, 3]
        if self.binarize:
            target = (target > 0).astype(np.uint8) * 255

        bgr = self.transform(bgr)
        target = self.transform(target)
        return bgr, target
