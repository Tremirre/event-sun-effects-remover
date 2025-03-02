import logging
import pathlib

import numpy as np
import pytorch_lightning as pl
import torch.utils.data
import torchvision.transforms as T

from src.data import dataset, transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

DATA_PATTERN = "**/*.npy"


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: pathlib.Path,
        ref_dir: pathlib.Path | None = None,
        train_img_glob: str = DATA_PATTERN,
        val_prob: float = 0.1,
        test_prob: float = 0.1,
        num_workers: int = 0,
        batch_size: int = 32,
        frac_used: float = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.ref_dir = ref_dir
        self.train_img_glob = train_img_glob
        self.ref_paths = sorted(ref_dir.glob(DATA_PATTERN)) if ref_dir else []
        self.img_paths = list(self.data_dir.glob(DATA_PATTERN))
        self.val_prob = val_prob
        self.test_prob = test_prob
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frac_used = frac_used

        np.random.shuffle(self.img_paths)  # type: ignore
        n = len(self.img_paths)
        self.img_paths = self.img_paths[: int(self.frac_used * n)]
        n = len(self.img_paths)
        self.val_n = int(n * self.val_prob)
        self.test_n = int(n * self.test_prob)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.ref_dataset = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        raise NotImplementedError

    def get_dataset_sizes(self) -> dict[str, int]:
        train_files = self.img_paths[self.val_n + self.test_n :]
        train_files = [p for p in train_files if p.match(self.train_img_glob)]
        return {
            "train_size": len(train_files),
            "val_size": self.val_n,
            "test_size": self.test_n,
            "ref_size": len(self.ref_paths),
        }

    def train_dataloader(self):
        assert self.train_dataset is not None, "Train dataset not set up"
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):  # -> DataLoader:
        assert self.val_dataset is not None, "Val dataset not set up"
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        assert self.test_dataset is not None, "Test dataset not set up"
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def ref_dataloader(self):
        assert self.ref_dataset is not None, "Ref dataset not set up"
        return torch.utils.data.DataLoader(
            self.ref_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )


class EventDataModule(BaseDataModule):
    def __init__(
        self,
        ref_threshold: int = 100,
        sep_event_channel: bool = False,
        yuv_interpolation: bool = False,
        mask_blur_factor: int = 0,
        sun_aug_prob: float = 0,
        gs_patch_prob: float = 0,
        glare_aug_prob: float = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.ref_threshold = ref_threshold
        self.yuv_interpolation = yuv_interpolation
        self.mask_blur_factor = mask_blur_factor
        self.sun_aug_prob = sun_aug_prob
        self.gs_patch_prob = gs_patch_prob
        self.glare_aug_prob = glare_aug_prob
        self.sep_event_channel = sep_event_channel

    def setup(self, stage: str) -> None:
        val_files = self.img_paths[: self.val_n]
        test_files = self.img_paths[self.val_n : self.val_n + self.test_n]
        train_files = self.img_paths[self.val_n + self.test_n :]
        train_files = [p for p in train_files if p.match(self.train_img_glob)]
        assert len(train_files) > 0, "No training files found"

        logger.info(f"Setting up datasets for stage: {stage}")
        if stage == "fit" or stage is None:
            self.train_dataset = dataset.BGREMDataset(
                train_files,
                masker=transforms.RandomizedMasker(),
                bgr_transform=T.Compose(
                    [
                        T.ToPILImage(),
                        T.ToTensor(),
                    ]
                ),
                event_transform=T.Compose(
                    [
                        transforms.RandomizedSunAdder(self.sun_aug_prob),
                        transforms.RandomizedBrightnessScaler(0.5, 1.5),
                        transforms.RandomizedContrastScaler(0.5, 1.5),
                    ]
                ),
                masked_bgr_transform=T.Compose(
                    [
                        transforms.RandomizedMaskAwareGlareAdder(self.glare_aug_prob),
                        transforms.RandomizedMaskAwareGrayscaleAdder(
                            self.gs_patch_prob
                        ),
                    ]
                ),
                separate_event_channel=self.sep_event_channel,
                blur_factor=self.mask_blur_factor,
                yuv_interpolation=self.yuv_interpolation,
            )
            self.val_dataset = dataset.BGREMDataset(
                val_files,
                masker=transforms.RandomizedMasker(fix_by_idx=True),
                bgr_transform=T.Compose(
                    [
                        T.ToTensor(),
                    ]
                ),
                separate_event_channel=self.sep_event_channel,
                blur_factor=self.mask_blur_factor,
                yuv_interpolation=self.yuv_interpolation,
            )
        if stage == "test" or stage is None:
            self.test_dataset = dataset.BGREMDataset(
                test_files,
                masker=transforms.RandomizedMasker(fix_by_idx=True),
                bgr_transform=T.Compose(
                    [
                        T.ToTensor(),
                    ]
                ),
                separate_event_channel=self.sep_event_channel,
                blur_factor=self.mask_blur_factor,
                yuv_interpolation=self.yuv_interpolation,
            )
        if stage == "ref" or stage is None:
            self.ref_dataset = dataset.BGREMDataset(
                self.ref_paths,
                masker=transforms.DiffIntensityMasker(self.ref_threshold),
                bgr_transform=T.Compose(
                    [
                        T.ToTensor(),
                    ]
                ),
                separate_event_channel=self.sep_event_channel,
                blur_factor=self.mask_blur_factor,
                yuv_interpolation=self.yuv_interpolation,
            )


class ArtifactDetectionDataModule(BaseDataModule):
    def __init__(
        self,
        p_flare: float,
        p_sun: float,
        p_glare: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.p_flare = p_flare
        self.p_sun = p_sun
        self.p_glare = p_glare

    def get_augmenter(
        self, fix_by_idx: bool = False
    ) -> transforms.CompositeLightArtifactAugmenter:
        return transforms.CompositeLightArtifactAugmenter(
            augmenters=[
                transforms.LensFlareAdder(1, 10, 5, 100, 0.25, 0.8, 0.2),
                transforms.VeilingGlareAdder(10, 150, 10, 150, 0.6),
                transforms.SunAdder(10, 25, 0.2, 0.5, 0.5),
            ],
            probs=[self.p_flare, self.p_glare, self.p_sun],
            fix_by_idx=fix_by_idx,
        )

    def setup(self, stage: str) -> None:
        val_files = self.img_paths[: self.val_n]
        test_files = self.img_paths[self.val_n : self.val_n + self.test_n]
        train_files = self.img_paths[self.val_n + self.test_n :]
        train_files = [p for p in train_files if p.match(self.train_img_glob)]
        assert len(train_files) > 0, "No training files found"

        logger.info(f"Setting up datasets for stage: {stage}")
        if stage == "fit" or stage is None:
            self.train_dataset = dataset.BGRArtifcatDataset(
                train_files,
                transform=T.Compose(
                    [
                        T.ToTensor(),
                    ]
                ),
                augmenter=self.get_augmenter(fix_by_idx=False),
            )
            self.val_dataset = dataset.BGRArtifcatDataset(
                val_files,
                transform=T.Compose(
                    [
                        T.ToTensor(),
                    ]
                ),
                augmenter=self.get_augmenter(fix_by_idx=True),
            )
        if stage == "test" or stage is None:
            self.test_dataset = dataset.BGRArtifcatDataset(
                test_files,
                transform=T.Compose(
                    [
                        T.ToTensor(),
                    ]
                ),
                augmenter=self.get_augmenter(fix_by_idx=True),
            )
        if stage == "ref" or stage is None:
            self.ref_dataset = dataset.BGRArtifcatDataset(
                self.ref_paths,
                transform=T.Compose(
                    [
                        T.ToTensor(),
                    ]
                ),
                augmenter=transforms.CompositeLightArtifactAugmenter(
                    augmenters=[],
                    probs=[],
                ),
            )
