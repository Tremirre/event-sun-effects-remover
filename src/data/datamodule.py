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


class EventDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: pathlib.Path,
        ref_dir: pathlib.Path | None = None,
        ref_threshold: int = 100,
        num_workers: int = 0,
        val_prob: float = 0.1,
        test_prob: float = 0.1,
        batch_size: int = 32,
        frac_used: float = 1,
        sep_event_channel: bool = False,
        img_glob: str = "**/*.npy",
    ) -> None:
        super().__init__()
        self.num_workers = num_workers
        self.ref_threshold = ref_threshold
        self.data_dir = data_dir
        self.val_prob = val_prob
        self.test_prob = test_prob
        self.batch_size = batch_size
        self.frac_used = frac_used
        self.ref_dir = ref_dir
        self.ref_paths = list(ref_dir.glob("**/*.npy")) if ref_dir else []
        self.img_paths = list(self.data_dir.glob(img_glob))
        self.sep_event_channel = sep_event_channel
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
        val_files = self.img_paths[: self.val_n]
        test_files = self.img_paths[self.val_n : self.val_n + self.test_n]
        train_files = self.img_paths[self.val_n + self.test_n :]

        logger.info(f"Setting up datasets for stage: {stage}")
        if stage == "fit" or stage is None:
            self.train_dataset = dataset.BGREMDataset(
                train_files,
                masker=transforms.RandomizedMasker(),
                bgr_transform=T.Compose(
                    [
                        T.ToPILImage(),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                    ]
                ),
                event_transform=T.Compose(
                    [
                        transforms.RandomizedBrightnessScaler(0.5, 1.5),
                        transforms.RandomizedContrastScaler(0.5, 1.5),
                        transforms.RadnomizedGaussianBlur(1, 10),
                    ]
                ),
                separate_event_channel=self.sep_event_channel,
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
            )

    def get_dataset_sizes(self) -> dict[str, int]:
        train_n = len(self.img_paths) - self.val_n - self.test_n
        return {
            "train_size": train_n,
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
