import pathlib

import numpy as np
import pytorch_lightning as pl
import torch.utils.data
import torchvision.transforms as T

from src.data import dataset, transforms


class EventDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: pathlib.Path,
        num_workers: int = 4,
        val_prob: float = 0.1,
        test_prob: float = 0.1,
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.val_prob = val_prob
        self.test_prob = test_prob
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        img_paths = list(self.data_dir.glob("**/*.npy"))
        np.random.shuffle(img_paths)  # type: ignore
        n = len(img_paths)
        val_n = int(n * self.val_prob)
        test_n = int(n * self.test_prob)

        val_files = img_paths[:val_n]
        test_files = img_paths[val_n : val_n + test_n]
        train_files = img_paths[val_n + test_n :]

        self.train_dataset = dataset.BGREMDataset(
            train_files,
            masker=transforms.RandomizedMasker(1, 6),
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
                ]
            ),
        )
        self.val_dataset = dataset.BGREMDataset(
            val_files,
            masker=transforms.RandomizedMasker(1, 6),
            bgr_transform=T.Compose(
                [
                    T.ToTensor(),
                ]
            ),
        )
        self.test_dataset = dataset.BGREMDataset(
            test_files,
            masker=transforms.RandomizedMasker(1, 6),
            bgr_transform=T.Compose(
                [
                    T.ToTensor(),
                ]
            ),
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
