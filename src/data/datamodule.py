import logging
import pathlib

import pytorch_lightning as pl
import torch.utils.data
import torchvision.transforms as T

from src import const
from src.data import artifacts, dataset, transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_paths: list[pathlib.Path],
        val_paths: list[pathlib.Path],
        test_paths: list[pathlib.Path],
        ref_paths: list[pathlib.Path] | None = None,
        num_workers: int = 0,
        batch_size: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = test_paths
        self.ref_paths = ref_paths or []
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.ref_dataset = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        raise NotImplementedError

    def get_dataset_sizes(self) -> dict[str, int]:
        return {
            "train_size": len(self.train_paths),
            "val_size": len(self.val_paths),
            "test_size": len(self.test_paths),
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


class JointDataModule(BaseDataModule):
    def __init__(
        self,
        p_flare: float = 0.0,
        p_sun: float = 0.0,
        p_glare: float = 0.0,
        p_hq_flare: float = 0.0,
        target_binarization: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.p_flare = p_flare
        self.p_sun = p_sun
        self.p_glare = p_glare
        self.p_hq_flare = p_hq_flare
        self.target_binarization = target_binarization
        logger.info("Initialized Joint Data module")

    def get_artifact_source(self) -> artifacts.CompositeLightArtifactGenerator:
        return artifacts.CompositeLightArtifactGenerator(
            augmenters=[
                artifacts.LensFlareAdder(1, 10, 1, 20, 0.6, 0.9, 0.2),
                artifacts.VeilingGlareAdder(10, 150, 10, 60, 0.3),
                artifacts.SunAdder(4, 25, 0.0, 0.5, 0.3, 0.8),
                artifacts.HQFlareBasedAugmenter(
                    list(const.FLARES_DIR.glob("**/*.png"))
                ),
            ],
            probs=[self.p_flare, self.p_glare, self.p_sun, self.p_hq_flare],
        )

    def setup(self, stage: str) -> None:
        logger.info(f"Setting up datasets for stage: {stage}")
        if stage == "fit" or stage is None:
            self.train_dataset = dataset.BGREADataset(
                self.train_paths,
                transform=T.Compose(
                    [
                        transforms.RandomizedEventBrightnessScaler(0.5, 1.5),
                        transforms.RandomizedEventContrastScaler(0.5, 1.5),
                        transforms.RandomizedEventSunAdder(0.05),
                        transforms.EventMaskChannelRemover(),
                        T.ToTensor(),
                    ]
                ),
                artifact_source=self.get_artifact_source(fix_by_idx=False),
            )
            self.val_dataset = dataset.BGREADataset(
                self.val_paths,
                transform=T.Compose(
                    [
                        transforms.EventMaskChannelRemover(),
                        T.ToTensor(),
                    ]
                ),
                artifact_source=artifacts.LightArtifactExtractor(),
            )
        if stage == "test" or stage is None:
            self.test_dataset = dataset.BGREADataset(
                self.test_paths,
                transform=T.Compose(
                    [
                        transforms.EventMaskChannelRemover(),
                        T.ToTensor(),
                    ]
                ),
                artifact_source=artifacts.LightArtifactExtractor(),
            )
        if stage == "ref" or stage is None:
            self.ref_dataset = dataset.BGREADataset(
                self.ref_paths,
                transform=T.Compose(
                    [
                        transforms.EventMaskChannelRemover(),
                        T.ToTensor(),
                    ]
                ),
            )
