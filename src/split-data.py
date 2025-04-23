import pathlib

import numpy as np
import tqdm

from src import const
from src.data import artifacts


def get_artifact_source() -> artifacts.SingleChoiceArtifactSource:
    return artifacts.SingleChoiceArtifactSource(
        augmenters=[
            artifacts.LensFlareAdder(1, 10, 1, 20, 0.6, 0.9, 0.2),
            artifacts.VeilingGlareAdder(10, 150, 10, 60, 0.3),
            artifacts.SunAdder(4, 25, 0.0, 0.5, 0.3, 0.8),
            artifacts.HQFlareBasedAugmenter(list(const.FLARES_DIR.glob("**/*.png"))),
            artifacts.NoOpAugmenter(),
        ],
        probs=[0.2, 0.1, 0.2, 0.45, 0.05],
    )


SRC_PATH = pathlib.Path("data") / "processed"
SRC_GLOB = "**/{scene}*.npy"
TARGET_PATH = pathlib.Path("data") / "split"

ASSINGMENT = {
    "train": [
        "001-town01",
        "001-town02",
        "interlaken_00_c",
        "interlaken_00_f",
        "zurich_00_b",
        "zurich_city_05_b",
        "zurich_city_06_a",
        "back3",
        "back4",
        "highway2",
        "cityscapes/",
    ],
    "val": [
        "interlaken_00_g",
        "zurich_city_07_a",
        "back1",
        "back9",
        "street2",
    ],
    "test": [
        "interlaken_00_d",
        "zurich_01_a",
        "zurich_city_13_b",
        "highway1",
    ],
}

if __name__ == "__main__":
    np.random.seed(0)
    generator = get_artifact_source()
    for split, scenes in ASSINGMENT.items():
        split_path = TARGET_PATH / split
        split_path.mkdir(parents=True, exist_ok=True)
        # Remove existing files in the split directory
        print(f"Cleaning out {split_path}")
        for file in split_path.glob("**/*"):
            if file.is_file():
                file.unlink()

        split_files = []
        for scene in scenes:
            filled_glob = SRC_GLOB.format(scene=scene)
            split_files.extend(list(SRC_PATH.glob(filled_glob)))
        pbar = tqdm.tqdm(split_files, desc=f"Processing {split} files")
        for file in pbar:
            if not file.is_file():
                pbar.set_postfix_str(f"Skipping {file} (not a file)")
                continue
            target_path = split_path / file.relative_to(SRC_PATH)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            img = np.load(file)
            suffix = ""
            if img.shape[2] != 5:
                fullwhite = np.ones_like(img[:, :, 3]) * 255
                img = np.concatenate([img, fullwhite[:, :, np.newaxis]], axis=-1)
                suffix = " (+ full white)"

            if split != "train":
                artifact_map = generator(img)
                img = np.dstack([img, artifact_map])
                suffix += " (+ artifact map)"

            np.save(target_path, img)
            pbar.set_postfix_str(f"Saved {target_path}{suffix}")
