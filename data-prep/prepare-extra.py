from __future__ import annotations

import argparse
import dataclasses
import pathlib

import cv2
import numpy as np
import tqdm
import utils


@dataclasses.dataclass
class Args:
    input_folder: pathlib.Path
    output_folder: pathlib.Path
    target_height: int
    target_width: int

    def __post_init__(self):
        self.input_folder.mkdir(parents=True, exist_ok=True)
        self.output_folder.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_args(cls) -> Args:
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_folder", type=pathlib.Path, required=True)
        parser.add_argument("--output_folder", type=pathlib.Path, required=True)
        parser.add_argument("--target_height", type=int, required=False, default=480)
        parser.add_argument("--target_width", type=int, required=False, default=640)
        return cls(**vars(parser.parse_args()))


def preprocess_image(img: np.ndarray) -> np.ndarray:
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gs = cv2.bilateralFilter(img_gs, 4, 75, 75)
    img_gs_gb = cv2.GaussianBlur(img_gs, (5, 5), 0)
    img_gs = cv2.addWeighted(img_gs, 1.5, img_gs_gb, -0.5, 0)
    return img_gs


if __name__ == "__main__":
    args = Args.from_args()

    imgs = list(args.input_folder.glob("**/*.png"))

    for img_path in tqdm.tqdm(imgs):
        img = cv2.imread(str(img_path))
        img = utils.crop_img_to_size(img, args.target_width, args.target_height)
        img_gs = preprocess_image(img.copy())
        # add a channel dimension
        img_gs = np.expand_dims(img_gs, axis=-1)
        img = np.concatenate([img, img_gs], axis=-1)
        new_name = img_path.stem + ".npy"
        np.save(str(args.output_folder / new_name), img)
