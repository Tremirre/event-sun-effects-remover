import pathlib

import numpy as np
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PRETRAINED_DIR = pathlib.Path("pretrained")
META_DIR = pathlib.Path("experiments/meta")
VIDEOS_DIR = pathlib.Path("experiments/videos")
IMAGES_DIR = pathlib.Path("experiments/images")
META_DIR.mkdir(exist_ok=True)
VIDEOS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

EVENT_MTX = np.array(
    [
        [1.103e03, 0.000e00, 3.200e02],
        [0.000e00, 5.620e02, 2.920e02],
        [0.000e00, 0.000e00, 1.000e00],
    ]
)
EVENT_DIST = np.array([[-3.1e-01, -6.3e-02, 1.3e-02, 1.0e-05, 1.0e-04]])
