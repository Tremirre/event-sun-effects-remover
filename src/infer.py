import logging

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import tqdm

from src import const
from src.config import Config
from src.data import dataset, transforms

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s %(name)s %(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    config = Config.from_args()
    config.prepare_inference()
    model = config.get_model().to(DEVICE)
    img_paths = list(config.data_dir.glob("**/*.npy"))
    logger.info(f"Found {len(img_paths)} images in {config.data_dir}")
    infer_dataset = dataset.BGREMDataset(
        img_paths,
        masker=transforms.DiffIntensityMasker(config.diff_intensity),
        bgr_transform=T.Compose([T.ToTensor()]),
        separate_event_channel=config.event_channel,
        yuv_interpolation=config.yuv_interpolation,
        soft_mask=config.soft_masking,
    )
    infer_loader = torch.utils.data.DataLoader(
        infer_dataset,
        batch_size=config.batch_size,
        num_workers=4,
        shuffle=False,
        persistent_workers=True,
    )
    num_batches = len(infer_loader)
    logger.info(f"Starting inference on {num_batches} batches")
    iter_batches = tqdm.tqdm(infer_loader, total=num_batches)
    all_bgrem = []
    for batch in iter_batches:
        bgrem, target = batch
        bgr_orig = bgrem[:, :3].detach().cpu().numpy()
        bgr_orig = np.transpose(bgr_orig, (0, 2, 3, 1))
        last_channel = bgrem.shape[1] - 1
        mask = bgrem[:, last_channel].unsqueeze(-1).detach().cpu().numpy()
        bgrem = bgrem.to(DEVICE)
        bgr = model(bgrem)
        bgr = bgr.detach().cpu().numpy()

        bgr = np.transpose(bgr, (0, 2, 3, 1))
        if not config.full_pred:
            bgr = np.where(mask, bgr, bgr_orig)
        bgr = np.clip(bgr * 255, 0, 255).astype(np.uint8)
        all_bgrem.append(bgr)

    all_bgrem = np.concatenate(all_bgrem, axis=0)

    logger.info(f"Saving output to {config.output}")
    out = cv2.VideoWriter(
        str(config.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (const.IMG_WIDTH, const.IMG_HEIGHT),
    )
    for frame in all_bgrem:
        out.write(frame)
    out.release()
