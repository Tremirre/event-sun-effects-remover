import logging
import random

import cv2
import numpy as np
import PIL.Image
import pytorch_lightning.loggers as pl_loggers
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def set_global_seed(seed: int):
    torch.backends.cudnn.deterministic = True
    logger.info(f"Setting global seed: {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_image_batch(
    event_gt: torch.Tensor,
    bgr_gt: torch.Tensor,
    bgr_with_artifact: torch.Tensor,
    bgr_inpaint: torch.Tensor,
    est_artifact_map: torch.Tensor,
    artifact_map: torch.Tensor,
    logger: pl_loggers.Logger,
    global_step: int,
    tag: str,
):
    """If not Neptune Logger -> skip"""
    if not isinstance(logger, pl_loggers.NeptuneLogger):
        return
    event_gt_np = event_gt.cpu().numpy()
    bgr_gt_np = bgr_gt.cpu().numpy()
    bgr_with_artifact_np = bgr_with_artifact.cpu().numpy()
    bgr_inpaint_np = bgr_inpaint.cpu().numpy()
    est_artifact_map_np = est_artifact_map.cpu().numpy()
    artifact_map_np = artifact_map.cpu().numpy()

    bgr_gt_np = (bgr_gt_np * 255).astype(np.uint8)
    bgr_gt_np = np.transpose(bgr_gt_np, (0, 2, 3, 1))

    bgr_with_artifact_np = (bgr_with_artifact_np * 255).astype(np.uint8)
    bgr_with_artifact_np = np.transpose(bgr_with_artifact_np, (0, 2, 3, 1))

    bgr_inpaint_np = (bgr_inpaint_np * 255).astype(np.uint8)
    bgr_inpaint_np = np.transpose(bgr_inpaint_np, (0, 2, 3, 1))

    event_gt_np = (event_gt_np * 255).astype(np.uint8)
    event_gt_np = np.transpose(event_gt_np, (0, 2, 3, 1))
    event_gt_np = np.repeat(event_gt_np, 3, axis=3)

    est_artifact_map_np = (est_artifact_map_np * 255).astype(np.uint8)
    est_artifact_map_np = np.transpose(est_artifact_map_np, (0, 2, 3, 1))
    est_artifact_map_np = np.repeat(est_artifact_map_np, 3, axis=3)

    artifact_map_np = (artifact_map_np * 255).astype(np.uint8)
    artifact_map_np = np.transpose(artifact_map_np, (0, 2, 3, 1))
    artifact_map_np = np.repeat(artifact_map_np, 3, axis=3)

    detection_result = np.concatenate(
        [bgr_with_artifact_np, est_artifact_map_np, artifact_map_np],
        axis=1,
    )
    detection_result = np.concatenate(detection_result, axis=1)
    detection_result = detection_result[:, :, ::-1]
    detection_result = cv2.resize(
        detection_result,
        (detection_result.shape[1] // 2, detection_result.shape[0] // 2),
        interpolation=cv2.INTER_LINEAR,
    )

    inpainting_result = np.concatenate(
        [bgr_with_artifact_np, bgr_inpaint_np, bgr_gt_np, event_gt_np],
        axis=1,
    )
    inpainting_result = np.concatenate(inpainting_result, axis=1)
    inpainting_result = inpainting_result[:, :, ::-1]
    inpainting_result = cv2.resize(
        inpainting_result,
        (inpainting_result.shape[1] // 2, inpainting_result.shape[0] // 2),
        interpolation=cv2.INTER_LINEAR,
    )

    detection_result_pil = PIL.Image.fromarray(detection_result)
    inpainting_result_pil = PIL.Image.fromarray(inpainting_result)

    # scale down 2 times

    logger.experiment[f"{tag}_det_comparison"].append(detection_result_pil)
    logger.experiment[f"{tag}_rec_comparison"].append(inpainting_result_pil)
