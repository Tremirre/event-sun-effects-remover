import json
import pathlib
from multiprocessing import Pool

import numpy as np
import tqdm
from skimage.metrics import structural_similarity as ssim

from .comp_eval import read_video

COMP_RES_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "comp-results"

# ---------------------------
# Type aliases
# ---------------------------
VideoRGB = np.ndarray  # shape (N, H, W, 3)
VideoGray = np.ndarray  # shape (N, H, W)


def to_luminance(video):
    """Convert RGB video to grayscale luminance (Y channel)."""
    if video.ndim == 4 and video.shape[-1] == 3:
        # standard Rec. 709 weights
        return 0.2126 * video[..., 0] + 0.7152 * video[..., 1] + 0.0722 * video[..., 2]
    return video  # already grayscale


def ntsd(video: VideoRGB, eps: float = 1e-6) -> float:
    gray = to_luminance(video)
    mu = gray.mean(axis=0)
    sigma = gray.std(axis=0)
    return float(np.mean(sigma / (mu + eps)))


def tgm(video: VideoRGB) -> float:
    gray = to_luminance(video)
    diff = np.abs(np.diff(gray, axis=0))
    return float(diff.mean())


def tssim(video: VideoRGB) -> float:
    gray = to_luminance(video)
    N = gray.shape[0]
    ssim_vals = [
        ssim(gray[t], gray[t + 1], data_range=gray.max() - gray.min())
        for t in range(N - 1)
    ]
    return float(np.mean(ssim_vals))


def flicker_energy(
    video: VideoRGB, f_low: float = 0.5, f_high: float = 15, fps: float = 30
) -> float:
    """
    Compute mean flicker energy in frequency band [f_low,f_high] in Hz
    """
    gray = to_luminance(video)
    N, H, W = gray.shape
    fft_vals = np.fft.rfft(gray, axis=0)
    freqs = np.fft.rfftfreq(N, 1 / fps)
    band_mask = (freqs >= f_low) & (freqs <= f_high)
    fe = np.sum(np.abs(fft_vals[band_mask]) ** 2, axis=0)
    return float(fe.mean())


def sfpr(video: VideoRGB, tau: float = 0.05, eps: float = 1e-6) -> float:
    gray = to_luminance(video)
    mu = gray.mean(axis=0)
    sigma = gray.std(axis=0)
    strong_pixels = (sigma / (mu + eps)) > tau
    return float(np.mean(strong_pixels))


def fiv(video: VideoRGB) -> float:
    gray = to_luminance(video)
    mean_intensity_by_frame = gray.mean(axis=(1, 2))
    return float(mean_intensity_by_frame.var())


def compute_metrics_for_video(args):
    """Helper function for parallel processing."""
    video_path, competitor_name, division_name, metrics = args
    vid = read_video(video_path, verbose=False)
    results = []
    for metric in metrics:
        result = metric(vid)
        results.append(
            {
                "competitor": competitor_name,
                "division": division_name,
                "metric": metric.__name__,
                "value": result,
                "image": video_path.stem,
                "type": "flicker",
            }
        )
    return results


if __name__ == "__main__":
    # Prepare tasks for parallel processing
    tasks = []

    # Tasks for competitor videos
    for comp in sorted((COMP_RES_DIR / "preds" / "vids").iterdir()):
        for sequence in sorted(comp.glob("*.mp4")):
            tasks.append(
                (
                    sequence,
                    comp.name,
                    sequence.stem,
                    [ntsd, tgm, flicker_energy, sfpr, fiv],
                )
            )

    # Tasks for input videos
    for sequence in sorted((COMP_RES_DIR.parent / "videos").glob("*.mp4")):
        tasks.append(
            (sequence, "input", sequence.stem, [ntsd, tgm, flicker_energy, sfpr, fiv])
        )

    # Run in parallel
    n_cores = 4
    print(f"Processing {len(tasks)} videos using {n_cores} processes...")
    with Pool(processes=n_cores) as pool:
        results_list = list(
            tqdm.tqdm(
                pool.imap(compute_metrics_for_video, tasks),
                total=len(tasks),
                desc="Computing flicker metrics",
            )
        )

    # Flatten results
    flicker_res = [item for sublist in results_list for item in sublist]
    (COMP_RES_DIR / "scores" / "flicker_metrics.json").write_text(
        json.dumps(flicker_res, indent=4)
    )
