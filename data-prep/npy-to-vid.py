"""
Script to display video frames from a .npy file
"""

import argparse
import pathlib

import cv2
import numpy as np
import tqdm


def load_frames(path: pathlib.Path) -> np.ndarray:
    """
    Load frames from a .npy file or a directory containing .npy files.

    Args:
        path (pathlib.Path): Path to the .npy file or directory.

    Returns:
        np.ndarray: Array of frames.
    """
    if path.is_dir():
        files = sorted(path.glob("*.npy"))
        frames = [
            np.load(file)[:, :, :3] for file in tqdm.tqdm(files, desc="Loading frames")
        ]
        return np.array(frames)
    if path.suffix == ".npy":
        return np.load(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def export_frames_to_mp4(frames: np.ndarray, output: pathlib.Path) -> None:
    """
    Export frames to an MP4 file.

    Args:
        frames (np.ndarray): Array of frames to export.
    """
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    out = cv2.VideoWriter(str(output), fourcc, 30.0, (width, height))

    for frame in tqdm.tqdm(frames, desc="Exporting frames to MP4"):
        out.write(frame.astype(np.uint8))

    out.release()
    print("Frames exported to output.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display video frames from a .npy file"
    )
    parser.add_argument(
        "frames",
        help="Path to the .npy or directory file containing RGB frames",
        type=pathlib.Path,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
    )

    args = parser.parse_args()
    assert args.frames.exists(), f"File {args.frames} does not exist"
    assert args.output is not None, "Output path must be specified"

    frames = load_frames(args.frames)
    export_frames_to_mp4(frames, args.output)
