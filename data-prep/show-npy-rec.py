"""
Script to display video frames from a .npy file
"""

import argparse
import os

import cv2
import numpy as np


def show_video_from_npy(npy_file_path, fps=30, scale=1.0):
    """
    Load and display a video from a .npy file containing RGB frames

    Args:
        npy_file_path (str): Path to the .npy file
        fps (int): Frames per second for playback
        scale (float): Scale factor for display size
    """
    # Check if file exists
    if not os.path.exists(npy_file_path):
        print(f"Error: File '{npy_file_path}' not found.")
        return

    try:
        # Load the numpy array from file
        print(f"Loading frames from {npy_file_path}...")
        frames = np.load(npy_file_path)

        # Check if the array contains frames
        if len(frames.shape) < 3:
            print(
                "Error: The numpy array doesn't have the expected shape for video frames."
            )
            return

        num_frames = frames.shape[0]
        height = frames.shape[1]
        width = frames.shape[2]

        print(f"Loaded {num_frames} frames of size {width}x{height}")

        # Create a window
        window_name = os.path.basename(npy_file_path)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        if scale != 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            cv2.resizeWindow(window_name, new_width, new_height)

        # Calculate frame delay in milliseconds
        delay = int(1000 / fps)

        # Display each frame
        for i in range(num_frames):
            frame = frames[i]

            # Check if frame is in RGB format
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Convert RGB to BGR for OpenCV display
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Display the frame
                cv2.imshow(window_name, bgr_frame)

                # Exit if ESC key is pressed
                key = cv2.waitKey(delay) & 0xFF
                if key == 27:  # ESC key
                    break
            else:
                print(f"Error: Frame {i} is not in the expected RGB format.")
                break

        cv2.destroyAllWindows()
        print("Playback finished")

    except Exception as e:
        print(f"Error while processing the file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display video frames from a .npy file"
    )
    parser.add_argument("file", help="Path to the .npy file containing RGB frames")
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for playback (default: 30)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor for display (default: 1.0)",
    )

    args = parser.parse_args()

    show_video_from_npy(args.file, args.fps, args.scale)
