import pathlib

CHANNELS_OUT = 3
IMG_WIDTH = 640
IMG_HEIGHT = 480
REFERENCE_LOGGING_FREQ = 5
VALIDATION_LOGGING_FREQ = 5

DATA_DIR = pathlib.Path("data")
SPLIT_DIR = DATA_DIR / "split"
TRAIN_DIR = SPLIT_DIR / "train"
VAL_DIR = SPLIT_DIR / "val"
TEST_DIR = SPLIT_DIR / "test"
DATA_PATTERN = "**/*.npy"

REF_DIR = DATA_DIR / "ref"
FLARES_DIR = DATA_DIR / "detect" / "flare"

TEST_EST_MAP_OUT = "estimate_map.mp4"
TEST_POST_EST_MAP_OUT = "post_estimate_map.mp4"
TEST_REC_FRAMES_OUT = "reconstructed.mp4"
