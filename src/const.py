import pathlib

CHANNELS_IN = 4  # RGB + Event + Mask
CHANNELS_OUT = 3  # RGB
IMG_WIDTH = 640
IMG_HEIGHT = 480

DATA_DIR = pathlib.Path("data")
TRAIN_VAL_TEST_DIR = DATA_DIR / "processed"
REF_DIR = DATA_DIR / "ref"
