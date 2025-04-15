import pathlib

CHANNELS_OUT = 3
IMG_WIDTH = 640
IMG_HEIGHT = 480
REFERENCE_LOGGING_FREQ = 5

DATA_DIR = pathlib.Path("data")
INPAINT_DIR = DATA_DIR / "inpaint"
TRAIN_DIR = INPAINT_DIR / "train"
VAL_DIR = INPAINT_DIR / "val"
TEST_DIR = INPAINT_DIR / "test"
DATA_PATTERN = "**/*.npy"

REF_DIR = DATA_DIR / "ref"
FLARES_DIR = DATA_DIR / "detect" / "flare"
ARTIFACT_DET_TEST_DIR = DATA_DIR / "detect" / "test"
