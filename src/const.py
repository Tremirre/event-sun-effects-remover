import pathlib

CHANNELS_OUT = 3
IMG_WIDTH = 640
IMG_HEIGHT = 480
REFERENCE_LOGGING_FREQ = 5

DATA_DIR = pathlib.Path("data")
TRAIN_VAL_TEST_DIR = DATA_DIR / "processed"
REF_DIR = DATA_DIR / "ref"
FLARES_DIR = DATA_DIR / "detect" / "flare"
ARTIFACT_DET_TEST_DIR = DATA_DIR / "detect" / "test"
