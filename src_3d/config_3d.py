from pathlib import Path
import torch
import random
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_ROOT = BASE_DIR / "baseball_kaggle_dataset_trimmed_only"
TRAIN_VIDEO_DIR = DATA_ROOT / "train_trimmed"
TEST_VIDEO_DIR  = DATA_ROOT / "test"

TRAIN_CSV     = DATA_ROOT / "data" / "train_ground_truth.csv"
TEST_FEAT_CSV = DATA_ROOT / "data" / "test_features.csv"

TEMPLATE_CSV = BASE_DIR / "test_submission_template.csv"

OUTPUT_3D_DIR = BASE_DIR / "outputs_3d"
OUTPUT_3D_DIR.mkdir(exist_ok=True)

TABULAR_MODEL_CLASS_3D_PATH = OUTPUT_3D_DIR / "tabular_class_3d_xgb.json"
TABULAR_MODEL_ZONE_3D_PATH  = OUTPUT_3D_DIR / "tabular_zone_3d_xgb.json"

DEEP_MODEL_3D_PATH = OUTPUT_3D_DIR / "deep_model_3d_best.pth"
SUBMISSION_3D_PATH = OUTPUT_3D_DIR / "my_submission_3d.csv"

NUM_FRAMES_3D   = 16      
IMAGE_SIZE_3D   = 160
BATCH_SIZE_3D   = 8
NUM_EPOCHS_3D   = 25
LEARNING_RATE_3D = 1e-4
RANDOM_SEED_3D   = 42

NUM_WORKERS_TRAIN_3D = 4
NUM_WORKERS_TEST_3D  = 4

DEVICE_3D = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TABULAR_COLS = [
    "sz_top", "sz_bot",
    "release_speed", "effective_speed",
    "release_spin_rate",
    "release_pos_x", "release_pos_y", "release_pos_z",
    "release_extension",
    "pfx_x", "pfx_z",
    "stand", "p_throws",
]

CROP_X1 = 0.25
CROP_X2 = 0.70
CROP_Y1 = 0.2
CROP_Y2 = 0.8


def set_seed_3d(seed: int = RANDOM_SEED_3D):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
