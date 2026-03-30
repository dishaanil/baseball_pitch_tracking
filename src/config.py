# src/config.py
from pathlib import Path
import torch
import random
import numpy as np

# ---- Paths ----
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_ROOT = BASE_DIR / "baseball_kaggle_dataset_trimmed_only"
TRAIN_VIDEO_DIR = DATA_ROOT / "train_trimmed"
TEST_VIDEO_DIR  = DATA_ROOT / "test"

TRAIN_CSV     = DATA_ROOT / "data" / "train_ground_truth.csv"
TEST_FEAT_CSV = DATA_ROOT / "data" / "test_features.csv"

TEMPLATE_CSV = BASE_DIR / "test_submission_template.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TABULAR_MODEL_CLASS_PATH = OUTPUT_DIR / "tabular_class_xgb.json"
TABULAR_MODEL_ZONE_PATH  = OUTPUT_DIR / "tabular_zone_xgb.json"
DEEP_MODEL_PATH          = OUTPUT_DIR / "deep_model_best.pth"
SUBMISSION_PATH          = OUTPUT_DIR / "my_submission.csv"

# ---- Hyperparams (minimal strong config) ----
NUM_FRAMES   = 24
IMAGE_SIZE   = 160
BATCH_SIZE   = 16          # moderate, not too tiny, not too huge
NUM_EPOCHS   = 10
LEARNING_RATE = 1e-4
RANDOM_SEED   = 42

# Dataloader workers (tune per machine)
NUM_WORKERS_TRAIN = 4
NUM_WORKERS_TEST  = 4

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tabular columns
TABULAR_COLS = [
    "sz_top", "sz_bot",
    "release_speed", "effective_speed",
    "release_spin_rate",
    "release_pos_x", "release_pos_y", "release_pos_z",
    "release_extension",
    "pfx_x", "pfx_z",
    "stand", "p_throws",
]


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
