# src/dataset.py
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import TABULAR_COLS, NUM_FRAMES, TRAIN_VIDEO_DIR, TEST_VIDEO_DIR
from .utils_video import load_and_sample_video


def preprocess_tabular(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Encode categoricals and compute mean/std.
    Returns modified df, means, stds (float32).
    """
    df = df.copy()
    for col in ["stand", "p_throws"]:
        df[col] = df[col].map({"L": 0.0, "R": 1.0}).astype("float32")

    df[TABULAR_COLS] = df[TABULAR_COLS].astype("float32")
    df[TABULAR_COLS] = df[TABULAR_COLS].fillna(df[TABULAR_COLS].mean())

    means = df[TABULAR_COLS].mean().astype("float32")
    stds = df[TABULAR_COLS].std().replace(0, 1.0).astype("float32")

    return df, means, stds


class PitchTrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, video_dir,
                 tab_means: pd.Series, tab_stds: pd.Series,
                 is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.video_dir = video_dir
        self.is_train = is_train
        self.tab_means = tab_means
        self.tab_stds = tab_stds

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row["file_name"]
        video_path = self.video_dir / file_name

        video = load_and_sample_video(str(video_path), NUM_FRAMES, is_train=self.is_train)

        tab = row[TABULAR_COLS].values.astype("float32")
        tab = (tab - self.tab_means.values) / (self.tab_stds.values + 1e-6)
        tab = torch.from_numpy(tab).float()

        pitch_class = 1.0 if row["pitch_class"] == "strike" else 0.0
        zone = int(row["zone"]) - 1

        plate_x = float(row["plate_x"])
        plate_z = float(row["plate_z"])

        return (
            video,                          # [T,C,H,W]
            tab,                            # [F]
            torch.tensor(pitch_class, dtype=torch.float32),
            torch.tensor(zone,      dtype=torch.long),
            torch.tensor(plate_x,   dtype=torch.float32),
            torch.tensor(plate_z,   dtype=torch.float32),
            file_name,
        )


class PitchTestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, video_dir,
                 tab_means: pd.Series, tab_stds: pd.Series):
        self.df = df.reset_index(drop=True)
        self.video_dir = video_dir
        self.tab_means = tab_means
        self.tab_stds = tab_stds

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row["file_name"]
        video_path = self.video_dir / file_name

        video = load_and_sample_video(str(video_path), NUM_FRAMES, is_train=False)

        tab = row[TABULAR_COLS].values.astype("float32")
        tab = (tab - self.tab_means.values) / (self.tab_stds.values + 1e-6)
        tab = torch.from_numpy(tab).float()

        return video, tab, file_name
