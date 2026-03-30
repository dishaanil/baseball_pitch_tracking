import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config_3d import TABULAR_COLS, NUM_FRAMES_3D
from .utils_video_3d import load_and_sample_video_3d


def preprocess_tabular_3d(df: pd.DataFrame):
    df = df.copy()

    for col in ["stand", "p_throws"]:
        if col in df.columns:
            df[col] = df[col].map({"L": 0.0, "R": 1.0}).astype("float32")

    df[TABULAR_COLS] = df[TABULAR_COLS].astype("float32")

    df[TABULAR_COLS] = df[TABULAR_COLS].fillna(df[TABULAR_COLS].mean())

    means = df[TABULAR_COLS].mean()
    stds = df[TABULAR_COLS].std().replace(0, 1.0)

    return df, means.astype("float32"), stds.astype("float32")


class PitchTrainDataset3D(Dataset):
    def __init__(self, df: pd.DataFrame, video_dir: str,
                 tab_means, tab_stds, is_train: bool = True):
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
        video_path = os.path.join(self.video_dir, file_name)

        video = load_and_sample_video_3d(
            video_path,
            num_frames=NUM_FRAMES_3D,
            is_train=self.is_train,
        )

        tab = row[TABULAR_COLS].values.astype("float32")
        tab = (tab - self.tab_means.values) / (self.tab_stds.values + 1e-6)
        tab = torch.from_numpy(tab).float()

        pitch_class = 1.0 if row["pitch_class"] == "strike" else 0.0
        zone = int(row["zone"]) - 1  

        plate_x = float(row["plate_x"])
        plate_z = float(row["plate_z"])

        return (
            video,                                  
            tab,                                    
            torch.tensor(pitch_class, dtype=torch.float32),
            torch.tensor(zone,       dtype=torch.long),
            torch.tensor(plate_x,    dtype=torch.float32),
            torch.tensor(plate_z,    dtype=torch.float32),
            file_name,
        )


class PitchTestDataset3D(Dataset):
    def __init__(self, df: pd.DataFrame, video_dir: str,
                 tab_means, tab_stds):
        self.df = df.reset_index(drop=True)
        self.video_dir = video_dir
        self.tab_means = tab_means
        self.tab_stds = tab_stds

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row["file_name"]
        video_path = os.path.join(self.video_dir, file_name)

        video = load_and_sample_video_3d(
            video_path,
            num_frames=NUM_FRAMES_3D,
            is_train=False,
        )

        tab = row[TABULAR_COLS].values.astype("float32")
        tab = (tab - self.tab_means.values) / (self.tab_stds.values + 1e-6)
        tab = torch.from_numpy(tab).float()

        return video, tab, file_name
