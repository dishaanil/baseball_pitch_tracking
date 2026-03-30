# src/train_deep.py
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .config import (
    TRAIN_CSV,
    TRAIN_VIDEO_DIR,
    TABULAR_COLS,
    BATCH_SIZE,
    NUM_WORKERS_TRAIN,
    NUM_WORKERS_TEST,
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
    DEEP_MODEL_PATH,
    set_seed,
)
from .dataset import PitchTrainDataset, preprocess_tabular
from .models import DeepPitchModel


def compute_metrics_class_zone(y_class_true, y_class_prob, y_zone_true, y_zone_logits):
    """
    Compute:
      - acc_class (strike/ball)
      - acc_zone  (14 zones)
      - score = 0.7 * acc_class + 0.3 * acc_zone
    """
    y_class_pred = (y_class_prob > 0.5).astype(int)
    acc_class = (y_class_pred == y_class_true).mean()

    y_zone_pred = y_zone_logits.argmax(axis=1)
    acc_zone = (y_zone_pred == y_zone_true).mean()

    score = 0.7 * acc_class + 0.3 * acc_zone
    return acc_class, acc_zone, score


def train_deep_model():
    print("Using device:", DEVICE)
    set_seed(RANDOM_SEED)

    df = pd.read_csv(TRAIN_CSV)
    df, tab_means, tab_stds = preprocess_tabular(df)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df["zone"].astype(int),
    )

    train_ds = PitchTrainDataset(train_df, TRAIN_VIDEO_DIR, tab_means, tab_stds, is_train=True)
    val_ds   = PitchTrainDataset(val_df,   TRAIN_VIDEO_DIR, tab_means, tab_stds, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS_TRAIN,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS_TEST,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = DeepPitchModel(tab_dim=len(TABULAR_COLS))
    model.to(DEVICE)

    bce = nn.BCEWithLogitsLoss()
    ce  = nn.CrossEntropyLoss()
    l1  = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_score = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0

        print(f"\n=== Epoch {epoch} ===")
        batch_idx = 0

        for batch in train_loader:
            t0 = time.time()
            video, tab, y_class, y_zone, y_px, y_pz, _ = batch
            t_after_load = time.time()

            video   = video.to(DEVICE, non_blocking=True)
            tab     = tab.to(DEVICE,   non_blocking=True)
            y_class = y_class.to(DEVICE, non_blocking=True)
            y_zone  = y_zone.to(DEVICE,  non_blocking=True)
            y_px    = y_px.to(DEVICE,    non_blocking=True)
            y_pz    = y_pz.to(DEVICE,    non_blocking=True)

            optimizer.zero_grad()
            logit_class, logits_zone, px_pred, pz_pred = model(video, tab)

            loss_class = bce(logit_class, y_class)
            loss_zone  = ce(logits_zone,  y_zone)
            loss_px    = l1(px_pred,      y_px)
            loss_pz    = l1(pz_pred,      y_pz)

            # 🔒 ORIGINAL LOSS MIX
            loss = 0.7 * loss_class + 0.3 * loss_zone + 0.05 * (loss_px + loss_pz)

            loss.backward()
            optimizer.step()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t_after_step = time.time()

            train_loss += loss.item() * video.size(0)

            if batch_idx % 20 == 0:
                print(
                    f"batch {batch_idx:3d}: "
                    f"load={t_after_load - t0:.3f}s, "
                    f"step={t_after_step - t_after_load:.3f}s"
                )

            batch_idx += 1

        train_loss /= len(train_loader.dataset)

        # =====================
        # VALIDATION
        # =====================
        model.eval()
        val_loss = 0.0
        class_probs_all = []
        class_true_all  = []
        zone_logits_all = []
        zone_true_all   = []

        with torch.no_grad():
            for batch in val_loader:
                video, tab, y_class, y_zone, y_px, y_pz, _ = batch

                video   = video.to(DEVICE, non_blocking=True)
                tab     = tab.to(DEVICE,   non_blocking=True)
                y_class = y_class.to(DEVICE, non_blocking=True)
                y_zone  = y_zone.to(DEVICE,  non_blocking=True)
                y_px    = y_px.to(DEVICE,    non_blocking=True)
                y_pz    = y_pz.to(DEVICE,    non_blocking=True)

                logit_class, logits_zone, px_pred, pz_pred = model(video, tab)

                loss_class = bce(logit_class, y_class)
                loss_zone  = ce(logits_zone,  y_zone)
                loss_px    = l1(px_pred,      y_px)
                loss_pz    = l1(pz_pred,      y_pz)

                loss = 0.7 * loss_class + 0.3 * loss_zone + 0.05 * (loss_px + loss_pz)
                val_loss += loss.item() * video.size(0)

                class_probs_all.append(torch.sigmoid(logit_class).cpu().numpy())
                class_true_all.append(y_class.cpu().numpy())
                zone_logits_all.append(logits_zone.cpu().numpy())
                zone_true_all.append(y_zone.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        class_probs_all = np.concatenate(class_probs_all)
        class_true_all  = np.concatenate(class_true_all)
        zone_logits_all = np.concatenate(zone_logits_all)
        zone_true_all   = np.concatenate(zone_true_all)

        acc_class, acc_zone, score = compute_metrics_class_zone(
            class_true_all, class_probs_all, zone_true_all, zone_logits_all
        )

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"acc_class={acc_class:.4f}, acc_zone={acc_zone:.4f}, score={score:.4f}"
        )

        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "tab_means": tab_means,
                    "tab_stds": tab_stds,
                },
                DEEP_MODEL_PATH,
            )
            print(f"  Saved new best model (score={best_score:.4f}) → {DEEP_MODEL_PATH}")

    print("Training done. Best val score:", best_score)


if __name__ == "__main__":
    train_deep_model()
