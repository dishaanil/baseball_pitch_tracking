# src_3d/train_deep_3d.py
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .config_3d import (
    TRAIN_VIDEO_DIR, TRAIN_CSV,
    TABULAR_COLS,
    BATCH_SIZE_3D, NUM_WORKERS_TRAIN_3D, NUM_WORKERS_TEST_3D,
    DEVICE_3D, NUM_EPOCHS_3D, LEARNING_RATE_3D, RANDOM_SEED_3D,
    DEEP_MODEL_3D_PATH, set_seed_3d,
)
from .dataset_3d import PitchTrainDataset3D, preprocess_tabular_3d
from .models_3d import DeepPitchModel3D


def compute_metrics_class_zone(y_class_true, y_class_prob, y_zone_true, y_zone_logits):
    y_class_pred = (y_class_prob > 0.5).astype(int)
    acc_class = (y_class_pred == y_class_true).mean()

    y_zone_pred = y_zone_logits.argmax(axis=1)
    acc_zone = (y_zone_pred == y_zone_true).mean()

    score = 0.7 * acc_class + 0.3 * acc_zone
    return acc_class, acc_zone, score


def train_deep_model_3d():
    set_seed_3d(RANDOM_SEED_3D)

    df = pd.read_csv(TRAIN_CSV)
    df, tab_means, tab_stds = preprocess_tabular_3d(df)

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED_3D,
        stratify=df["zone"].astype(int),
    )

    train_ds = PitchTrainDataset3D(train_df, str(TRAIN_VIDEO_DIR), tab_means, tab_stds, is_train=True)
    val_ds   = PitchTrainDataset3D(val_df,   str(TRAIN_VIDEO_DIR), tab_means, tab_stds, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE_3D,
        shuffle=True,
        num_workers=NUM_WORKERS_TRAIN_3D,
        pin_memory=(DEVICE_3D.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE_3D,
        shuffle=False,
        num_workers=NUM_WORKERS_TEST_3D,
        pin_memory=(DEVICE_3D.type == "cuda"),
    )

    model = DeepPitchModel3D()
    model.to(DEVICE_3D)

    bce = nn.BCEWithLogitsLoss()
    ce  = nn.CrossEntropyLoss()
    l1  = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_3D)
    best_score = -1.0

    for epoch in range(1, NUM_EPOCHS_3D + 1):
        model.train()
        train_loss = 0.0

        print(f"\n=== [3D] Epoch {epoch} ===")
        batch_idx = 0

        for batch in train_loader:
            t0 = time.time()
            video, tab, y_class, y_zone, y_px, y_pz, _ = batch
            t_after_load = time.time()

            video = video.to(DEVICE_3D)   # [B,T,C,H,W]
            tab   = tab.to(DEVICE_3D)
            y_class = y_class.to(DEVICE_3D)
            y_zone  = y_zone.to(DEVICE_3D)
            y_px    = y_px.to(DEVICE_3D)
            y_pz    = y_pz.to(DEVICE_3D)

            optimizer.zero_grad()
            logit_class, logits_zone, px_pred, pz_pred = model(video, tab)

            loss_class = bce(logit_class, y_class)
            loss_zone  = ce(logits_zone,  y_zone)
            loss_px    = l1(px_pred,      y_px)
            loss_pz    = l1(pz_pred,      y_pz)

            loss = 0.7 * loss_class + 0.3 * loss_zone + 0.05 * (loss_px + loss_pz)

            loss.backward()
            optimizer.step()
            if DEVICE_3D.type == "cuda":
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

        # ===== validation =====
        model.eval()
        val_loss = 0.0
        class_probs_all = []
        class_true_all = []
        zone_logits_all = []
        zone_true_all = []

        with torch.no_grad():
            for batch in val_loader:
                video, tab, y_class, y_zone, y_px, y_pz, _ = batch

                video = video.to(DEVICE_3D)
                tab   = tab.to(DEVICE_3D)
                y_class = y_class.to(DEVICE_3D)
                y_zone  = y_zone.to(DEVICE_3D)
                y_px    = y_px.to(DEVICE_3D)
                y_pz    = y_pz.to(DEVICE_3D)

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
            f"[3D] Epoch {epoch}: "
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
                DEEP_MODEL_3D_PATH,
            )
            print(f"  [3D] Saved new best model (score={best_score:.4f}) → {DEEP_MODEL_3D_PATH}")

    print("[3D] Training done. Best val score:", best_score)


if __name__ == "__main__":
    train_deep_model_3d()
