# src/infer_ensemble.py

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import xgboost as xgb
from sklearn.model_selection import train_test_split

from .config import (
    TRAIN_CSV,
    TEST_FEAT_CSV,
    TRAIN_VIDEO_DIR,
    TEST_VIDEO_DIR,
    TEMPLATE_CSV,
    SUBMISSION_PATH,
    TABULAR_MODEL_CLASS_PATH,
    TABULAR_MODEL_ZONE_PATH,
    DEEP_MODEL_PATH,
    TABULAR_COLS,
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS_TEST,
    RANDOM_SEED,
)
from .dataset import PitchTrainDataset, PitchTestDataset, preprocess_tabular
from .models import DeepPitchModel
from .config import set_seed


def compute_score(y_class_true, y_zone_true, probs_class, probs_zone):
    """
    Compute the same Kaggle-like score:
        score = 0.7 * acc_class + 0.3 * acc_zone
    """
    y_class_true = y_class_true.astype(int)
    y_zone_true = y_zone_true.astype(int)

    class_pred = (probs_class > 0.5).astype(int)
    zone_pred = probs_zone.argmax(axis=1)

    acc_class = (class_pred == y_class_true).mean()
    acc_zone = (zone_pred == y_zone_true).mean()
    score = 0.7 * acc_class + 0.3 * acc_zone
    return acc_class, acc_zone, score


def tune_ensemble_weight(
    train_df: pd.DataFrame,
    tab_means,
    tab_stds,
    deep_model: DeepPitchModel,
    model_class: xgb.Booster,
    model_zone: xgb.Booster,
    w_grid=(0.3, 0.4, 0.5, 0.6, 0.7),
):
    """
    Use a validation split from TRAIN_CSV to find the best w_deep
    for blending deep + tabular predictions.
    """

    # ---- Build validation split ----
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=train_df["zone"].astype(int),
    )

    val_ds = PitchTrainDataset(
        val_df, TRAIN_VIDEO_DIR, tab_means, tab_stds, is_train=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS_TEST,
        pin_memory=(DEVICE.type == "cuda"),
    )

    deep_model.eval()

    deep_probs_class = []
    deep_logits_zone = []
    tab_probs_class = []
    tab_probs_zone = []
    y_class_true_all = []
    y_zone_true_all = []

    with torch.no_grad():
        for batch in val_loader:
            # NOTE: PitchTrainDataset returns 7 values; unpack them all
            video, tab, y_class, y_zone, y_px, y_pz, file_names = batch

            video = video.to(DEVICE)
            tab_t = tab.to(DEVICE)

            # Deep model
            logit_class, logits_zone, _, _ = deep_model(video, tab_t)
            probs_class = torch.sigmoid(logit_class).cpu().numpy()
            logits_zone_np = logits_zone.cpu().numpy()

            deep_probs_class.append(probs_class)
            deep_logits_zone.append(logits_zone_np)

            # Ground truth labels
            y_class_true_all.append(y_class.cpu().numpy())
            y_zone_true_all.append(y_zone.cpu().numpy())

            # Tabular XGBoost (CPU)
            X_tab = tab.numpy()
            dval = xgb.DMatrix(X_tab)
            probs_class_tab = model_class.predict(dval)
            probs_zone_tab = model_zone.predict(dval)

            tab_probs_class.append(probs_class_tab)
            tab_probs_zone.append(probs_zone_tab)

    # Concatenate across batches
    deep_probs_class = np.concatenate(deep_probs_class)
    deep_logits_zone = np.concatenate(deep_logits_zone)
    tab_probs_class = np.concatenate(tab_probs_class)
    tab_probs_zone = np.concatenate(tab_probs_zone)
    y_class_true_all = np.concatenate(y_class_true_all)
    y_zone_true_all = np.concatenate(y_zone_true_all)

    # Deep logits -> probabilities
    deep_probs_zone = torch.softmax(
        torch.from_numpy(deep_logits_zone), dim=1
    ).numpy()

    best_w = None
    best_score = -1.0

    print("Tuning ensemble weight w_deep on validation split...")
    for w_deep in w_grid:
        w_tab = 1.0 - w_deep

        final_probs_class = w_deep * deep_probs_class + w_tab * tab_probs_class
        final_probs_zone = w_deep * deep_probs_zone + w_tab * tab_probs_zone

        acc_c, acc_z, score = compute_score(
            y_class_true_all, y_zone_true_all,
            final_probs_class, final_probs_zone
        )
        print(
            f"  w_deep={w_deep:.2f} | "
            f"acc_class={acc_c:.4f}, acc_zone={acc_z:.4f}, score={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_w = w_deep

    print(
        f"Best ensemble weight on val: w_deep={best_w:.2f}, "
        f"score={best_score:.4f}"
    )
    return best_w


def run_ensemble_inference():
    set_seed(RANDOM_SEED)

    # ---------- Load and preprocess train df (for means/stds + tuning) ----------
    train_df = pd.read_csv(TRAIN_CSV)
    train_df, tab_means, tab_stds = preprocess_tabular(train_df)

    # ---------- Load test features ----------
    test_df = pd.read_csv(TEST_FEAT_CSV)
    test_df, _, _ = preprocess_tabular(test_df)

    # ---------- Load tabular models ----------
    model_class = xgb.Booster()
    model_class.load_model(str(TABULAR_MODEL_CLASS_PATH))

    model_zone = xgb.Booster()
    model_zone.load_model(str(TABULAR_MODEL_ZONE_PATH))

    # ---------- Load deep model ----------
    checkpoint = torch.load(DEEP_MODEL_PATH, map_location=DEVICE)
    deep_model = DeepPitchModel(tab_dim=len(TABULAR_COLS))
    deep_model.load_state_dict(checkpoint["model_state"])
    deep_model.to(DEVICE)
    deep_model.eval()

    # Prefer tab_means/stds saved with deep model if present
    tab_means = checkpoint.get("tab_means", tab_means)
    tab_stds = checkpoint.get("tab_stds", tab_stds)

    # ---------- Tune ensemble weight on a hold-out split ----------
    w_deep = tune_ensemble_weight(
        train_df, tab_means, tab_stds,
        deep_model, model_class, model_zone,
        w_grid=(0.3, 0.4, 0.5, 0.6, 0.7),
    )
    w_tab = 1.0 - w_deep
    print(f"Using w_deep={w_deep:.2f}, w_tab={w_tab:.2f} for TEST inference.\n")

    # ---------- Build test dataset ----------
    test_ds = PitchTestDataset(test_df, TEST_VIDEO_DIR, tab_means, tab_stds)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS_TEST,
        pin_memory=(DEVICE.type == "cuda"),
    )

    file_names_all = []
    deep_probs_class = []
    deep_logits_zone = []
    tab_probs_class = []
    tab_probs_zone = []

    # ---------- Inference loop over TEST ----------
    with torch.no_grad():
        for i, (video, tab, file_names) in enumerate(test_loader):
            video = video.to(DEVICE)
            tab_t = tab.to(DEVICE)

            # Deep model
            logit_class, logits_zone, _, _ = deep_model(video, tab_t)
            probs_class = torch.sigmoid(logit_class).cpu().numpy()
            logits_zone_np = logits_zone.cpu().numpy()

            deep_probs_class.append(probs_class)
            deep_logits_zone.append(logits_zone_np)
            file_names_all.extend(list(file_names))

            # Tabular models
            X_tab = tab.numpy()
            dtest = xgb.DMatrix(X_tab)
            probs_class_tab = model_class.predict(dtest)
            probs_zone_tab = model_zone.predict(dtest)

            tab_probs_class.append(probs_class_tab)
            tab_probs_zone.append(probs_zone_tab)

            if (i + 1) % 20 == 0:
                print(f"Processed { (i+1) * BATCH_SIZE } test samples...")

    deep_probs_class = np.concatenate(deep_probs_class)
    deep_logits_zone = np.concatenate(deep_logits_zone)
    tab_probs_class = np.concatenate(tab_probs_class)
    tab_probs_zone = np.concatenate(tab_probs_zone)
    file_names_all = np.array(file_names_all)

    deep_probs_zone = torch.softmax(
        torch.from_numpy(deep_logits_zone), dim=1
    ).numpy()

    # ---------- Final ensemble on TEST ----------
    final_probs_class = w_deep * deep_probs_class + w_tab * tab_probs_class
    final_probs_zone = w_deep * deep_probs_zone + w_tab * tab_probs_zone

    pitch_class_pred = np.where(final_probs_class > 0.5, "strike", "ball")
    zone_pred = final_probs_zone.argmax(axis=1) + 1  # back to 1..14

    # ---------- Build submission ----------
    sub_template = pd.read_csv(TEMPLATE_CSV)
    pred_df = pd.DataFrame(
        {
            "file_name": file_names_all,
            "pitch_class": pitch_class_pred,
            "zone": zone_pred.astype(int),
        }
    )

    # Just in case: ensure 1 row per file_name
    pred_df = pred_df.drop_duplicates(subset=["file_name"], keep="first")
    print(
        "Unique predictions:", len(pred_df),
        "| Expected:", len(sub_template)
    )

    sub = sub_template[["file_name"]].merge(
        pred_df, on="file_name", how="left"
    )
    sub.to_csv(SUBMISSION_PATH, index=False)
    print("Saved submission to:", SUBMISSION_PATH)


if __name__ == "__main__":
    run_ensemble_inference()
