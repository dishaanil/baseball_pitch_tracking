import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import xgboost as xgb

from .config_3d import (
    TEST_VIDEO_DIR, TRAIN_CSV, TEST_FEAT_CSV,
    TEMPLATE_CSV,
    TABULAR_MODEL_CLASS_3D_PATH, TABULAR_MODEL_ZONE_3D_PATH,
    DEEP_MODEL_3D_PATH, SUBMISSION_3D_PATH,
    TABULAR_COLS,
    BATCH_SIZE_3D, NUM_WORKERS_TEST_3D, DEVICE_3D,
)
from .dataset_3d import PitchTestDataset3D, preprocess_tabular_3d
from .models_3d import DeepPitchModel3D


def run_ensemble_inference_3d():
    train_df = pd.read_csv(TRAIN_CSV)
    train_df, tab_means, tab_stds = preprocess_tabular_3d(train_df)

    test_df = pd.read_csv(TEST_FEAT_CSV)
    test_df, _, _ = preprocess_tabular_3d(test_df)

    model_class = xgb.Booster()
    model_class.load_model(str(TABULAR_MODEL_CLASS_3D_PATH))
    model_zone = xgb.Booster()
    model_zone.load_model(str(TABULAR_MODEL_ZONE_3D_PATH))

    checkpoint = torch.load(DEEP_MODEL_3D_PATH, map_location=DEVICE_3D)
    deep_model = DeepPitchModel3D()
    deep_model.load_state_dict(checkpoint["model_state"])
    deep_model.to(DEVICE_3D)
    deep_model.eval()

    tab_means = checkpoint.get("tab_means", tab_means)
    tab_stds  = checkpoint.get("tab_stds", tab_stds)

    test_ds = PitchTestDataset3D(test_df, str(TEST_VIDEO_DIR), tab_means, tab_stds)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE_3D,
        shuffle=False,
        num_workers=NUM_WORKERS_TEST_3D,
        pin_memory=(DEVICE_3D.type == "cuda"),
    )

    file_names_all = []
    deep_probs_class = []
    deep_logits_zone = []
    tab_probs_class = []
    tab_probs_zone = []

    with torch.no_grad():
        for video, tab, file_names in test_loader:
            video = video.to(DEVICE_3D)
            tab_t = tab.to(DEVICE_3D)

            logit_class, logits_zone, _, _ = deep_model(video, tab_t)
            probs_class = torch.sigmoid(logit_class).cpu().numpy()
            logits_zone_np = logits_zone.cpu().numpy()

            deep_probs_class.append(probs_class)
            deep_logits_zone.append(logits_zone_np)
            file_names_all.extend(list(file_names))

            X_tab = tab.numpy()
            dtest = xgb.DMatrix(X_tab)
            probs_class_tab = model_class.predict(dtest)
            probs_zone_tab  = model_zone.predict(dtest)

            tab_probs_class.append(probs_class_tab)
            tab_probs_zone.append(probs_zone_tab)

    deep_probs_class = np.concatenate(deep_probs_class)
    deep_logits_zone = np.concatenate(deep_logits_zone)
    tab_probs_class  = np.concatenate(tab_probs_class)
    tab_probs_zone   = np.concatenate(tab_probs_zone)
    file_names_all   = np.array(file_names_all)

    deep_probs_zone = torch.softmax(torch.from_numpy(deep_logits_zone), dim=1).numpy()

    w_deep = 0.4
    w_tab  = 0.6

    final_probs_class = w_deep * deep_probs_class + w_tab * tab_probs_class
    final_probs_zone  = w_deep * deep_probs_zone  + w_tab * tab_probs_zone

    pitch_class_pred = np.where(final_probs_class > 0.6, "strike", "ball")
    zone_pred        = final_probs_zone.argmax(axis=1) + 1  

    pred_df = pd.DataFrame({
        "file_name": file_names_all,
        "pitch_class": pitch_class_pred,
        "zone": zone_pred.astype(int),
    })

    pred_df = pred_df.drop_duplicates(subset=["file_name"], keep="first")

    sub_template = pd.read_csv(TEMPLATE_CSV)
    print("Unique predictions:", len(pred_df), "Expected:", len(sub_template))

    sub = sub_template[["file_name"]].merge(pred_df, on="file_name", how="left")
    sub.to_csv(SUBMISSION_3D_PATH, index=False)
    print("Saved 3D submission to:", SUBMISSION_3D_PATH)


if __name__ == "__main__":
    run_ensemble_inference_3d()
