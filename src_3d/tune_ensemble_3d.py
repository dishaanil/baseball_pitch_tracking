import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import xgboost as xgb
from sklearn.model_selection import train_test_split

from .config_3d import (
    TRAIN_VIDEO_DIR,
    TRAIN_CSV,
    TABULAR_COLS,
    TABULAR_MODEL_CLASS_3D_PATH,
    TABULAR_MODEL_ZONE_3D_PATH,
    DEEP_MODEL_3D_PATH,
    BATCH_SIZE_3D,
    NUM_WORKERS_TEST_3D,
    DEVICE_3D,
    RANDOM_SEED_3D,
    set_seed_3d,
)
from .dataset_3d import PitchTrainDataset3D, preprocess_tabular_3d
from .models_3d import DeepPitchModel3D


def compute_score(y_class_true, y_class_prob, y_zone_true, y_zone_prob, thr: float):
    y_class_pred = (y_class_prob > thr).astype(int)
    acc_class = (y_class_pred == y_class_true).mean()

    y_zone_pred = y_zone_prob.argmax(axis=1)
    acc_zone = (y_zone_pred == y_zone_true).mean()

    score = 0.7 * acc_class + 0.3 * acc_zone
    return acc_class, acc_zone, score


def collect_val_outputs():
    set_seed_3d(RANDOM_SEED_3D)

    df = pd.read_csv(TRAIN_CSV)
    df, tab_means, tab_stds = preprocess_tabular_3d(df)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_SEED_3D,
        stratify=df["zone"].astype(int),
    )

    val_ds = PitchTrainDataset3D(
        val_df,
        str(TRAIN_VIDEO_DIR),
        tab_means,
        tab_stds,
        is_train=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE_3D,
        shuffle=False,
        num_workers=NUM_WORKERS_TEST_3D,
        pin_memory=(DEVICE_3D.type == "cuda"),
    )

    checkpoint = torch.load(DEEP_MODEL_3D_PATH, map_location=DEVICE_3D)
    deep_model = DeepPitchModel3D()
    deep_model.load_state_dict(checkpoint["model_state"])
    deep_model.to(DEVICE_3D)
    deep_model.eval()

    deep_class_dict = {}
    deep_zone_dict = {}

    with torch.no_grad():
        for video, tab, y_class, y_zone, y_px, y_pz, file_names in val_loader:
            video = video.to(DEVICE_3D)
            tab_t = tab.to(DEVICE_3D)

            logit_class, logits_zone, _, _ = deep_model(video, tab_t)
            probs_class = torch.sigmoid(logit_class).cpu().numpy()  
            logits_zone_np = logits_zone.cpu().numpy()               

            for i, fn in enumerate(file_names):
                deep_class_dict[fn] = probs_class[i]
                deep_zone_dict[fn] = logits_zone_np[i]

    model_class = xgb.Booster()
    model_class.load_model(str(TABULAR_MODEL_CLASS_3D_PATH))
    model_zone = xgb.Booster()
    model_zone.load_model(str(TABULAR_MODEL_ZONE_3D_PATH))

    X_val = val_df[TABULAR_COLS].values.astype("float32")
    y_class_true = (val_df["pitch_class"] == "strike").astype("int32").values
    y_zone_true = (val_df["zone"].astype(int) - 1).astype("int32").values

    dval = xgb.DMatrix(X_val)
    probs_class_tab = model_class.predict(dval)   
    probs_zone_tab = model_zone.predict(dval)     

    tab_class_dict = {}
    tab_zone_dict = {}
    for i, fn in enumerate(val_df["file_name"]):
        tab_class_dict[fn] = probs_class_tab[i]
        tab_zone_dict[fn] = probs_zone_tab[i]

    file_names = list(val_df["file_name"])

    deep_probs_class = np.array([deep_class_dict[fn] for fn in file_names])
    deep_zone_logits = np.stack([deep_zone_dict[fn] for fn in file_names], axis=0)

    tab_probs_class = np.array([tab_class_dict[fn] for fn in file_names])
    tab_probs_zone = np.stack([tab_zone_dict[fn] for fn in file_names], axis=0)

    deep_probs_zone = torch.softmax(
        torch.from_numpy(deep_zone_logits), dim=1
    ).numpy()

    return (
        file_names,
        y_class_true,
        y_zone_true,
        deep_probs_class,
        deep_probs_zone,
        tab_probs_class,
        tab_probs_zone,
    )


def tune_ensemble():
    (
        file_names,
        y_class_true,
        y_zone_true,
        deep_probs_class,
        deep_probs_zone,
        tab_probs_class,
        tab_probs_zone,
    ) = collect_val_outputs()

    print(f"Validation samples: {len(file_names)}")

    w_list = [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    thr_list = [0.40, 0.45, 0.50, 0.55, 0.60]

    results = []

    for w_deep in w_list:
        w_tab = 1.0 - w_deep

        final_probs_class = w_deep * deep_probs_class + w_tab * tab_probs_class
        final_probs_zone = w_deep * deep_probs_zone + w_tab * tab_probs_zone

        for thr in thr_list:
            acc_c, acc_z, score = compute_score(
                y_class_true,
                final_probs_class,
                y_zone_true,
                final_probs_zone,
                thr,
            )
            results.append(
                {
                    "w_deep": w_deep,
                    "w_tab": w_tab,
                    "threshold": thr,
                    "acc_class": acc_c,
                    "acc_zone": acc_z,
                    "score": score,
                }
            )

    results_df = pd.DataFrame(results).sort_values("score", ascending=False)
    print("\nTop 10 configs on validation:")
    print(results_df.head(10).to_string(index=False))

    best = results_df.iloc[0]
    print("\nBest config:")
    print(
        f"  w_deep={best['w_deep']:.2f}, "
        f"w_tab={best['w_tab']:.2f}, "
        f"threshold={best['threshold']:.2f}, "
        f"score={best['score']:.4f}, "
        f"acc_class={best['acc_class']:.4f}, "
        f"acc_zone={best['acc_zone']:.4f}"
    )


if __name__ == "__main__":
    tune_ensemble()
