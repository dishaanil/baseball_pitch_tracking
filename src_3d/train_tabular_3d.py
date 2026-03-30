import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from .config_3d import (
    TRAIN_CSV,
    TABULAR_COLS,
    RANDOM_SEED_3D,
    TABULAR_MODEL_CLASS_3D_PATH,
    TABULAR_MODEL_ZONE_3D_PATH,
)
from .dataset_3d import preprocess_tabular_3d


def train_tabular_models_3d():
    df = pd.read_csv(TRAIN_CSV)
    df, _, _ = preprocess_tabular_3d(df)

    X = df[TABULAR_COLS].values.astype("float32")
    y_class = (df["pitch_class"] == "strike").astype("int32").values
    y_zone  = (df["zone"].astype(int) - 1).astype("int32").values

    X_train, X_val, y_train_c, y_val_c, y_train_z, y_val_z = train_test_split(
        X, y_class, y_zone,
        test_size=0.2,
        random_state=RANDOM_SEED_3D,
        stratify=y_zone,
    )

    dtrain_c = xgb.DMatrix(X_train, label=y_train_c)
    dval_c   = xgb.DMatrix(X_val,   label=y_val_c)

    params_class = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": RANDOM_SEED_3D,
    }

    model_class = xgb.train(
        params_class,
        dtrain_c,
        num_boost_round=500,
        evals=[(dtrain_c, "train"), (dval_c, "val")],
        early_stopping_rounds=30,
        verbose_eval=50,
    )

    preds_c = (model_class.predict(dval_c) > 0.5).astype("int32")
    acc_c = accuracy_score(y_val_c, preds_c)
    print(f"[3D] Tabular pitch_class accuracy: {acc_c:.4f}")
    model_class.save_model(str(TABULAR_MODEL_CLASS_3D_PATH))
    print("[3D] Saved tabular class model to:", TABULAR_MODEL_CLASS_3D_PATH)

    dtrain_z = xgb.DMatrix(X_train, label=y_train_z)
    dval_z   = xgb.DMatrix(X_val,   label=y_val_z)

    params_zone = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": 14,
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": RANDOM_SEED_3D,
    }

    model_zone = xgb.train(
        params_zone,
        dtrain_z,
        num_boost_round=600,
        evals=[(dtrain_z, "train"), (dval_z, "val")],
        early_stopping_rounds=40,
        verbose_eval=50,
    )

    preds_z = model_zone.predict(dval_z).argmax(axis=1)
    acc_z = accuracy_score(y_val_z, preds_z)
    print(f"[3D] Tabular zone accuracy: {acc_z:.4f}")
    model_zone.save_model(str(TABULAR_MODEL_ZONE_3D_PATH))
    print("[3D] Saved tabular zone model to:", TABULAR_MODEL_ZONE_3D_PATH)


if __name__ == "__main__":
    train_tabular_models_3d()
