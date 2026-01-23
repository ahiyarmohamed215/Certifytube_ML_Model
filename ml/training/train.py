from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from ml.training.split import split_train_test


# =========================
# CONFIG
# =========================
DATA_PATH = Path("data/processed/sessions_features.csv")

LABEL_COL = "engagement_label"
GROUP_COL = "user_id"

ARTIFACTS_DIR = Path("ml/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

ENGAGEMENT_THRESHOLD = 0.85


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in dataset")

    print("Preprocessing data...")

    y = df[LABEL_COL].astype(int)

    DROP_COLS = [
        LABEL_COL,
        GROUP_COL,
        "session_id",
        "video_id",
        "video_title",
    ]

    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.fillna(0.0)

    feature_columns = list(X.columns)
    if not feature_columns:
        raise ValueError("No numeric feature columns found after preprocessing.")

    clean_df = X.copy()
    clean_df[LABEL_COL] = y
    if GROUP_COL and GROUP_COL in df.columns:
        clean_df[GROUP_COL] = df[GROUP_COL]

    print("Splitting train/test data...")
    train_df, test_df = split_train_test(
        clean_df,
        label_col=LABEL_COL,
        group_col=GROUP_COL,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    X_train = train_df[feature_columns].values
    y_train = train_df[LABEL_COL].values

    X_test = test_df[feature_columns].values
    y_test = test_df[LABEL_COL].values

    # -------------------------
    # Native XGBoost training (stable)
    # -------------------------
    print("Training XGBoost (native booster)...")

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_columns)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_columns)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "seed": RANDOM_STATE,
        "tree_method": "hist",
    }

    evals = [(dtrain, "train"), (dtest, "test")]

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=evals,
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    # -------------------------
    # Evaluation
    # -------------------------
    print("Evaluating model...")

    y_prob = booster.predict(dtest)  # probabilities
    y_pred_05 = (y_prob >= 0.5).astype(int)
    y_pred_085 = (y_prob >= ENGAGEMENT_THRESHOLD).astype(int)

    metrics = {
        "precision_0.5": precision_score(y_test, y_pred_05, zero_division=0),
        "recall_0.5": recall_score(y_test, y_pred_05, zero_division=0),
        "f1_0.5": f1_score(y_test, y_pred_05, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob),
        "precision_0.85": precision_score(y_test, y_pred_085, zero_division=0),
        "recall_0.85": recall_score(y_test, y_pred_085, zero_division=0),
        "f1_0.85": f1_score(y_test, y_pred_085, zero_division=0),
        "confusion_matrix_0.85": confusion_matrix(y_test, y_pred_085).tolist(),
        "best_iteration": int(getattr(booster, "best_iteration", -1)),
    }

    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # -------------------------
    # Save artifacts
    # -------------------------
    print("Saving artifacts...")

    # Save Booster (joblib works fine)
    joblib.dump(booster, ARTIFACTS_DIR / "model.joblib")

    with open(ARTIFACTS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)

    metadata = {
        "trained_at": datetime.utcnow().isoformat(),
        "model": "xgboost.Booster (native training)",
        "xgboost_version": xgb.__version__,
        "n_features": len(feature_columns),
        "label_col": LABEL_COL,
        "group_col": GROUP_COL,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "engagement_threshold": ENGAGEMENT_THRESHOLD,
        "dropped_columns": DROP_COLS,
        "metrics": metrics,
    }

    with open(ARTIFACTS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Training completed successfully.")
    print(f"Artifacts saved to: {ARTIFACTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
