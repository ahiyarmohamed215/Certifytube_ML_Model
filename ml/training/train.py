from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

from ml.training.split import split_train_test


# =========================
# CONFIG (CHANGE IF NEEDED)
# =========================
DATA_PATH = Path("data/processed/sessions_features.csv")

LABEL_COL = "label"        # <-- change if your label column name is different
GROUP_COL = "user_id"      # <-- set to None if you don't have user_id

ARTIFACTS_DIR = Path("ml/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

ENGAGEMENT_THRESHOLD = 0.85


# =========================
# TRAINING PIPELINE
# =========================
def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in dataset")

    # -------------------------
    # Basic preprocessing
    # -------------------------
    print("Preprocessing data...")

    # Separate features and label
    y = df[LABEL_COL].astype(int)
    X = df.drop(columns=[LABEL_COL])

    # Remove group column from features if present
    if GROUP_COL and GROUP_COL in X.columns:
        X = X.drop(columns=[GROUP_COL])

    # Handle missing values
    # Rule:
    # - counts / seconds / ratios: fill with 0
    X = X.fillna(0.0)

    feature_columns = list(X.columns)

    # Rebuild dataframe for splitting
    clean_df = X.copy()
    clean_df[LABEL_COL] = y
    if GROUP_COL and GROUP_COL in df.columns:
        clean_df[GROUP_COL] = df[GROUP_COL]

    # -------------------------
    # Train / test split
    # -------------------------
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
    # Model definition
    # -------------------------
    print("Training XGBoost model...")

    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
        early_stopping_rounds=30,
    )

    # -------------------------
    # Evaluation
    # -------------------------
    print("Evaluating model...")

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred_05 = (y_prob >= 0.5).astype(int)
    y_pred_085 = (y_prob >= ENGAGEMENT_THRESHOLD).astype(int)

    metrics = {
        "precision_0.5": precision_score(y_test, y_pred_05),
        "recall_0.5": recall_score(y_test, y_pred_05),
        "f1_0.5": f1_score(y_test, y_pred_05),
        "auc": roc_auc_score(y_test, y_prob),
        "precision_0.85": precision_score(y_test, y_pred_085),
        "recall_0.85": recall_score(y_test, y_pred_085),
        "f1_0.85": f1_score(y_test, y_pred_085),
        "confusion_matrix_0.85": confusion_matrix(y_test, y_pred_085).tolist(),
    }

    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # -------------------------
    # Save artifacts
    # -------------------------
    print("Saving artifacts...")

    joblib.dump(model, ARTIFACTS_DIR / "model.joblib")

    with open(ARTIFACTS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)

    metadata = {
        "trained_at": datetime.utcnow().isoformat(),
        "model": "XGBoostClassifier",
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "label_col": LABEL_COL,
        "group_col": GROUP_COL,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "engagement_threshold": ENGAGEMENT_THRESHOLD,
        "metrics": metrics,
    }

    with open(ARTIFACTS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Training completed successfully.")
    print(f"Artifacts saved to: {ARTIFACTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
