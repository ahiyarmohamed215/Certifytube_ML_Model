from __future__ import annotations

import json
from pathlib import Path
from xml.parsers.expat import model

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import xgboost as xgb


from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

# =========================
# CONFIG
# =========================
DATA_PATH = Path("data/processed/sessions_features.csv")
ARTIFACTS_DIR = Path("ml/artifacts")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "engagement_label"     # must match train.py
GROUP_COL = "user_id"    # same as train.py
ENGAGEMENT_THRESHOLD = 0.85


def main():
    print("Loading artifacts...")
    model = joblib.load(ARTIFACTS_DIR / "model.joblib")

    with open(ARTIFACTS_DIR / "feature_columns.json") as f:
        feature_columns = json.load(f)

    with open(ARTIFACTS_DIR / "metadata.json") as f:
        metadata = json.load(f)

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # Separate
    y = df[LABEL_COL].astype(int)
    X = df[feature_columns].fillna(0.0).values

    print("Running predictions...")
    dmat = xgb.DMatrix(X, feature_names=feature_columns)
    y_prob = model.predict(dmat)
    y_pred_05 = (y_prob >= 0.5).astype(int)
    y_pred_085 = (y_prob >= ENGAGEMENT_THRESHOLD).astype(int)

    # Metrics
    metrics = {
        "precision_0.5": precision_score(y, y_pred_05),
        "recall_0.5": recall_score(y, y_pred_05),
        "f1_0.5": f1_score(y, y_pred_05),
        "auc": roc_auc_score(y, y_prob),
        "precision_0.85": precision_score(y, y_pred_085),
        "recall_0.85": recall_score(y, y_pred_085),
        "f1_0.85": f1_score(y, y_pred_085),
        "confusion_matrix_0.85": confusion_matrix(y, y_pred_085).tolist(),
    }

    print("Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Save metrics
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # -------------------------
    # Confusion Matrix (0.85)
    # -------------------------
    cm = confusion_matrix(y, y_pred_085)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix (Threshold = 0.85)")
    plt.savefig(REPORTS_DIR / "confusion_matrix_085.png", bbox_inches="tight")
    plt.close()

    # -------------------------
    # ROC Curve
    # -------------------------
    RocCurveDisplay.from_predictions(y, y_prob)
    plt.title("ROC Curve – Engagement Model")
    plt.savefig(REPORTS_DIR / "roc_curve.png", bbox_inches="tight")
    plt.close()

    print(f"Reports saved to: {REPORTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
