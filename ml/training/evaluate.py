from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.calibration import CalibrationDisplay


DATA_PATH = Path("data/processed/sessions_features.csv")
ARTIFACTS_DIR = Path("ml/artifacts")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "engagement_label"
ENGAGEMENT_THRESHOLD = 0.85


def main():
    print("Loading artifacts...")
    model = joblib.load(ARTIFACTS_DIR / "model.joblib")

    with open(ARTIFACTS_DIR / "feature_columns.json") as f:
        feature_columns = json.load(f)

    with open(ARTIFACTS_DIR / "split.json") as f:
        split_info = json.load(f)

    test_session_ids = set(split_info["test_session_ids"])

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    if "session_id" not in df.columns:
        raise ValueError("sessions_features.csv must contain session_id for split filtering.")

    df["session_id"] = df["session_id"].astype(str)
    test_df = df[df["session_id"].isin(test_session_ids)].copy()

    if test_df.empty:
        raise ValueError("Test split is empty after filtering. Check split.json and sessions_features.csv.")

    y = test_df[LABEL_COL].astype(int).values
    X = test_df[feature_columns].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).values

    print(f"Evaluating on test size: {len(test_df)}")

    dmat = xgb.DMatrix(X, feature_names=feature_columns)
    y_prob = model.predict(dmat)

    y_pred_05 = (y_prob >= 0.5).astype(int)
    y_pred_085 = (y_prob >= ENGAGEMENT_THRESHOLD).astype(int)

    metrics = {
        "auc_roc": float(roc_auc_score(y, y_prob)),
        "auc_pr": float(average_precision_score(y, y_prob)),

        "accuracy_0.5": float(accuracy_score(y, y_pred_05)),
        "precision_0.5": float(precision_score(y, y_pred_05, zero_division=0)),
        "recall_0.5": float(recall_score(y, y_pred_05, zero_division=0)),
        "f1_0.5": float(f1_score(y, y_pred_05, zero_division=0)),

        "accuracy_0.85": float(accuracy_score(y, y_pred_085)),
        "precision_0.85": float(precision_score(y, y_pred_085, zero_division=0)),
        "recall_0.85": float(recall_score(y, y_pred_085, zero_division=0)),
        "f1_0.85": float(f1_score(y, y_pred_085, zero_division=0)),

        "confusion_matrix_0.85": confusion_matrix(y, y_pred_085).tolist(),
        "threshold_used": ENGAGEMENT_THRESHOLD,
    }

    print("Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    with open(REPORTS_DIR / "metrics_test.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Confusion Matrix ---
    cm = confusion_matrix(y, y_pred_085)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"Confusion Matrix (Threshold = {ENGAGEMENT_THRESHOLD}) – Test Split")
    plt.savefig(REPORTS_DIR / "confusion_matrix_085.png", bbox_inches="tight")
    plt.close()

    # --- ROC Curve ---
    RocCurveDisplay.from_predictions(y, y_prob)
    plt.title("ROC Curve – Test Split")
    plt.savefig(REPORTS_DIR / "roc_curve.png", bbox_inches="tight")
    plt.close()

    # --- PR Curve ---
    PrecisionRecallDisplay.from_predictions(y, y_prob)
    plt.title("Precision–Recall Curve – Test Split")
    plt.savefig(REPORTS_DIR / "pr_curve.png", bbox_inches="tight")
    plt.close()

    # --- Calibration plot ---
    CalibrationDisplay.from_predictions(y, y_prob, n_bins=10, strategy="quantile")
    plt.title("Calibration Plot – Test Split")
    plt.savefig(REPORTS_DIR / "calibration.png", bbox_inches="tight")
    plt.close()

    # --- Feature importance plot (gain) ---
    score = model.get_score(importance_type="gain")
    if score:
        imp = (
            pd.DataFrame({"feature": list(score.keys()), "gain": list(score.values())})
            .sort_values("gain", ascending=False)
            .head(20)
        )
        imp.plot(kind="barh", x="feature", y="gain", legend=False)
        plt.gca().invert_yaxis()
        plt.title("Top-20 Feature Importance (Gain)")
        plt.savefig(REPORTS_DIR / "feature_importance_gain.png", bbox_inches="tight")
        plt.close()

    # --- Accuracy / F1 vs Threshold (policy + gaming discussion) ---
    thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "accuracy": float(accuracy_score(y, yp)),
            "precision": float(precision_score(y, yp, zero_division=0)),
            "recall": float(recall_score(y, yp, zero_division=0)),
            "f1": float(f1_score(y, yp, zero_division=0)),
        })

    sweep = pd.DataFrame(rows)
    sweep.to_csv(REPORTS_DIR / "threshold_sweep.csv", index=False)

    plt.figure()
    plt.plot(sweep["threshold"], sweep["accuracy"], label="Accuracy")
    plt.plot(sweep["threshold"], sweep["f1"], label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Accuracy and F1 vs Threshold – Test Split")
    plt.legend()
    plt.savefig(REPORTS_DIR / "accuracy_f1_vs_threshold.png", bbox_inches="tight")
    plt.close()

    print(f"Reports saved to: {REPORTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
