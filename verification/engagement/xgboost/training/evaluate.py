from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from verification.engagement.common.dataset import load_csv_with_header_fallback
from verification.engagement.common.preprocessing import load_preprocessing_artifact, transform_numeric_frame

DATA_PATH = Path("data/processed/sessions_features.csv")
ARTIFACTS_DIR = Path("verification/engagement/xgboost/artifacts")
REPORTS_DIR = Path("reports/xgboost")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "engagement_label"
ENGAGEMENT_THRESHOLD = 0.85


def _load_eval_model():
    model_eval_path = ARTIFACTS_DIR / "model_eval.joblib"
    model_prod_path = ARTIFACTS_DIR / "model.joblib"

    if model_eval_path.exists():
        return joblib.load(model_eval_path)
    if model_prod_path.exists():
        return joblib.load(model_prod_path)
    raise FileNotFoundError("Neither model_eval.joblib nor model.joblib was found.")


def _load_eval_preprocessing():
    eval_path = ARTIFACTS_DIR / "preprocessing_eval.json"
    prod_path = ARTIFACTS_DIR / "preprocessing.json"

    if eval_path.exists():
        return load_preprocessing_artifact(eval_path)
    if prod_path.exists():
        return load_preprocessing_artifact(prod_path)
    raise FileNotFoundError("No preprocessing artifact found.")


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred_05 = (y_prob >= 0.5).astype(int)
    y_pred_085 = (y_prob >= ENGAGEMENT_THRESHOLD).astype(int)

    return {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "auc_pr": float(average_precision_score(y_true, y_prob)),
        "accuracy_0.5": float(accuracy_score(y_true, y_pred_05)),
        "precision_0.5": float(precision_score(y_true, y_pred_05, zero_division=0)),
        "recall_0.5": float(recall_score(y_true, y_pred_05, zero_division=0)),
        "f1_0.5": float(f1_score(y_true, y_pred_05, zero_division=0)),
        "accuracy_0.85": float(accuracy_score(y_true, y_pred_085)),
        "precision_0.85": float(precision_score(y_true, y_pred_085, zero_division=0)),
        "recall_0.85": float(recall_score(y_true, y_pred_085, zero_division=0)),
        "f1_0.85": float(f1_score(y_true, y_pred_085, zero_division=0)),
        "confusion_matrix_0.85": confusion_matrix(y_true, y_pred_085).tolist(),
        "threshold_used": ENGAGEMENT_THRESHOLD,
    }


def _build_predictions_frame(test_df: pd.DataFrame, y_prob: np.ndarray) -> pd.DataFrame:
    predictions = test_df[["session_id", "user_id", LABEL_COL]].copy()
    predictions["y_prob"] = y_prob
    predictions["y_pred_0.5"] = (y_prob >= 0.5).astype(int)
    predictions["y_pred_0.85"] = (y_prob >= ENGAGEMENT_THRESHOLD).astype(int)
    predictions["error_type_0.85"] = np.where(
        predictions[LABEL_COL] == predictions["y_pred_0.85"],
        "correct",
        np.where(predictions["y_pred_0.85"] == 1, "false_positive", "false_negative"),
    )
    return predictions


def main():
    print("Loading artifacts...")
    model = _load_eval_model()
    preprocessing = _load_eval_preprocessing()

    with open(ARTIFACTS_DIR / "split.json") as f:
        split_info = json.load(f)

    test_session_ids = set(split_info["test_session_ids"])

    print("Loading dataset...")
    df = load_csv_with_header_fallback(DATA_PATH, required_columns=[LABEL_COL, "session_id"])

    if "session_id" not in df.columns:
        raise ValueError("sessions_features.csv must contain session_id for split filtering.")

    df["session_id"] = df["session_id"].astype(str)
    if "user_id" in df.columns:
        df["user_id"] = df["user_id"].astype(str)
    test_df = df[df["session_id"].isin(test_session_ids)].copy()

    if test_df.empty:
        raise ValueError("Test split is empty after filtering. Check split.json and sessions_features.csv.")

    y = test_df[LABEL_COL].astype(int).values
    x_df = transform_numeric_frame(test_df, preprocessing)

    print(f"Evaluating on test size: {len(test_df)}")

    dmat = xgb.DMatrix(x_df.values, feature_names=preprocessing.feature_columns)
    y_prob = model.predict(dmat)

    metrics = _compute_metrics(y, y_prob)

    print("Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    with open(REPORTS_DIR / "metrics_test_recomputed.json", "w") as f:
        json.dump(metrics, f, indent=2)

    predictions_df = _build_predictions_frame(test_df, y_prob)
    predictions_df.to_csv(REPORTS_DIR / "test_predictions_recomputed.csv", index=False)
    predictions_df[predictions_df["error_type_0.85"] != "correct"].to_csv(
        REPORTS_DIR / "test_misclassifications_recomputed.csv",
        index=False,
    )

    cm = confusion_matrix(y, (y_prob >= ENGAGEMENT_THRESHOLD).astype(int))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"Confusion Matrix (Threshold = {ENGAGEMENT_THRESHOLD}) - Test Split")
    plt.savefig(REPORTS_DIR / "confusion_matrix_085.png", bbox_inches="tight")
    plt.close()

    RocCurveDisplay.from_predictions(y, y_prob)
    plt.title("ROC Curve - Test Split")
    plt.savefig(REPORTS_DIR / "roc_curve.png", bbox_inches="tight")
    plt.close()

    PrecisionRecallDisplay.from_predictions(y, y_prob)
    plt.title("Precision-Recall Curve - Test Split")
    plt.savefig(REPORTS_DIR / "pr_curve.png", bbox_inches="tight")
    plt.close()

    CalibrationDisplay.from_predictions(y, y_prob, n_bins=10, strategy="quantile")
    plt.title("Calibration Plot - Test Split")
    plt.savefig(REPORTS_DIR / "calibration.png", bbox_inches="tight")
    plt.close()

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
        plt.savefig(REPORTS_DIR / "feature_importance_gain_eval.png", bbox_inches="tight")
        plt.close()

    thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(y, y_pred)),
                "precision": float(precision_score(y, y_pred, zero_division=0)),
                "recall": float(recall_score(y, y_pred, zero_division=0)),
                "f1": float(f1_score(y, y_pred, zero_division=0)),
            }
        )

    sweep = pd.DataFrame(rows)
    sweep.to_csv(REPORTS_DIR / "threshold_sweep.csv", index=False)

    plt.figure()
    plt.plot(sweep["threshold"], sweep["accuracy"], label="Accuracy")
    plt.plot(sweep["threshold"], sweep["f1"], label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Accuracy and F1 vs Threshold - Test Split")
    plt.legend()
    plt.savefig(REPORTS_DIR / "accuracy_f1_vs_threshold.png", bbox_inches="tight")
    plt.close()

    print(f"Reports saved to: {REPORTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
