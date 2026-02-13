"""
evaluate_ebm.py – Full evaluation suite for the EBM model.

Mirrors evaluate.py but with EBM-specific additions:
  - Global term importances
  - Per-feature shape function plots
  - Standard: ROC, PR, calibration, confusion matrix, threshold sweep
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

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


def _plot_shape_functions(ebm, out_dir: Path, max_plots: int = 12) -> None:
    """
    Plot top-N EBM shape functions (partial-dependence-like curves).
    These are the glass-box internals: y = Σ f_i(x_i).
    Only 1D main-effect terms are plotted; interaction terms are skipped.
    """
    importances = ebm.term_importances()

    # Collect only plottable 1D main-effect terms
    plottable = []
    for term_idx in range(len(importances)):
        scores = ebm.term_scores_[term_idx]
        if scores.ndim != 1:
            continue  # skip 2D interaction terms
        bins_list = ebm.bins_[term_idx]
        if not bins_list or len(bins_list) == 0:
            continue
        plottable.append((term_idx, importances[term_idx]))

    if not plottable:
        print("  [!] No plottable 1D shape functions found.")
        return

    # Sort by importance descending, take top N
    plottable.sort(key=lambda x: x[1], reverse=True)
    plottable = plottable[:max_plots]
    n_plots = len(plottable)

    fig, axes = plt.subplots(
        nrows=(n_plots + 2) // 3,
        ncols=3,
        figsize=(18, 4 * ((n_plots + 2) // 3)),
    )
    axes = axes.flatten() if n_plots > 1 else [axes]

    for idx, (term_idx, imp) in enumerate(plottable):
        ax = axes[idx]
        term_name = ebm.term_names_[term_idx]
        bins = ebm.bins_[term_idx][0]
        scores = ebm.term_scores_[term_idx]

        # bins: cut-point edges; scores may include extra entries for
        # missing/unknown. Build x from bins and trim scores to match.
        x_vals = np.concatenate([[bins[0] - 1], bins])
        y_vals = scores[: len(x_vals)]

        ax.step(x_vals, y_vals, where="post", linewidth=1.5)
        ax.set_xlabel(term_name)
        ax.set_ylabel("Score contribution")
        ax.set_title(f"{term_name}\n(imp={imp:.3f})")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    # Hide unused axes
    for j in range(len(plottable), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("EBM Shape Functions (Top Features)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / "ebm_shape_functions.png", bbox_inches="tight", dpi=150)
    plt.close()


def main():
    print("=" * 60)
    print("EBM Evaluation Pipeline")
    print("=" * 60)

    # ---- Load artifacts ----
    print("\nLoading EBM artifacts...")
    ebm = joblib.load(ARTIFACTS_DIR / "ebm_model.joblib")

    with open(ARTIFACTS_DIR / "ebm_feature_columns.json") as f:
        feature_columns = json.load(f)

    with open(ARTIFACTS_DIR / "ebm_split.json") as f:
        split_info = json.load(f)

    test_session_ids = set(split_info["test_session_ids"])

    # ---- Load dataset ----
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    if "session_id" not in df.columns:
        raise ValueError("sessions_features.csv must contain session_id for split filtering.")

    df["session_id"] = df["session_id"].astype(str)
    test_df = df[df["session_id"].isin(test_session_ids)].copy()

    if test_df.empty:
        raise ValueError("Test split is empty. Check ebm_split.json and sessions_features.csv.")

    y = test_df[LABEL_COL].astype(int).values
    X = test_df[feature_columns].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).values

    print(f"Evaluating on test size: {len(test_df)}")

    # ---- Predict ----
    y_prob = ebm.predict_proba(X)[:, 1]
    y_pred_05 = (y_prob >= 0.5).astype(int)
    y_pred_085 = (y_prob >= ENGAGEMENT_THRESHOLD).astype(int)

    metrics = {
        "model": "EBM",
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

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    with open(REPORTS_DIR / "ebm_metrics_test.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Confusion Matrix ---
    cm = confusion_matrix(y, y_pred_085)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"EBM – Confusion Matrix (Threshold = {ENGAGEMENT_THRESHOLD})")
    plt.savefig(REPORTS_DIR / "ebm_confusion_matrix_085.png", bbox_inches="tight")
    plt.close()

    # --- ROC Curve ---
    RocCurveDisplay.from_predictions(y, y_prob)
    plt.title("EBM – ROC Curve")
    plt.savefig(REPORTS_DIR / "ebm_roc_curve.png", bbox_inches="tight")
    plt.close()

    # --- PR Curve ---
    PrecisionRecallDisplay.from_predictions(y, y_prob)
    plt.title("EBM – Precision–Recall Curve")
    plt.savefig(REPORTS_DIR / "ebm_pr_curve.png", bbox_inches="tight")
    plt.close()

    # --- Calibration plot ---
    CalibrationDisplay.from_predictions(y, y_prob, n_bins=10, strategy="quantile")
    plt.title("EBM – Calibration Plot")
    plt.savefig(REPORTS_DIR / "ebm_calibration.png", bbox_inches="tight")
    plt.close()

    # --- Global importance ---
    importances = ebm.term_importances()
    names = ebm.term_names_
    imp_df = (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
    )
    imp_df.plot(kind="barh", x="feature", y="importance", legend=False)
    plt.gca().invert_yaxis()
    plt.title("EBM – Top-20 Term Importances")
    plt.savefig(REPORTS_DIR / "ebm_feature_importance.png", bbox_inches="tight")
    plt.close()

    # --- Shape function plots ---
    try:
        _plot_shape_functions(ebm, REPORTS_DIR, max_plots=12)
    except Exception as e:
        print(f"  [!] Shape function plots skipped: {e}")

    # --- Threshold sweep ---
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
    sweep.to_csv(REPORTS_DIR / "ebm_threshold_sweep.csv", index=False)

    plt.figure()
    plt.plot(sweep["threshold"], sweep["accuracy"], label="Accuracy")
    plt.plot(sweep["threshold"], sweep["f1"], label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("EBM – Accuracy and F1 vs Threshold")
    plt.legend()
    plt.savefig(REPORTS_DIR / "ebm_accuracy_f1_vs_threshold.png", bbox_inches="tight")
    plt.close()

    print(f"\n✓ EBM evaluation complete. Reports: {REPORTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
