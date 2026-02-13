"""
train_ebm.py – Production-grade EBM training pipeline.

Mirrors the XGBoost pipeline in train.py but uses InterpretML's
ExplainableBoostingClassifier (glass-box GAM).

Training strategy:
  1. Same data prep / stratified-grouped split as XGBoost
  2. 40-trial random search (same approach as XGBoost) scored
     by mean CV AUC via StratifiedGroupKFold
  3. outer_bags=1 + max_rounds=500 for fast CV evaluation
  4. Final model retrained on full train split with best params
     (max_rounds=5000 + validation_size=0.15 for early stopping)
  5. Learning curves, calibration, and artifact dump
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold

from interpret.glassbox import ExplainableBoostingClassifier

from ml.training.split import split_train_test

# ===========================
# CONFIG
# ===========================
DATA_PATH = Path("data/processed/sessions_features.csv")

LABEL_COL = "engagement_label"
GROUP_COL = "user_id"

ARTIFACTS_DIR = Path("ml/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

ENGAGEMENT_THRESHOLD = 0.85

DROP_COLS = [
    LABEL_COL,
    GROUP_COL,
    "session_id",
    "video_id",
    "video_title",
]

# ===========================
# HYPERPARAMETER SEARCH SPACE
# ===========================
# Production approach: 40-trial random search (same as the XGBoost pipeline).
# EBM is more expensive per trial than XGBoost, so random search gives
# better coverage-per-minute than exhaustive grid.
# Final model uses max_rounds=5000 + early stopping; CV uses max_rounds=500.

PARAM_SPACE = {
    "max_bins":           [128, 256, 512],
    "learning_rate":      [0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
    "max_leaves":         [2, 3, 4, 5, 6],
    "min_samples_leaf":   [2, 4, 5, 8, 10, 15],
    "interactions":       [0, 3, 5, 8, 10],
    "max_interaction_bins": [16, 32, 64],
}

N_RANDOM_TRIALS = 40  # matches XGBoost's 40-trial random search


def _prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """Identical to XGBoost pipeline — single source of truth would be ideal."""
    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing label column: {LABEL_COL}")

    y = df[LABEL_COL].astype(int).values

    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()

    # Median impute
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    feature_columns = list(X.columns)
    if not feature_columns:
        raise ValueError("No numeric features found after preprocessing.")

    return X.values, y, feature_columns, X


def _make_cv_folds(
    df_train: pd.DataFrame,
    feature_columns: List[str],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y = df_train[LABEL_COL].astype(int).values
    groups = df_train[GROUP_COL].astype(str).values
    X_dummy = df_train[feature_columns].values

    folds = []
    for tr_idx, va_idx in sgkf.split(X_dummy, y, groups):
        folds.append((tr_idx, va_idx))
    return folds


def _sample_random_params(rng: np.random.Generator) -> Dict:
    """Sample one random hyperparameter configuration."""
    return {
        "max_bins": int(rng.choice(PARAM_SPACE["max_bins"])),
        "learning_rate": float(rng.choice(PARAM_SPACE["learning_rate"])),
        "max_leaves": int(rng.choice(PARAM_SPACE["max_leaves"])),
        "min_samples_leaf": int(rng.choice(PARAM_SPACE["min_samples_leaf"])),
        "interactions": int(rng.choice(PARAM_SPACE["interactions"])),
        "max_interaction_bins": int(rng.choice(PARAM_SPACE["max_interaction_bins"])),
    }


def _train_evaluate_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_columns: List[str],
    params: Dict,
) -> float:
    """Train EBM on one fold, return validation AUC."""
    ebm = ExplainableBoostingClassifier(
        feature_names=feature_columns,
        max_bins=params["max_bins"],
        learning_rate=params["learning_rate"],
        max_leaves=params["max_leaves"],
        min_samples_leaf=params["min_samples_leaf"],
        interactions=params["interactions"],
        max_interaction_bins=params["max_interaction_bins"],
        max_rounds=500,            # capped for CV speed; final model uses 5000
        outer_bags=1,              # no bagging in CV (we handle splits ourselves)
        validation_size=0.0,       # we supply our own CV fold; no inner split
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    ebm.fit(X_train, y_train)

    y_prob = ebm.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_prob)
    return auc


def _plot_global_importances(ebm, feature_columns: List[str], out_dir: Path) -> None:
    """Plot top-20 EBM term importances (mean absolute score)."""
    importances = ebm.term_importances()
    names = ebm.term_names_

    # Build DataFrame for the main (non-interaction) terms
    imp_df = pd.DataFrame({"feature": names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(imp_df)), imp_df["importance"].values, align="center")
    plt.yticks(range(len(imp_df)), imp_df["feature"].values)
    plt.gca().invert_yaxis()
    plt.xlabel("Mean Absolute Score")
    plt.title("EBM – Top-20 Term Importances")
    plt.tight_layout()
    plt.savefig(out_dir / "ebm_term_importances.png", bbox_inches="tight", dpi=150)
    plt.close()


def main():
    print("=" * 60)
    print("EBM Training Pipeline (Production-Grade)")
    print("=" * 60)

    # ---- 1. Load & prep ----
    print("\n[1/6] Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in dataset")
    if GROUP_COL not in df.columns:
        raise ValueError(f"Group column '{GROUP_COL}' not found in dataset")

    X_all, y_all, feature_columns, X_df = _prepare_features(df)

    clean_df = X_df.copy()
    clean_df[LABEL_COL] = y_all
    clean_df[GROUP_COL] = df[GROUP_COL].astype(str)
    clean_df["session_id"] = (
        df["session_id"].astype(str) if "session_id" in df.columns else df.index.astype(str)
    )

    print(f"  Features: {len(feature_columns)}")
    print(f"  Samples:  {len(df)}")
    print(f"  Label distribution: {pd.Series(y_all).value_counts().to_dict()}")

    # ---- 2. Split ----
    print("\n[2/6] Splitting train/test (stratified + grouped)...")
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

    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

    # ---- 3. Hyperparameter random search with CV ----
    print(f"\n[3/6] Random search ({N_RANDOM_TRIALS} trials × 5-fold CV)...")
    folds = _make_cv_folds(train_df, feature_columns)
    rng = np.random.default_rng(RANDOM_STATE)
    n_total = N_RANDOM_TRIALS

    best = {"mean_auc": -1.0, "params": None, "all_results": []}

    for i in range(1, n_total + 1):
        params = _sample_random_params(rng)
        fold_aucs = []

        for fold_idx, (tr_idx, va_idx) in enumerate(folds):
            X_tr_fold = X_train[tr_idx]
            y_tr_fold = y_train[tr_idx]
            X_va_fold = X_train[va_idx]
            y_va_fold = y_train[va_idx]

            try:
                auc = _train_evaluate_fold(
                    X_tr_fold, y_tr_fold, X_va_fold, y_va_fold,
                    feature_columns, params,
                )
                fold_aucs.append(auc)
            except Exception as e:
                print(f"    [!] Fold {fold_idx} failed for combo {i}: {e}")
                fold_aucs.append(0.0)

        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))

        result_entry = {
            "trial": i,
            "params": params,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "fold_aucs": fold_aucs,
        }
        best["all_results"].append(result_entry)

        if mean_auc > best["mean_auc"]:
            best["mean_auc"] = mean_auc
            best["params"] = params
            print(f"  Trial {i}/{n_total}: ★ NEW BEST  AUC={mean_auc:.4f} ± {std_auc:.4f}  {params}")
        elif i % 50 == 0:
            print(f"  Trial {i}/{n_total}: AUC={mean_auc:.4f} (best so far: {best['mean_auc']:.4f})")

    if best["params"] is None:
        raise RuntimeError("Grid search failed: no valid param combination found.")

    print(f"\n  ✓ Best CV AUC: {best['mean_auc']:.4f}")
    print(f"  ✓ Best params: {best['params']}")

    # ---- 4. Train final model on full train split ----
    print("\n[4/6] Training final EBM on full train split...")
    final_ebm = ExplainableBoostingClassifier(
        feature_names=feature_columns,
        max_bins=best["params"]["max_bins"],
        learning_rate=best["params"]["learning_rate"],
        max_leaves=best["params"]["max_leaves"],
        min_samples_leaf=best["params"]["min_samples_leaf"],
        interactions=best["params"]["interactions"],
        max_interaction_bins=best["params"]["max_interaction_bins"],
        max_rounds=5000,
        early_stopping_rounds=50,
        validation_size=0.15,      # inner early-stop split for final training
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    final_ebm.fit(X_train, y_train)

    # ---- 5. Evaluate on held-out test ----
    print("\n[5/6] Evaluating on held-out test split...")
    y_prob = final_ebm.predict_proba(X_test)[:, 1]
    y_pred_05 = (y_prob >= 0.5).astype(int)
    y_pred_085 = (y_prob >= ENGAGEMENT_THRESHOLD).astype(int)

    metrics = {
        "model": "EBM (ExplainableBoostingClassifier)",
        "auc_roc": float(roc_auc_score(y_test, y_prob)),
        "auc_pr": float(average_precision_score(y_test, y_prob)),

        "precision_0.5": float(precision_score(y_test, y_pred_05, zero_division=0)),
        "recall_0.5": float(recall_score(y_test, y_pred_05, zero_division=0)),
        "f1_0.5": float(f1_score(y_test, y_pred_05, zero_division=0)),

        "precision_0.85": float(precision_score(y_test, y_pred_085, zero_division=0)),
        "recall_0.85": float(recall_score(y_test, y_pred_085, zero_division=0)),
        "f1_0.85": float(f1_score(y_test, y_pred_085, zero_division=0)),

        "confusion_matrix_0.85": confusion_matrix(y_test, y_pred_085).tolist(),
        "threshold_used": ENGAGEMENT_THRESHOLD,

        "cv_best_auc": best["mean_auc"],
    }

    print("\n  Test Metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    # ---- 6. Save artifacts ----
    print("\n[6/6] Saving artifacts...")

    # Model
    joblib.dump(final_ebm, ARTIFACTS_DIR / "ebm_model.joblib")

    # Feature columns
    with open(ARTIFACTS_DIR / "ebm_feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)

    # Split info (shared format)
    split_info = {
        "train_session_ids": train_df["session_id"].astype(str).tolist(),
        "test_session_ids": test_df["session_id"].astype(str).tolist(),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "group_col": GROUP_COL,
        "label_col": LABEL_COL,
    }
    with open(ARTIFACTS_DIR / "ebm_split.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # Metadata (full audit trail)
    # Serialize all_results: convert numpy types and trim fold_aucs for size
    serializable_results = []
    for r in best["all_results"]:
        serializable_results.append({
            "trial": r["trial"],
            "params": {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v) for k, v in r["params"].items()},
            "mean_auc": round(r["mean_auc"], 6),
            "std_auc": round(r["std_auc"], 6),
        })

    # Keep only top-10 trials in metadata to avoid huge files
    top_10_results = sorted(serializable_results, key=lambda x: x["mean_auc"], reverse=True)[:10]

    metadata = {
        "trained_at": datetime.utcnow().isoformat(),
        "model": "ExplainableBoostingClassifier",
        "interpret_version": ">=0.6.4",
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "label_col": LABEL_COL,
        "group_col": GROUP_COL,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "engagement_threshold": ENGAGEMENT_THRESHOLD,
        "drop_cols": DROP_COLS,
        "best_params": best["params"],
        "cv_best_auc": round(best["mean_auc"], 6),
        "total_trials": N_RANDOM_TRIALS,
        "search_strategy": "random",
        "top_10_trials": top_10_results,
        "metrics_test": metrics,
        "n_terms": len(final_ebm.term_names_),
        "term_names": list(final_ebm.term_names_),
    }
    with open(ARTIFACTS_DIR / "ebm_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Reports
    with open(REPORTS_DIR / "ebm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Global importance plot
    _plot_global_importances(final_ebm, feature_columns, REPORTS_DIR)

    print("\n" + "=" * 60)
    print("✓ EBM training complete.")
    print(f"  Artifacts: {ARTIFACTS_DIR.resolve()}")
    print(f"  Reports:   {REPORTS_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
