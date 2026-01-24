from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
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

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2  # 80/20

ENGAGEMENT_THRESHOLD = 0.85

DROP_COLS = [
    LABEL_COL,
    GROUP_COL,
    "session_id",
    "video_id",
    "video_title",
]


def _prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing label column: {LABEL_COL}")

    y = df[LABEL_COL].astype(int).values

    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()

    # Median impute (0 can be a real signal)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    feature_columns = list(X.columns)
    if not feature_columns:
        raise ValueError("No numeric features found after preprocessing.")

    return X.values, y, feature_columns, X


def _sample_params(rng: np.random.Generator) -> Dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "logloss"],   # IMPORTANT: logloss helps diagnose overfit too
        "tree_method": "hist",
        "seed": RANDOM_STATE,

        "max_depth": int(rng.integers(3, 8)),
        "min_child_weight": float(rng.uniform(1, 10)),
        "eta": float(rng.uniform(0.01, 0.2)),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "gamma": float(rng.uniform(0.0, 5.0)),
        "lambda": float(rng.uniform(0.1, 10.0)),
        "alpha": float(rng.uniform(0.0, 5.0)),
    }


def _make_cv_folds(df_train: pd.DataFrame, feature_columns: List[str]) -> List[Tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedGroupKFold

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y = df_train[LABEL_COL].astype(int).values
    groups = df_train[GROUP_COL].astype(str).values
    X_dummy = df_train[feature_columns].values

    folds = []
    for tr_idx, va_idx in sgkf.split(X_dummy, y, groups):
        folds.append((tr_idx, va_idx))
    return folds


def _plot_learning_curves(evals_result: Dict, out_path: Path) -> None:
    """
    Overfitting/Underfitting evidence:
      - Train vs Test AUC
      - Train vs Test logloss
    """
    # evals_result structure:
    # evals_result["train"]["auc"] etc.
    train_auc = evals_result.get("train", {}).get("auc", [])
    test_auc = evals_result.get("test", {}).get("auc", [])
    train_ll = evals_result.get("train", {}).get("logloss", [])
    test_ll = evals_result.get("test", {}).get("logloss", [])

    if train_auc and test_auc:
        plt.figure()
        plt.plot(train_auc, label="Train AUC")
        plt.plot(test_auc, label="Test AUC")
        plt.xlabel("Boosting Rounds")
        plt.ylabel("AUC")
        plt.title("Learning Curve: Train vs Test AUC")
        plt.legend()
        plt.savefig(out_path / "learning_curve_auc.png", bbox_inches="tight")
        plt.close()

    if train_ll and test_ll:
        plt.figure()
        plt.plot(train_ll, label="Train Logloss")
        plt.plot(test_ll, label="Test Logloss")
        plt.xlabel("Boosting Rounds")
        plt.ylabel("Logloss")
        plt.title("Learning Curve: Train vs Test Logloss")
        plt.legend()
        plt.savefig(out_path / "learning_curve_logloss.png", bbox_inches="tight")
        plt.close()


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in dataset")
    if GROUP_COL not in df.columns:
        raise ValueError(f"Group column '{GROUP_COL}' not found in dataset")

    X_all, y_all, feature_columns, X_df = _prepare_features(df)

    clean_df = X_df.copy()
    clean_df[LABEL_COL] = y_all
    clean_df[GROUP_COL] = df[GROUP_COL].astype(str)
    clean_df["session_id"] = df["session_id"].astype(str) if "session_id" in df.columns else df.index.astype(str)

    print("Splitting train/test (80/20) with stratified grouped split...")
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

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_columns)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_columns)

    print("Tuning hyperparameters (random search + xgb.cv)...")
    rng = np.random.default_rng(RANDOM_STATE)
    folds = _make_cv_folds(train_df, feature_columns)

    n_trials = 40
    best = {"auc": -1.0, "params": None, "best_round": None}

    for t in range(n_trials):
        params = _sample_params(rng)

        cv = xgb.cv(
            params=params,
            dtrain=dtrain,
            num_boost_round=5000,
            folds=folds,
            early_stopping_rounds=50,
            verbose_eval=False,
            as_pandas=True,
        )

        best_round = len(cv)
        auc_mean = float(cv["test-auc-mean"].iloc[-1])

        if auc_mean > best["auc"]:
            best = {"auc": auc_mean, "params": params, "best_round": best_round}
            print(f"  Trial {t+1}/{n_trials}: NEW BEST AUC={auc_mean:.4f}, rounds={best_round}")

    if best["params"] is None:
        raise RuntimeError("Tuning failed: no params selected.")

    print("Best CV AUC:", best["auc"])
    print("Best params:", best["params"])
    print("Best num_boost_round:", best["best_round"])

    # ---- Train final model and CAPTURE eval history for learning curves ----
    print("Training final booster on full train split...")
    evals_result: Dict = {}

    booster = xgb.train(
        params=best["params"],
        dtrain=dtrain,
        num_boost_round=int(best["best_round"]),
        evals=[(dtrain, "train"), (dtest, "test")],
        evals_result=evals_result,
        verbose_eval=False,
    )

    # Save eval history so your thesis can cite it
    with open(REPORTS_DIR / "training_evals_result.json", "w") as f:
        json.dump(evals_result, f, indent=2)

    # Make learning-curve plots (overfit/underfit evidence)
    _plot_learning_curves(evals_result, REPORTS_DIR)

    # ---- Evaluate on held-out test ----
    print("Evaluating on held-out test split...")
    y_prob = booster.predict(dtest)
    y_pred_05 = (y_prob >= 0.5).astype(int)
    y_pred_085 = (y_prob >= ENGAGEMENT_THRESHOLD).astype(int)

    metrics = {
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

        "cv_best_auc": float(best["auc"]),
        "cv_best_round": int(best["best_round"]),
    }

    print("Test Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # ---- Save artifacts ----
    print("Saving artifacts...")
    joblib.dump(booster, ARTIFACTS_DIR / "model.joblib")

    with open(ARTIFACTS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)

    split_info = {
        "train_session_ids": train_df["session_id"].astype(str).tolist(),
        "test_session_ids": test_df["session_id"].astype(str).tolist(),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "group_col": GROUP_COL,
        "label_col": LABEL_COL,
    }
    with open(ARTIFACTS_DIR / "split.json", "w") as f:
        json.dump(split_info, f, indent=2)

    metadata = {
        "trained_at": datetime.utcnow().isoformat(),
        "model": "xgboost.Booster",
        "xgboost_version": xgb.__version__,
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "label_col": LABEL_COL,
        "group_col": GROUP_COL,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "engagement_threshold": ENGAGEMENT_THRESHOLD,
        "drop_cols": DROP_COLS,
        "best_params": best["params"],
        "best_round": int(best["best_round"]),
        "metrics_test": metrics,
    }
    with open(ARTIFACTS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Done.")
    print(f"Artifacts: {ARTIFACTS_DIR.resolve()}")
    print(f"Reports:   {REPORTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
