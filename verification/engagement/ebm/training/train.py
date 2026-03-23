from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from interpret.glassbox import ExplainableBoostingClassifier

from verification.engagement.common.dataset import load_csv_with_header_fallback
from verification.engagement.common.preprocessing import (
    fit_numeric_preprocessing,
    save_preprocessing_artifact,
    select_numeric_feature_frame,
    transform_numeric_frame,
)
from verification.engagement.common.split import split_train_test

# ===========================
# CONFIG
# ===========================
DATA_PATH = Path("data/processed/sessions_features.csv")

LABEL_COL = "engagement_label"
GROUP_COL = "user_id"

ARTIFACTS_DIR = Path("verification/engagement/ebm/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = Path("reports/ebm")
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

PARAM_SPACE = {
    "max_bins": [128, 256, 512],
    "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
    "max_leaves": [2, 3, 4, 5, 6],
    "min_samples_leaf": [2, 4, 5, 8, 10, 15],
    "interactions": [0, 3, 5, 8, 10],
    "max_interaction_bins": [16, 32, 64],
}

N_RANDOM_TRIALS = 40


def _to_python_number(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _prepare_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing label column: {LABEL_COL}")
    if GROUP_COL not in df.columns:
        raise ValueError(f"Missing group column: {GROUP_COL}")

    feature_df = select_numeric_feature_frame(df, drop_cols=DROP_COLS)
    feature_columns = list(feature_df.columns)
    if not feature_columns:
        raise ValueError("No numeric features found after preprocessing.")

    clean_df = feature_df.copy()
    clean_df[LABEL_COL] = df[LABEL_COL].astype(int).values
    clean_df[GROUP_COL] = df[GROUP_COL].astype(str).values
    clean_df["session_id"] = (
        df["session_id"].astype(str).values
        if "session_id" in df.columns
        else df.index.astype(str).values
    )

    return clean_df, feature_columns, feature_df


def _sample_random_params(rng: np.random.Generator) -> Dict[str, object]:
    return {
        "max_bins": int(rng.choice(PARAM_SPACE["max_bins"])),
        "learning_rate": float(rng.choice(PARAM_SPACE["learning_rate"])),
        "max_leaves": int(rng.choice(PARAM_SPACE["max_leaves"])),
        "min_samples_leaf": int(rng.choice(PARAM_SPACE["min_samples_leaf"])),
        "interactions": int(rng.choice(PARAM_SPACE["interactions"])),
        "max_interaction_bins": int(rng.choice(PARAM_SPACE["max_interaction_bins"])),
    }


def _make_cv_folds(df_train: pd.DataFrame, feature_columns: List[str]) -> List[Tuple[np.ndarray, np.ndarray]]:
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y = df_train[LABEL_COL].astype(int).values
    groups = df_train[GROUP_COL].astype(str).values
    x_dummy = df_train[feature_columns].values

    folds = []
    for tr_idx, va_idx in sgkf.split(x_dummy, y, groups):
        folds.append((tr_idx, va_idx))
    return folds


def _class_balance(values: pd.Series) -> Dict[str, object]:
    counts = values.value_counts().sort_index()
    total = int(counts.sum())
    positive_count = int(counts.get(1, 0))
    return {
        "counts": {str(int(label)): int(count) for label, count in counts.items()},
        "positive_rate": round((positive_count / total) if total else 0.0, 4),
    }


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, object]:
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


def _plot_label_distribution(df: pd.DataFrame, out_path: Path) -> None:
    counts = df[LABEL_COL].value_counts().sort_index()
    plt.figure()
    plt.bar(["Not Engaged", "Engaged"], [counts.get(0, 0), counts.get(1, 0)], color=["#d95f02", "#1b9e77"])
    plt.ylabel("Sessions")
    plt.title("Engagement Label Distribution")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


def _plot_missingness(feature_df: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    missing_pct = feature_df.isna().mean().sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0].head(top_n)
    if missing_pct.empty:
        return

    plt.figure(figsize=(10, 6))
    plt.barh(missing_pct.index[::-1], missing_pct.values[::-1] * 100.0, color="#7570b3")
    plt.xlabel("Missing Values (%)")
    plt.title("Top Missing Features Before Imputation")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


def _plot_feature_target_correlation(
    clean_df: pd.DataFrame,
    feature_columns: List[str],
    out_path: Path,
    top_n: int = 20,
) -> None:
    non_constant_features = [
        column
        for column in feature_columns
        if clean_df[column].nunique(dropna=True) > 1
    ]
    if not non_constant_features:
        return

    corr = clean_df[non_constant_features].corrwith(clean_df[LABEL_COL]).dropna()
    if corr.empty:
        return

    corr = corr.abs().sort_values(ascending=False).head(top_n).sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(corr.index, corr.values, color="#66a61e")
    plt.xlabel("Absolute Correlation with Engagement Label")
    plt.title("Top Feature-Label Correlations")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


def _plot_sessions_per_user(clean_df: pd.DataFrame, out_path: Path) -> None:
    sessions_per_user = clean_df.groupby(GROUP_COL).size()
    if sessions_per_user.empty:
        return

    plt.figure()
    plt.hist(sessions_per_user.values, bins=min(30, max(5, sessions_per_user.nunique())), color="#1f78b4")
    plt.xlabel("Sessions per User")
    plt.ylabel("Users")
    plt.title("User Session Count Distribution")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


def _plot_probability_distribution(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    plt.figure()
    plt.hist(y_prob[y_true == 0], bins=30, alpha=0.6, label="Not Engaged", color="#d95f02")
    plt.hist(y_prob[y_true == 1], bins=30, alpha=0.6, label="Engaged", color="#1b9e77")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Sessions")
    plt.title("Predicted Score Distribution on Held-Out Test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


def _plot_global_importances(ebm: ExplainableBoostingClassifier, out_path: Path, top_n: int = 20) -> None:
    importances = ebm.term_importances()
    names = ebm.term_names_
    imp_df = (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance", ascending=True)
    )

    if imp_df.empty:
        return

    plt.figure(figsize=(10, 8))
    plt.barh(imp_df["feature"], imp_df["importance"], color="#e7298a")
    plt.xlabel("Mean Absolute Score")
    plt.title("EBM Top Term Importances")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


def _save_term_importances_csv(
    ebm: ExplainableBoostingClassifier,
    out_path: Path,
    top_n: int = 20,
) -> None:
    importances = ebm.term_importances()
    names = ebm.term_names_
    imp_df = (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    if imp_df.empty:
        return
    imp_df.to_csv(out_path, index=False)


def _save_cv_results(results: List[Dict[str, object]], out_dir: Path) -> None:
    rows = []
    for result in results:
        row = {
            "trial": int(result["trial"]),
            "mean_auc": float(result["mean_auc"]),
            "std_auc": float(result["std_auc"]),
        }
        row.update({f"param_{key}": _to_python_number(value) for key, value in result["params"].items()})
        rows.append(row)

    pd.DataFrame(rows).sort_values("mean_auc", ascending=False).to_csv(
        out_dir / "cv_results.csv",
        index=False,
    )


def _build_predictions_frame(test_df: pd.DataFrame, y_prob: np.ndarray) -> pd.DataFrame:
    predictions = test_df[[GROUP_COL, "session_id", LABEL_COL]].copy()
    predictions["y_prob"] = y_prob
    predictions["y_pred_0.5"] = (y_prob >= 0.5).astype(int)
    predictions["y_pred_0.85"] = (y_prob >= ENGAGEMENT_THRESHOLD).astype(int)

    def classify_error(row: pd.Series) -> str:
        if row[LABEL_COL] == row["y_pred_0.85"]:
            return "correct"
        if row["y_pred_0.85"] == 1:
            return "false_positive"
        return "false_negative"

    predictions["error_type_0.85"] = predictions.apply(classify_error, axis=1)
    return predictions


def _train_evaluate_fold(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    feature_columns: List[str],
    params: Dict[str, object],
) -> float:
    fold_preprocessing = fit_numeric_preprocessing(train_frame[feature_columns])
    x_train = transform_numeric_frame(train_frame[feature_columns], fold_preprocessing).values
    x_val = transform_numeric_frame(val_frame[feature_columns], fold_preprocessing).values
    y_train = train_frame[LABEL_COL].astype(int).values
    y_val = val_frame[LABEL_COL].astype(int).values

    ebm = ExplainableBoostingClassifier(
        feature_names=feature_columns,
        max_bins=int(params["max_bins"]),
        learning_rate=float(params["learning_rate"]),
        max_leaves=int(params["max_leaves"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        interactions=int(params["interactions"]),
        max_interaction_bins=int(params["max_interaction_bins"]),
        max_rounds=500,
        outer_bags=1,
        validation_size=0.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    ebm.fit(x_train, y_train)
    y_prob = ebm.predict_proba(x_val)[:, 1]
    return float(roc_auc_score(y_val, y_prob))


def _build_ebm_model(
    feature_columns: List[str],
    params: Dict[str, object],
) -> ExplainableBoostingClassifier:
    return ExplainableBoostingClassifier(
        feature_names=feature_columns,
        max_bins=int(params["max_bins"]),
        learning_rate=float(params["learning_rate"]),
        max_leaves=int(params["max_leaves"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        interactions=int(params["interactions"]),
        max_interaction_bins=int(params["max_interaction_bins"]),
        max_rounds=5000,
        early_stopping_rounds=50,
        validation_size=0.15,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def main():
    print("Loading dataset...")
    df = load_csv_with_header_fallback(DATA_PATH, required_columns=[LABEL_COL, GROUP_COL, "session_id"])

    clean_df, feature_columns, feature_df = _prepare_feature_frame(df)

    print("Generating dataset profile plots...")
    _plot_label_distribution(clean_df, REPORTS_DIR / "label_distribution.png")
    _plot_missingness(feature_df, REPORTS_DIR / "missingness_top20.png")
    _plot_feature_target_correlation(clean_df, feature_columns, REPORTS_DIR / "feature_label_correlation_top20.png")
    _plot_sessions_per_user(clean_df, REPORTS_DIR / "sessions_per_user.png")

    print("Splitting train/test (80/20) with stratified grouped split...")
    train_df, test_df = split_train_test(
        clean_df,
        label_col=LABEL_COL,
        group_col=GROUP_COL,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    eval_preprocessing = fit_numeric_preprocessing(train_df[feature_columns])
    x_train_df = transform_numeric_frame(train_df[feature_columns], eval_preprocessing)
    x_test_df = transform_numeric_frame(test_df[feature_columns], eval_preprocessing)
    y_train = train_df[LABEL_COL].astype(int).values
    y_test = test_df[LABEL_COL].astype(int).values

    print(f"Random search ({N_RANDOM_TRIALS} trials x 5-fold CV)...")
    folds = _make_cv_folds(train_df, feature_columns)
    rng = np.random.default_rng(RANDOM_STATE)

    best = {"mean_auc": -1.0, "params": None, "all_results": []}

    for trial in range(1, N_RANDOM_TRIALS + 1):
        params = _sample_random_params(rng)
        fold_aucs = []

        for tr_idx, va_idx in folds:
            train_fold = train_df.iloc[tr_idx]
            val_fold = train_df.iloc[va_idx]
            try:
                auc = _train_evaluate_fold(train_fold, val_fold, feature_columns, params)
                fold_aucs.append(auc)
            except Exception as exc:
                print(f"  Trial {trial}: fold failed with {exc}")
                fold_aucs.append(0.0)

        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))
        result_entry = {
            "trial": trial,
            "params": params,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "fold_aucs": fold_aucs,
        }
        best["all_results"].append(result_entry)

        if mean_auc > best["mean_auc"]:
            best["mean_auc"] = mean_auc
            best["params"] = params
            print(f"  Trial {trial}/{N_RANDOM_TRIALS}: NEW BEST AUC={mean_auc:.4f} +/- {std_auc:.4f}")

    if best["params"] is None:
        raise RuntimeError("Random search failed: no valid parameter combination found.")

    print("Training evaluation model on train split...")
    eval_ebm = _build_ebm_model(feature_columns, best["params"])
    eval_ebm.fit(x_train_df.values, y_train)

    print("Evaluating on held-out test split...")
    y_prob = eval_ebm.predict_proba(x_test_df.values)[:, 1]
    metrics = _compute_metrics(y_test, y_prob)
    metrics["cv_best_auc"] = float(best["mean_auc"])

    predictions_df = _build_predictions_frame(test_df, y_prob)
    predictions_df.to_csv(REPORTS_DIR / "test_predictions.csv", index=False)
    predictions_df[predictions_df["error_type_0.85"] != "correct"].to_csv(
        REPORTS_DIR / "test_misclassifications.csv",
        index=False,
    )

    _plot_probability_distribution(y_test, y_prob, REPORTS_DIR / "predicted_score_distribution.png")
    _plot_global_importances(eval_ebm, REPORTS_DIR / "term_importances.png")
    _save_term_importances_csv(eval_ebm, REPORTS_DIR / "term_importances.csv")
    _save_cv_results(best["all_results"], REPORTS_DIR)

    print("Training final production model on full dataset...")
    production_preprocessing = fit_numeric_preprocessing(clean_df[feature_columns])
    x_full_df = transform_numeric_frame(clean_df[feature_columns], production_preprocessing)
    y_full = clean_df[LABEL_COL].astype(int).values

    final_ebm = _build_ebm_model(feature_columns, best["params"])
    final_ebm.fit(x_full_df.values, y_full)

    print("Saving artifacts...")
    joblib.dump(final_ebm, ARTIFACTS_DIR / "ebm_model.joblib")
    joblib.dump(eval_ebm, ARTIFACTS_DIR / "ebm_model_eval.joblib")

    with open(ARTIFACTS_DIR / "ebm_feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)

    save_preprocessing_artifact(production_preprocessing, ARTIFACTS_DIR / "ebm_preprocessing.json")
    save_preprocessing_artifact(eval_preprocessing, ARTIFACTS_DIR / "ebm_preprocessing_eval.json")

    split_info = {
        "train_session_ids": train_df["session_id"].astype(str).tolist(),
        "test_session_ids": test_df["session_id"].astype(str).tolist(),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "group_col": GROUP_COL,
        "label_col": LABEL_COL,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }
    with open(ARTIFACTS_DIR / "ebm_split.json", "w") as f:
        json.dump(split_info, f, indent=2)

    summary = {
        "dataset_rows_total": int(len(clean_df)),
        "dataset_users_total": int(clean_df[GROUP_COL].nunique()),
        "feature_count": int(len(feature_columns)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_users": int(train_df[GROUP_COL].nunique()),
        "test_users": int(test_df[GROUP_COL].nunique()),
        "class_balance_total": _class_balance(clean_df[LABEL_COL]),
        "class_balance_train": _class_balance(train_df[LABEL_COL]),
        "class_balance_test": _class_balance(test_df[LABEL_COL]),
        "best_params": {key: _to_python_number(value) for key, value in best["params"].items()},
        "metrics_test": metrics,
    }
    (REPORTS_DIR / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    top_trials = sorted(best["all_results"], key=lambda item: item["mean_auc"], reverse=True)[:10]
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": "ExplainableBoostingClassifier",
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "label_col": LABEL_COL,
        "group_col": GROUP_COL,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "engagement_threshold": ENGAGEMENT_THRESHOLD,
        "drop_cols": DROP_COLS,
        "best_params": {key: _to_python_number(value) for key, value in best["params"].items()},
        "cv_best_auc": round(float(best["mean_auc"]), 6),
        "metrics_test": metrics,
        "preprocessing": {
            "strategy": "median_imputation",
            "production_artifact": "ebm_preprocessing.json",
            "evaluation_artifact": "ebm_preprocessing_eval.json",
        },
        "models": {
            "production_model": "ebm_model.joblib",
            "evaluation_model": "ebm_model_eval.joblib",
        },
        "reports_dir": str(REPORTS_DIR),
        "top_cv_trials": [
            {
                "trial": int(item["trial"]),
                "mean_auc": round(float(item["mean_auc"]), 6),
                "std_auc": round(float(item["std_auc"]), 6),
                "params": {key: _to_python_number(value) for key, value in item["params"].items()},
            }
            for item in top_trials
        ],
        "dataset_summary": summary,
        "n_terms": len(final_ebm.term_names_),
        "term_names": list(final_ebm.term_names_),
    }
    with open(ARTIFACTS_DIR / "ebm_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(REPORTS_DIR / "metrics_test.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Done.")
    print(f"Artifacts: {ARTIFACTS_DIR.resolve()}")
    print(f"Reports:   {REPORTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
