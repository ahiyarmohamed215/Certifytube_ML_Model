from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import shap
import xgboost as xgb

from ml.inference.load import load_model, load_feature_columns

_explainer = None


def _get_explainer():
    global _explainer
    if _explainer is None:
        booster = load_model()  # xgboost.Booster
        _explainer = shap.TreeExplainer(booster)
    return _explainer


def compute_local_shap(features: Dict[str, float]) -> List[Dict[str, float]]:
    feature_columns = load_feature_columns()
    explainer = _get_explainer()

    x = np.array([[features[col] for col in feature_columns]], dtype=float)
    dmat = xgb.DMatrix(x, feature_names=feature_columns)

    shap_vals = explainer.shap_values(dmat)

    # shap_vals: (1, n_features)
    shap_row = shap_vals[0]

    out = []
    for i, col in enumerate(feature_columns):
        out.append(
            {"feature": col, "value": float(x[0, i]), "shap": float(shap_row[i])}
        )
    return out


def top_contributors(shap_rows: List[Dict[str, float]], k: int = 3) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    sorted_rows = sorted(shap_rows, key=lambda r: r["shap"])
    top_negative = sorted_rows[:k]
    top_positive = list(reversed(sorted_rows[-k:]))
    return top_negative, top_positive
