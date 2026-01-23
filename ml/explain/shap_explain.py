from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import shap

from ml.inference.load import load_model, load_feature_columns

# Cache explainer so we don't recreate it every request
_explainer = None


def _get_explainer():
    global _explainer
    if _explainer is None:
        model = load_model()
        # TreeExplainer is best for XGBoost tree models
        _explainer = shap.TreeExplainer(model)
    return _explainer


def compute_local_shap(features: Dict[str, float]) -> List[Dict[str, float]]:
    """
    Returns a list of dicts:
    [
      {"feature": "pause_total_sec", "value": 120.0, "shap": 0.08},
      ...
    ]
    Positive shap => pushes toward ENGAGED (class 1)
    Negative shap => pushes toward NOT_ENGAGED (class 0)
    """
    feature_columns = load_feature_columns()
    explainer = _get_explainer()

    # Ensure correct column order
    x = np.array([[features[col] for col in feature_columns]], dtype=float)

    # SHAP values for class 1 in binary classification can be:
    # - list of arrays [class0, class1] OR
    # - single array (depends on shap version/model)
    shap_vals = explainer.shap_values(x)

    if isinstance(shap_vals, list):
        # class 1
        shap_class1 = shap_vals[1][0]
    else:
        shap_class1 = shap_vals[0]

    out = []
    for i, col in enumerate(feature_columns):
        out.append(
            {
                "feature": col,
                "value": float(x[0, i]),
                "shap": float(shap_class1[i]),
            }
        )
    return out


def top_contributors(
    shap_rows: List[Dict[str, float]],
    k: int = 3,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """
    Returns (top_negative, top_positive)
    based on SHAP values for class 1.
    """
    sorted_rows = sorted(shap_rows, key=lambda r: r["shap"])
    top_negative = sorted_rows[:k]  # most negative
    top_positive = list(reversed(sorted_rows[-k:]))  # most positive
    return top_negative, top_positive
