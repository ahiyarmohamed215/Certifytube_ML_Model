"""Production-grade native EBM explanations for score-based inference.

EBM is a Generalized Additive Model: prediction = sum(f_i(x_i)) + intercept.
Each f_i(x_i) is the exact contribution of a feature for the given input, so
the explanation comes directly from the model rather than from an approximation.

The output shape matches ``shap_explain.py`` so the API layer can treat both
explainers the same way.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from verification.engagement.common.preprocessing import prepare_feature_array
from verification.engagement.ebm.inference.load import (
    load_ebm_feature_columns,
    load_ebm_model,
    load_ebm_preprocessing,
)

log = logging.getLogger(__name__)


def _extract_score(raw_score) -> float:
    """Normalize scalar or array-like score values into a single float."""
    if hasattr(raw_score, "__len__") and len(raw_score) > 0:
        return float(raw_score[-1])
    return float(raw_score)


def _extract_intercept(raw_intercept) -> float:
    """Normalize scalar or array-like intercept values into a single float."""
    if hasattr(raw_intercept, "__len__") and len(raw_intercept) > 0:
        return float(raw_intercept[-1])
    return float(raw_intercept)


def compute_local_ebm(features: Dict[str, float]) -> List[Dict[str, float]]:
    """Compute exact local contributions from the EBM regressor."""
    ebm = load_ebm_model()
    feature_columns = load_ebm_feature_columns()
    preprocessing = load_ebm_preprocessing()

    x = prepare_feature_array(features, preprocessing)

    local_explanation = ebm.explain_local(x)
    data = local_explanation.data(0)

    term_names = data["names"]
    raw_scores = data["scores"]
    raw_intercept = data.get("intercept", 0.0)

    term_scores = [_extract_score(score) for score in raw_scores]
    intercept = _extract_intercept(raw_intercept)

    computed_score = intercept + sum(term_scores)
    try:
        predicted_score = float(ebm.predict(x)[0])
        drift = abs(computed_score - predicted_score)

        if drift < 1e-4:
            log.debug(
                "[EBM-SANITY-OK] intercept=%.4f sum_scores=%.4f computed_score=%.4f predicted_score=%.4f drift=%.6f",
                intercept,
                sum(term_scores),
                computed_score,
                predicted_score,
                drift,
            )
        else:
            log.warning(
                "[EBM-SANITY-WARN] Explanation drift %.6f exceeds tolerance. intercept=%.4f sum_scores=%.4f computed_score=%.4f predicted_score=%.4f",
                drift,
                intercept,
                sum(term_scores),
                computed_score,
                predicted_score,
            )
    except AttributeError:
        log.debug(
            "[EBM-SANITY-SKIP] predict() not available; sanity check skipped. intercept=%.4f sum_scores=%.4f",
            intercept,
            sum(term_scores),
        )

    contribution_map: Dict[str, float] = {}
    has_interactions = False

    for name, score in zip(term_names, term_scores):
        name_str = str(name)
        if " x " in name_str:
            has_interactions = True
            parts = [part.strip() for part in name_str.split(" x ")]
            share = score / len(parts)
            for part in parts:
                contribution_map[part] = contribution_map.get(part, 0.0) + share
        else:
            contribution_map[name_str] = contribution_map.get(name_str, 0.0) + score

    if has_interactions:
        log.debug(
            "[EBM-INTERACTIONS] Interaction terms detected and split equally to component features."
        )

    out: List[Dict[str, float]] = []
    for idx, col in enumerate(feature_columns):
        out.append(
            {
                "feature": col,
                "value": float(x[0, idx]),
                "shap": float(contribution_map.get(col, 0.0)),
            }
        )

    return out


def top_contributors_ebm(
    ebm_rows: List[Dict[str, float]],
    k: int = 3,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """Return the strongest downward and upward contributors for the score."""
    if not ebm_rows:
        return [], []
    negative_rows = [row for row in ebm_rows if row.get("shap", 0.0) < 0.0]
    positive_rows = [row for row in ebm_rows if row.get("shap", 0.0) > 0.0]
    top_negative = sorted(negative_rows, key=lambda row: row.get("shap", 0.0))[:k]
    top_positive = sorted(
        positive_rows,
        key=lambda row: row.get("shap", 0.0),
        reverse=True,
    )[:k]
    return top_negative, top_positive
