"""
ebm_explain.py – Production-grade native EBM explanations (glass-box, exact).

EBM is a Generalized Additive Model: prediction = Σ f_i(x_i) + intercept.
Each f_i(x_i) is the *exact* contribution of feature i for the given input.
No SHAP approximation needed – the explanation IS the model.

Key differences from SHAP-based explanations:
  • Contributions are exact (not a Shapley value approximation).
  • We can verify: intercept + Σ contributions ≡ logit prediction (sanity check).
  • Interaction terms are explicit and split equally to their component features.

Interface matches shap_explain.py for seamless API-level swapping.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

from ml.inference.load import load_ebm_model, load_ebm_feature_columns

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_score(raw_score) -> float:
    """
    EBM explain_local() returns term scores that can be either:
      - a plain float (scalar)
      - a per-class array, e.g. [neg_class_score, pos_class_score]

    For binary classification we always want the positive-class (index 1)
    contribution, which is in log-odds space.
    """
    if hasattr(raw_score, "__len__") and len(raw_score) > 1:
        return float(raw_score[1])
    return float(raw_score)


def _extract_intercept(raw_intercept) -> float:
    """Same per-class handling for the intercept term."""
    if hasattr(raw_intercept, "__len__") and len(raw_intercept) > 1:
        return float(raw_intercept[1])
    return float(raw_intercept)


# ---------------------------------------------------------------------------
# Core: compute local EBM contributions
# ---------------------------------------------------------------------------

def compute_local_ebm(features: Dict[str, float]) -> List[Dict[str, float]]:
    """
    Compute exact local contributions from the EBM model.

    Returns a list matching shap_explain.compute_local_shap() output:
      [{"feature": str, "value": float, "shap": float}, ...]

    We re-use the "shap" key name so the downstream pipeline
    (text_explainer, API schemas) works unchanged.
    The value is the EBM term score – the exact contribution in log-odds
    space (not a SHAP approximation).
    """
    ebm = load_ebm_model()
    feature_columns = load_ebm_feature_columns()

    x = np.array(
        [[float(features.get(col, 0.0)) for col in feature_columns]],
        dtype=float,
    )

    # --- Extract local explanation -----------------------------------------
    local_explanation = ebm.explain_local(x)
    data = local_explanation.data(0)

    term_names = data["names"]
    raw_scores = data["scores"]
    raw_intercept = data.get("intercept", 0.0)

    # Robust extraction – handles both scalar and per-class arrays
    term_scores = [_extract_score(s) for s in raw_scores]
    intercept = _extract_intercept(raw_intercept)

    # --- Glass-box sanity check --------------------------------------------
    # The entire point of using EBM: we can PROVE the explanation is exact.
    # intercept + Σ term_scores  MUST equal the model's logit prediction.
    computed_logit = intercept + sum(term_scores)

    try:
        predicted_logit = float(ebm.decision_function(x)[0])
        drift = abs(computed_logit - predicted_logit)

        if drift < 1e-4:
            log.debug(
                "[EBM-SANITY-OK] intercept=%.4f  Σscores=%.4f  "
                "computed_logit=%.4f  predicted_logit=%.4f  drift=%.6f",
                intercept, sum(term_scores),
                computed_logit, predicted_logit, drift,
            )
        else:
            log.warning(
                "[EBM-SANITY-FAIL] Explanation drift %.6f exceeds tolerance!  "
                "intercept=%.4f  Σscores=%.4f  computed=%.4f  predicted=%.4f",
                drift, intercept, sum(term_scores),
                computed_logit, predicted_logit,
            )
    except AttributeError:
        # Some EBM versions may not expose decision_function;
        # skip check gracefully.
        log.debug(
            "[EBM-SANITY-SKIP] decision_function not available; "
            "sanity check skipped.  intercept=%.4f  Σscores=%.4f",
            intercept, sum(term_scores),
        )

    # --- Map term scores to feature columns --------------------------------
    # EBM terms include main effects and optionally interaction terms
    # like "feature_a x feature_b".  We split interactions 50/50 (standard
    # heuristic) so the output aligns with feature_columns.
    contribution_map: Dict[str, float] = {}
    has_interactions = False

    for name, score in zip(term_names, term_scores):
        name_str = str(name)
        if " x " in name_str:
            # Interaction term — split contribution equally
            has_interactions = True
            parts = [p.strip() for p in name_str.split(" x ")]
            share = score / len(parts)
            for part in parts:
                contribution_map[part] = contribution_map.get(part, 0.0) + share
        else:
            contribution_map[name_str] = (
                contribution_map.get(name_str, 0.0) + score
            )

    if has_interactions:
        log.debug(
            "[EBM-INTERACTIONS] Interaction terms detected and split "
            "equally to component features."
        )

    # --- Build output (same shape as shap_explain) -------------------------
    out: List[Dict[str, float]] = []
    for col in feature_columns:
        out.append({
            "feature": col,
            "value": float(features.get(col, 0.0)),
            "shap": float(contribution_map.get(col, 0.0)),
        })

    return out


# ---------------------------------------------------------------------------
# Top contributors (same interface as shap_explain.top_contributors)
# ---------------------------------------------------------------------------

def top_contributors_ebm(
    ebm_rows: List[Dict[str, float]],
    k: int = 3,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """
    Same interface as shap_explain.top_contributors().
    Returns (top_negative, top_positive) sorted by contribution magnitude.
    """
    if not ebm_rows:
        return [], []
    sorted_rows = sorted(ebm_rows, key=lambda r: r.get("shap", 0.0))
    top_negative = sorted_rows[:k]
    top_positive = list(reversed(sorted_rows[-k:]))
    return top_negative, top_positive
