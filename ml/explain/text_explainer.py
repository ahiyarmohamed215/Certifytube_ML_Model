from __future__ import annotations

from typing import Dict, List, Tuple

from ml.explain.behavior_map import get_behavior
from ml.explain.templates import engaged_template, not_engaged_template


def _collect_behaviors(
    shap_rows: List[Dict[str, float]],
) -> Tuple[List[str], List[str]]:
    """
    Given SHAP rows, return:
    (positive_behaviors, negative_behaviors)
    """
    positive_behaviors = []
    negative_behaviors = []

    for row in shap_rows:
        behavior = get_behavior(row["feature"])
        shap_value = row["shap"]

        if shap_value >= 0:
            positive_behaviors.append(behavior)
        else:
            negative_behaviors.append(behavior)

    return positive_behaviors, negative_behaviors


def build_explanation_text(
    shap_top_negative: List[Dict[str, float]],
    shap_top_positive: List[Dict[str, float]],
    status: str,
) -> str:
    """
    Builds a human-readable explanation sentence based on:
    - SHAP top contributors
    - engagement status
    """

    # Collect behaviors separately
    pos_behaviors_from_pos, neg_behaviors_from_pos = _collect_behaviors(shap_top_positive)
    pos_behaviors_from_neg, neg_behaviors_from_neg = _collect_behaviors(shap_top_negative)

    # Merge behavior lists
    positive_behaviors = pos_behaviors_from_pos + pos_behaviors_from_neg
    negative_behaviors = neg_behaviors_from_neg + neg_behaviors_from_pos

    if status == "ENGAGED":
        return engaged_template(
            top_positive_behaviors=positive_behaviors,
            top_negative_behaviors=negative_behaviors,
        )

    return not_engaged_template(
        top_negative_behaviors=negative_behaviors,
        top_positive_behaviors=positive_behaviors,
    )
