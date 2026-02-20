from __future__ import annotations

from typing import Dict, List, Tuple

from ml.explain.behavior_map import get_behavior

# Reason labels for human-readable text (status-neutral)
BEHAVIOR_LABELS = {
    "attention_consistency": "attention consistency",
    "coverage": "content coverage",
    "skipping": "skipping behavior",
    "rewatching": "rewatching patterns",
    "reflective_pausing": "pausing behavior",
    "speed_watching": "playback speed patterns",
    "playback_quality": "playback quality",
    "other": "other interaction signals",
}

# Deterministic ordering (so output is consistent)
BEHAVIOR_PRIORITY = {
    "attention_consistency": 10,
    "coverage": 20,
    "playback_quality": 30,
    "skipping": 40,
    "rewatching": 50,
    "reflective_pausing": 60,
    "speed_watching": 70,
    "other": 999,
}


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _collect_behaviors(shap_rows: List[Dict[str, float]]) -> List[str]:
    """Extract behavior categories from contributor rows."""
    behaviors = []
    for row in shap_rows:
        feature = row.get("feature")
        if feature is None:
            continue
        behaviors.append(get_behavior(feature))
    return behaviors


def _join_two(labels: List[str]) -> str:
    labels = [x for x in labels if x]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    return f"{labels[0]} and {labels[1]}"


def build_user_explanation(
    shap_top_negative: List[Dict[str, float]],
    shap_top_positive: List[Dict[str, float]],
) -> str:
    """
    Build a status-neutral explanation describing the key factors
    influencing the engagement score.

    Returns:
      explanation_text  (str)
    """

    # Collect behaviors from the top positive and negative contributors
    pos_behaviors = _dedupe_keep_order(_collect_behaviors(shap_top_positive))
    neg_behaviors = _dedupe_keep_order(_collect_behaviors(shap_top_negative))

    # Combine: positive factors first, then negative
    all_behaviors = _dedupe_keep_order(pos_behaviors + neg_behaviors)

    # Sort deterministically
    all_behaviors = sorted(
        all_behaviors,
        key=lambda b: (BEHAVIOR_PRIORITY.get(b, 500), b),
    )

    # Convert to human-readable labels
    labels = [BEHAVIOR_LABELS.get(b, "other interaction signals") for b in all_behaviors]
    labels = _dedupe_keep_order(labels)

    top = _join_two(labels[:2])
    if top:
        return f"The primary factors influencing this score were {top}."

    return "The engagement score was computed based on the session's behavioral signals."
