from __future__ import annotations

from typing import Dict, List, Tuple

from ml.explain.behavior_map import get_behavior

# Negative (NOT_ENGAGED) reason codes (stable, non-actionable)
NEG_BEHAVIOR_TO_REASON = {
    "attention_consistency": "LOW_ATTENTION",
    "coverage": "LOW_COVERAGE",
    "skipping": "EXCESSIVE_SKIPPING",
    "rewatching": "UNUSUAL_REWATCHING",
    "reflective_pausing": "UNUSUAL_PAUSING",
    "speed_watching": "PLAYBACK_SPEED_VARIATION",
    "playback_quality": "PLAYBACK_INTERRUPTION",
    "other": "OTHER_SIGNAL",
}

# Positive (ENGAGED) reason codes (stable, non-actionable)
POS_BEHAVIOR_TO_REASON = {
    "attention_consistency": "HIGH_ATTENTION",
    "coverage": "HIGH_COVERAGE",
    "skipping": "LOW_SKIPPING",
    "rewatching": "HEALTHY_REWATCHING",
    "reflective_pausing": "HEALTHY_PAUSING",
    "speed_watching": "NORMAL_SPEED_BEHAVIOR",
    "playback_quality": "STABLE_PLAYBACK",
    "other": "OTHER_SIGNAL",
}

# Labels for explanation_text (still non-instructional)
REASON_LABELS = {
    # Engaged
    "HIGH_ATTENTION": "sustained attention",
    "HIGH_COVERAGE": "strong coverage",
    "LOW_SKIPPING": "minimal skipping",
    "HEALTHY_REWATCHING": "rewatching patterns",
    "HEALTHY_PAUSING": "pausing patterns",
    "NORMAL_SPEED_BEHAVIOR": "playback speed patterns",
    "STABLE_PLAYBACK": "stable playback",

    # Not engaged
    "LOW_ATTENTION": "low attention",
    "LOW_COVERAGE": "limited viewing",
    "EXCESSIVE_SKIPPING": "skipping behavior",
    "UNUSUAL_REWATCHING": "rewatching patterns",
    "UNUSUAL_PAUSING": "pausing patterns",
    "PLAYBACK_SPEED_VARIATION": "playback speed changes",
    "PLAYBACK_INTERRUPTION": "playback interruptions",

    # Other
    "OTHER_SIGNAL": "other interaction signals",
}

# Deterministic ordering (so output is consistent and not "random")
REASON_PRIORITY = {
    # Primary signals first
    "HIGH_ATTENTION": 10,
    "LOW_ATTENTION": 10,
    "HIGH_COVERAGE": 20,
    "LOW_COVERAGE": 20,

    # Quality / playback next
    "STABLE_PLAYBACK": 30,
    "PLAYBACK_INTERRUPTION": 30,

    # Interaction patterns after
    "LOW_SKIPPING": 40,
    "EXCESSIVE_SKIPPING": 40,
    "HEALTHY_REWATCHING": 50,
    "UNUSUAL_REWATCHING": 50,
    "HEALTHY_PAUSING": 60,
    "UNUSUAL_PAUSING": 60,
    "NORMAL_SPEED_BEHAVIOR": 70,
    "PLAYBACK_SPEED_VARIATION": 70,

    # Always last
    "OTHER_SIGNAL": 999,
}


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _collect_behaviors(shap_rows: List[Dict[str, float]]) -> Tuple[List[str], List[str]]:
    """
    Returns (positive_behaviors, negative_behaviors) based on SHAP sign.
    """
    pos, neg = [], []
    for row in shap_rows:
        feature = row.get("feature")
        shap_value = row.get("shap")
        if feature is None or shap_value is None:
            continue
        behavior = get_behavior(feature)
        if shap_value >= 0:
            pos.append(behavior)
        else:
            neg.append(behavior)
    return pos, neg


def _behaviors_to_codes(behaviors: List[str], mapping: Dict[str, str]) -> List[str]:
    codes = [mapping.get(b, "OTHER_SIGNAL") for b in behaviors]
    codes = _dedupe_keep_order(codes)
    # Make ordering deterministic
    codes = sorted(codes, key=lambda c: (REASON_PRIORITY.get(c, 500), c))
    return codes


def _codes_to_labels(codes: List[str]) -> List[str]:
    labels = [REASON_LABELS.get(c, "other interaction signals") for c in codes]
    # labels list follows ordered codes; dedupe just in case
    return _dedupe_keep_order(labels)


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
    status: str,
) -> Tuple[str, List[str], List[str], List[str]]:
    """
    Returns:
      explanation_text,
      positive_behaviors,
      negative_behaviors,
      reason_codes (ordered, stable)
    """

    pos1, neg1 = _collect_behaviors(shap_top_positive)
    pos2, neg2 = _collect_behaviors(shap_top_negative)

    positive_behaviors = _dedupe_keep_order(pos1 + pos2)
    negative_behaviors = _dedupe_keep_order(neg2 + neg1)

    if status == "ENGAGED":
        reason_codes = _behaviors_to_codes(positive_behaviors, POS_BEHAVIOR_TO_REASON)
        labels = _codes_to_labels(reason_codes)

        # Keep it short (2 reasons max), still non-actionable
        top = _join_two(labels[:2])
        if top:
            return (
                f"Engagement was confirmed for this session due to {top}.",
                positive_behaviors,
                negative_behaviors,
                reason_codes,
            )
        return (
            "Engagement was confirmed for this session.",
            positive_behaviors,
            negative_behaviors,
            reason_codes,
        )

    # NOT_ENGAGED
    reason_codes = _behaviors_to_codes(negative_behaviors, NEG_BEHAVIOR_TO_REASON)
    labels = _codes_to_labels(reason_codes)

    top = _join_two(labels[:2])
    if top:
        return (
            f"Engagement could not be confirmed for this session due to {top}.",
            positive_behaviors,
            negative_behaviors,
            reason_codes,
        )

    return (
        "Engagement could not be confirmed for this session.",
        positive_behaviors,
        negative_behaviors,
        reason_codes,
    )
