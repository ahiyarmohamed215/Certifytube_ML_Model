from __future__ import annotations

from typing import Dict, List, Optional

from app.core.settings import settings
from verification.engagement.common.behavior_map import get_behavior

# ---------------------------------------------------------------------------
# Friendly, human-readable labels for each behavior category
# ---------------------------------------------------------------------------

BEHAVIOR_LABELS = {
    "coverage": "content coverage",
    "attention_consistency": "attention consistency",
    "rewatching": "active review",
    "reflective_pausing": "reflective pauses",
    "speed_watching": "playback pacing",
    "skipping": "continuous viewing",
    "playback_quality": "stable playback",
    "other": "learning behavior",
}

# ---------------------------------------------------------------------------
# Per-feature reasons (used for the contributor-level `reason` field)
# ---------------------------------------------------------------------------

POSITIVE_REASON = {
    "coverage": "You covered most of the lesson.",
    "attention_consistency": "Your attention looked consistent.",
    "rewatching": "You revisited important parts.",
    "reflective_pausing": "Your pauses looked reflective.",
    "speed_watching": "Your playback pace supported understanding.",
    "skipping": "You avoided excessive skipping.",
    "playback_quality": "Playback stayed stable enough for learning.",
    "other": "Your learning behavior supported understanding.",
}

NEGATIVE_REASON = {
    "coverage": "Coverage was not strong enough yet.",
    "attention_consistency": "Attention looked inconsistent in parts.",
    "rewatching": "There was limited review of key sections.",
    "reflective_pausing": "Pausing behavior did not show enough reflection.",
    "speed_watching": "Playback pacing may have reduced comprehension.",
    "skipping": "Frequent skipping reduced continuity.",
    "playback_quality": "Playback interruptions affected continuity.",
    "other": "Some behavior signals reduced engagement quality.",
}

# ---------------------------------------------------------------------------
# Conversational messages for the top-level `explanation` field.
# These are friendly, empathetic messages — like a tutor talking to a student.
# ---------------------------------------------------------------------------

CONGRATS_MESSAGES = {
    "coverage": "You watched most of the lesson thoroughly — great commitment!",
    "attention_consistency": "Your attention stayed consistent throughout — well done!",
    "rewatching": "You went back to review key parts — that shows real effort to understand!",
    "reflective_pausing": "Your thoughtful pausing shows you were really processing the material!",
    "speed_watching": "You kept a comfortable pace that supports real understanding!",
    "skipping": "You followed the lesson flow without jumping around — solid focus!",
    "playback_quality": "Your playback was smooth and uninterrupted — perfect conditions for learning!",
    "other": "Your overall learning behavior shows strong engagement!",
}

SORRY_MESSAGES = {
    "skipping": (
        "It looks like you skipped through some sections. "
        "Try watching the full lesson flow — the concepts build on each other!"
    ),
    "coverage": (
        "It seems like some parts of the lesson were missed. "
        "Watching the complete video helps build a stronger understanding."
    ),
    "speed_watching": (
        "Playing at high speed can make it harder to absorb the material. "
        "Try a comfortable pace where you can follow along."
    ),
    "attention_consistency": (
        "Your attention seemed to drift in parts. "
        "Try a focused session without distractions — it makes a big difference!"
    ),
    "reflective_pausing": (
        "Consider pausing after key ideas to let them sink in. "
        "Brief reflection helps lock in understanding."
    ),
    "rewatching": (
        "Rewatching tricky sections can really help. "
        "Next time, try going back to the parts that felt unclear."
    ),
    "playback_quality": (
        "Buffering interruptions made it harder to focus. "
        "A more stable connection would help you stay in the flow."
    ),
    "other": (
        "Some of your viewing patterns suggest you may not have fully engaged with the lesson. "
        "Try watching at a steady pace and take brief pauses to reflect."
    ),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _top_behaviors(rows: List[Dict[str, float]], limit: int = 2) -> List[str]:
    ordered: List[str] = []
    for row in rows:
        feature = row.get("feature")
        if not feature:
            continue
        behavior = get_behavior(str(feature))
        if behavior not in ordered:
            ordered.append(behavior)
        if len(ordered) >= limit:
            break
    return ordered


def _format_score(engagement_score: Optional[float]) -> str:
    if engagement_score is None:
        return "N/A"
    bounded = max(0.0, min(1.0, float(engagement_score)))
    return f"{bounded * 100:.0f}%"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feature_reason(feature: str, feature_value: float, shap_value: float) -> str:
    """Return a short reason for a single contributor (used in the contributor list)."""
    _ = feature_value  # intentionally unused to avoid exposing gameable thresholds.
    behavior = get_behavior(feature)
    if shap_value >= 0:
        return POSITIVE_REASON.get(behavior, POSITIVE_REASON["other"])
    return NEGATIVE_REASON.get(behavior, NEGATIVE_REASON["other"])


def build_user_explanation(
    shap_top_negative: List[Dict[str, float]],
    shap_top_positive: List[Dict[str, float]],
    engagement_score: Optional[float] = None,
) -> str:
    """Build a friendly, conversational explanation for the learner.

    - **Engaged**: Congratulatory message highlighting what they did well.
    - **Not engaged**: Empathetic message explaining the common reason and an actionable tip.
    """
    threshold = settings.engagement_threshold
    is_engaged = engagement_score is not None and engagement_score >= threshold
    score_text = _format_score(engagement_score)

    positive_behaviors = _top_behaviors(shap_top_positive, limit=2)
    negative_behaviors = _top_behaviors(shap_top_negative, limit=2)

    if is_engaged:
        # Pick the strongest positive behavior for the congrats message
        top_behavior = positive_behaviors[0] if positive_behaviors else "other"
        congrats = CONGRATS_MESSAGES.get(top_behavior, CONGRATS_MESSAGES["other"])
        return (
            f"Congratulations! Your engagement score is {score_text}. "
            f"{congrats} Keep this up in your next session!"
        )

    # Not engaged — use the top negative behavior for the sorry message
    top_negative = negative_behaviors[0] if negative_behaviors else "other"
    sorry = SORRY_MESSAGES.get(top_negative, SORRY_MESSAGES["other"])
    return (
        f"Your engagement score is {score_text}. "
        f"{sorry}"
    )
