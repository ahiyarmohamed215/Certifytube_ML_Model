from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional

from verification.engagement.common.behavior_map import get_behavior

EngagementStatus = Literal["engaged", "not_engaged"]

BEHAVIOR_LABELS = {
    "coverage": "coverage",
    "attention_consistency": "attention consistency",
    "rewatching": "review behavior",
    "reflective_pausing": "pause behavior",
    "speed_watching": "playback pace",
    "skipping": "lesson continuity",
    "playback_quality": "playback stability",
    "other": "session behavior",
}

def _safe_float(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _feature_map(
    rows: List[Dict[str, float]],
    session_features: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if session_features:
        for feature, value in session_features.items():
            numeric = _safe_float(value)
            if numeric is not None:
                metrics[str(feature)] = numeric
    for row in rows:
        feature = str(row.get("feature", "")).strip()
        numeric = _safe_float(row.get("value"))
        if feature and numeric is not None and feature not in metrics:
            metrics[feature] = numeric
    return metrics


def _metric(metrics: Dict[str, float], *names: str) -> Optional[float]:
    for name in names:
        numeric = _safe_float(metrics.get(name))
        if numeric is not None:
            return numeric
    return None


def _top_behavior_groups(
    rows: List[Dict[str, float]],
    limit: int = 2,
) -> List[tuple[str, List[Dict[str, float]]]]:
    grouped: Dict[str, List[Dict[str, float]]] = {}
    order: List[str] = []
    for row in rows:
        feature = str(row.get("feature", "")).strip()
        behavior = get_behavior(feature)
        if behavior not in grouped:
            grouped[behavior] = []
            order.append(behavior)
        grouped[behavior].append(row)
    return [(behavior, grouped[behavior]) for behavior in order[:limit]]


def _join_clauses(clauses: List[str]) -> str:
    cleaned = [clause.strip().rstrip(".") for clause in clauses if clause.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]}, and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def _format_score_percent(engagement_score: Optional[float]) -> str:
    score = 0.0 if engagement_score is None else max(0.0, min(1.0, float(engagement_score)))
    return f"{score * 100:.0f}%"


def _fallback_behavior_text(behavior: str, direction: str) -> str:
    label = BEHAVIOR_LABELS.get(behavior, BEHAVIOR_LABELS["other"])
    if direction == "positive":
        return f"{label} supported the session"
    return f"{label} pulled the session down"


def _coverage_text(metrics: Dict[str, float], direction: str) -> str:
    completion = _metric(metrics, "completion_ratio")
    effective = _metric(metrics, "effective_consumption_ratio", "watch_time_ratio")
    completed_flag = _metric(metrics, "completed_flag")

    if direction == "positive":
        if completed_flag is not None and completed_flag >= 0.5:
            return "you stayed with the lesson through to the end"
        if completion is not None and completion >= 0.9:
            return "coverage stayed strong across most of the lesson"
        if effective is not None:
            return "meaningful lesson coverage stayed strong"
        return "coverage stayed solid"

    if completion is not None and completion < 0.9:
        return "coverage stayed incomplete across important parts of the lesson"
    if effective is not None:
        return "meaningful lesson coverage stayed too low"
    return "coverage stayed limited"


def _skipping_text(metrics: Dict[str, float], direction: str) -> str:
    skip_ratio = _metric(metrics, "skip_time_ratio")
    forward_seeks = _metric(metrics, "num_seek_forward")
    largest_forward_seek = _metric(metrics, "largest_forward_seek_sec")
    early_skip = _metric(metrics, "early_skip_flag")
    skim_flag = _metric(metrics, "skim_flag")
    rage_seek_count = _metric(metrics, "rage_seek_count")

    if direction == "positive":
        if skip_ratio is not None and skip_ratio <= 0.05:
            return "the lesson was followed in order without much skipping"
        if forward_seeks is not None and forward_seeks <= 1:
            return "forward jumps stayed minimal"
        return "lesson flow stayed mostly intact"

    if skim_flag is not None and skim_flag >= 0.5:
        return "the session looked more like skimming than steady viewing"
    if skip_ratio is not None and skip_ratio > 0:
        return "forward skipping broke the lesson flow"
    if forward_seeks is not None and forward_seeks >= 1:
        return "frequent forward jumps broke the lesson flow"
    if largest_forward_seek is not None and largest_forward_seek > 0:
        return "large forward jumps removed important parts of the lesson flow"
    if early_skip is not None and early_skip >= 0.5:
        return "jumping ahead started early in the session"
    if rage_seek_count is not None and rage_seek_count >= 1:
        return "repeated rapid seeking suggested loss of continuity"
    return "forward jumps broke the lesson flow"


def _attention_text(metrics: Dict[str, float], direction: str) -> str:
    attention_index = _metric(metrics, "attention_index")
    seek_density = _metric(metrics, "seek_density_per_min")
    first_seek_time = _metric(metrics, "first_seek_time_sec")
    play_pause_ratio = _metric(metrics, "play_pause_ratio")

    if direction == "positive":
        if attention_index is not None and attention_index >= 0.75:
            return "the viewing pattern stayed steady for most of the lesson"
        if seek_density is not None and seek_density <= 0.5:
            return "jumps and interruptions stayed low"
        return "attention stayed fairly consistent"

    if seek_density is not None and seek_density >= 1.0:
        base = "the viewing pattern looked fragmented with repeated jumps"
        if first_seek_time is not None and first_seek_time < 30.0:
            return f"{base} and early jumping"
        return base
    if attention_index is not None:
        return "steady attention was weaker than in a focused session"
    if play_pause_ratio is not None and play_pause_ratio < 0.5:
        return "play segments were short between interruptions"
    return "the viewing pattern looked fragmented rather than sustained"


def _rewatching_text(metrics: Dict[str, float], direction: str) -> str:
    rewatch_ratio = _metric(metrics, "rewatch_time_ratio")
    backward_seeks = _metric(metrics, "num_seek_backward")
    deep_flag = _metric(metrics, "deep_flag")
    micro_rewatch_count = _metric(metrics, "micro_rewatch_count")

    if direction == "positive":
        if rewatch_ratio is not None and rewatch_ratio > 0:
            return "some earlier sections were revisited for review"
        if backward_seeks is not None and backward_seeks >= 1:
            return "backward jumps were used to revisit earlier points"
        if deep_flag is not None and deep_flag >= 0.5:
            return "there was evidence of deeper review"
        if micro_rewatch_count is not None and micro_rewatch_count >= 1:
            return "you revisited short sections to check understanding"
        return "targeted review supported the session"

    if rewatch_ratio is not None and rewatch_ratio <= 0.01:
        return "there was little targeted review of unclear sections"
    if backward_seeks is not None and backward_seeks == 0:
        return "there was no meaningful backward review"
    return "targeted review was limited"


def _pausing_text(metrics: Dict[str, float], direction: str) -> str:
    pause_freq = _metric(metrics, "pause_freq_per_min")
    long_pause_ratio = _metric(metrics, "long_pause_ratio")
    total_pause_duration = _metric(metrics, "total_pause_duration_sec")

    if direction == "positive":
        if pause_freq is not None and 0 < pause_freq < 0.5:
            return "a few pauses were used without breaking the flow"
        if total_pause_duration is not None and total_pause_duration > 0:
            return "pauses looked deliberate rather than disruptive"
        return "pause behavior stayed controlled"

    if long_pause_ratio is not None and long_pause_ratio >= 0.5:
        return "long pauses broke the session flow"
    if pause_freq is not None and pause_freq >= 1.0:
        return "pauses were frequent enough to break continuity"
    if total_pause_duration is not None and total_pause_duration > 0:
        return "pauses added too much downtime to the session"
    return "pausing interrupted the session flow"


def _speed_text(metrics: Dict[str, float], direction: str) -> str:
    fast_ratio = _metric(metrics, "fast_ratio")
    avg_speed = _metric(metrics, "avg_playback_rate_when_playing")
    num_ratechange = _metric(metrics, "num_ratechange")
    speed_variance = _metric(metrics, "playback_speed_variance")

    if direction == "positive":
        if avg_speed is not None and 0.95 <= avg_speed <= 1.05:
            return "playback pace stayed close to normal"
        if fast_ratio is not None and fast_ratio <= 0.05:
            return "very little of the session ran at fast speed"
        return "playback pacing stayed stable"

    if fast_ratio is not None and fast_ratio > 0:
        return "too much of the session ran above normal speed"
    if avg_speed is not None and abs(avg_speed - 1.0) >= 0.1:
        return "playback pace likely reduced comprehension"
    if num_ratechange is not None and num_ratechange >= 2:
        return "playback speed changed often enough to break continuity"
    if speed_variance is not None and speed_variance > 0:
        return "playback pace changed enough to reduce continuity"
    return "playback pacing was unstable"


def _playback_quality_text(metrics: Dict[str, float], direction: str) -> str:
    buffering_time = _metric(metrics, "buffering_time_sec")
    buffering_events = _metric(metrics, "num_buffering_events")

    if direction == "positive":
        if (buffering_events is not None and buffering_events == 0) or (
            buffering_time is not None and buffering_time == 0
        ):
            return "playback stayed smooth without buffering"
        return "playback quality stayed stable"

    if buffering_time is not None and buffering_time > 0:
        return "buffering interruptions hurt continuity"
    if buffering_events is not None and buffering_events >= 1:
        return "playback buffered often enough to break flow"
    return "playback interruptions broke continuity"


def _describe_behavior(
    behavior: str,
    rows: List[Dict[str, float]],
    direction: str,
    session_features: Optional[Dict[str, float]] = None,
) -> str:
    metrics = _feature_map(rows, session_features=session_features)

    if behavior == "coverage":
        return _coverage_text(metrics, direction)
    if behavior == "skipping":
        return _skipping_text(metrics, direction)
    if behavior == "attention_consistency":
        return _attention_text(metrics, direction)
    if behavior == "rewatching":
        return _rewatching_text(metrics, direction)
    if behavior == "reflective_pausing":
        return _pausing_text(metrics, direction)
    if behavior == "speed_watching":
        return _speed_text(metrics, direction)
    if behavior == "playback_quality":
        return _playback_quality_text(metrics, direction)
    return _fallback_behavior_text(behavior, direction)


def resolve_engagement_status(
    engagement_score: Optional[float],
    engagement_status: Optional[EngagementStatus] = None,
    engagement_threshold: float = 0.5,
) -> EngagementStatus:
    """Resolve the final binary engagement status for the response."""
    if engagement_status in {"engaged", "not_engaged"}:
        return engagement_status

    score = 0.0 if engagement_score is None else max(0.0, min(1.0, float(engagement_score)))
    threshold = max(0.0, min(1.0, float(engagement_threshold)))
    return "engaged" if score >= threshold else "not_engaged"


def build_feature_reason(feature: str, feature_value: float, shap_value: float) -> str:
    """Return a concrete reason for a single contributor."""
    behavior = get_behavior(feature)
    direction = "positive" if shap_value >= 0 else "negative"
    sentence = _describe_behavior(
        behavior,
        rows=[{"feature": feature, "value": feature_value, "shap": shap_value}],
        direction=direction,
        session_features={feature: feature_value},
    ).strip()
    if not sentence:
        sentence = _fallback_behavior_text(behavior, direction)
    return sentence[0].upper() + sentence[1:] + "."


def build_user_explanation(
    shap_top_negative: List[Dict[str, float]],
    shap_top_positive: List[Dict[str, float]],
    engagement_score: Optional[float] = None,
    engagement_status: Optional[EngagementStatus] = None,
    engagement_threshold: float = 0.5,
    session_features: Optional[Dict[str, float]] = None,
) -> str:
    """Build the final learner-facing explanation for the score."""
    status = resolve_engagement_status(
        engagement_score=engagement_score,
        engagement_status=engagement_status,
        engagement_threshold=engagement_threshold,
    )
    score_text = _format_score_percent(engagement_score)
    message_parts: List[str] = []

    negative_clauses = [
        _describe_behavior(behavior, rows, "negative", session_features=session_features)
        for behavior, rows in _top_behavior_groups(shap_top_negative, limit=2)
    ]
    negative_clauses = [clause for clause in negative_clauses if clause]

    positive_limit = 1 if negative_clauses else 2
    positive_clauses = [
        _describe_behavior(behavior, rows, "positive", session_features=session_features)
        for behavior, rows in _top_behavior_groups(shap_top_positive, limit=positive_limit)
    ]
    positive_clauses = [clause for clause in positive_clauses if clause]

    if status == "engaged":
        message_parts.append(
            f"Good job. Your engagement score for this session is {score_text}."
        )
        if positive_clauses:
            message_parts.append(
                f"This looked engaged because {_join_clauses(positive_clauses)}."
            )
        else:
            message_parts.append(
                "This looked engaged because the session showed positive learning patterns overall."
            )
    else:
        message_parts.append(f"Your engagement score for this session is {score_text}.")
        if negative_clauses:
            message_parts.append(
                f"This looked less engaged because {_join_clauses(negative_clauses)}."
            )
        else:
            message_parts.append(
                "This looked less engaged because the session showed weak learning patterns overall."
            )

    return " ".join(message_parts)
