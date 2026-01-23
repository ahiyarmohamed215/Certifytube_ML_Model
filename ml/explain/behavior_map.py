from __future__ import annotations

from typing import Dict


# Map each feature to a human-readable behavioral category.
# IMPORTANT: Keep this consistent with your thesis definitions.
FEATURE_TO_BEHAVIOR: Dict[str, str] = {
    # Skipping behaviour
    "seek_forward_count": "skipping",
    "seek_forward_sec": "skipping",
    "seek_forward_rate": "skipping",

    # Rewatching behaviour
    "seek_backward_count": "rewatching",
    "seek_backward_sec": "rewatching",
    "rewatch_ratio": "rewatching",

    # Reflective pausing
    "pause_count": "reflective_pausing",
    "pause_total_sec": "reflective_pausing",
    "pause_avg_sec": "reflective_pausing",

    # Speed watching
    "avg_playback_rate": "speed_watching",
    "rate_change_count": "speed_watching",

    # Coverage / completion
    "completion_ratio": "coverage",
    "watch_time_ratio": "coverage",
}


def get_behavior(feature_name: str) -> str:
    """
    Returns a behavior label for a feature.
    If unknown, returns 'other'.
    """
    return FEATURE_TO_BEHAVIOR.get(feature_name, "other")
