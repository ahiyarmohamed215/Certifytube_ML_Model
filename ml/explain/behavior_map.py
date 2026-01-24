from __future__ import annotations

from typing import Dict

FEATURE_TO_BEHAVIOR: Dict[str, str] = {
    # Coverage / completion
    "watch_time_ratio": "coverage",
    "completion_ratio": "coverage",
    "watch_time_sec": "coverage",
    "completed_flag": "coverage",
    "last_position_sec": "coverage",

    # Skipping (forward seeks + skip ratios)
    "num_seek_forward": "skipping",
    "total_seek_forward_sec": "skipping",
    "avg_seek_forward_sec": "skipping",
    "largest_forward_seek_sec": "skipping",
    "seek_forward_ratio": "skipping",
    "skip_time_ratio": "skipping",
    "early_skip_flag": "skipping",
    "skim_flag": "skipping",

    # Rewatching (backward seeks + rewatch ratios)
    "num_seek_backward": "rewatching",
    "total_seek_backward_sec": "rewatching",
    "avg_seek_backward_sec": "rewatching",
    "largest_backward_seek_sec": "rewatching",
    "seek_backward_ratio": "rewatching",
    "rewatch_time_ratio": "rewatching",
    "rewatch_to_skip_ratio": "rewatching",
    "deep_flag": "rewatching",

    # Reflective pausing
    "num_pause": "reflective_pausing",
    "total_pause_duration_sec": "reflective_pausing",
    "avg_pause_duration_sec": "reflective_pausing",
    "median_pause_duration_sec": "reflective_pausing",
    "long_pause_count": "reflective_pausing",
    "long_pause_ratio": "reflective_pausing",
    "pause_freq_per_min": "reflective_pausing",

    # Speed watching / playback behaviour
    "avg_playback_rate_when_playing": "speed_watching",
    "fast_ratio": "speed_watching",
    "slow_ratio": "speed_watching",
    "playback_speed_variance": "speed_watching",
    "num_ratechange": "speed_watching",
    "time_at_speed_lt1x_sec": "speed_watching",
    "time_at_speed_1x_sec": "speed_watching",
    "time_at_speed_gt1x_sec": "speed_watching",
    "unique_speed_levels": "speed_watching",

    # Attention / quality signals
    "attention_index": "attention_consistency",
    "engagement_velocity": "attention_consistency",
    "seek_density_per_min": "attention_consistency",
    "play_pause_ratio": "attention_consistency",

    # Buffering (not learner intent, but affects experience)
    "num_buffering_events": "playback_quality",
    "buffering_time_sec": "playback_quality",
    "buffering_freq_per_min": "playback_quality",
}


def get_behavior(feature_name: str) -> str:
    return FEATURE_TO_BEHAVIOR.get(feature_name, "other")
