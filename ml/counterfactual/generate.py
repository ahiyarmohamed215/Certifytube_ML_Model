from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import xgboost as xgb

from ml.inference.load import load_model, load_feature_columns
from ml.explain.shap_explain import compute_local_shap, top_contributors

TARGET_THRESHOLD = 0.85

# Bounds for common features (tune as needed)
FEATURE_BOUNDS = {
    "completion_ratio": (0.0, 1.0),
    "watch_time_ratio": (0.0, 1.0),
    "seek_forward_ratio": (0.0, 1.0),
    "seek_backward_ratio": (0.0, 1.0),
    "rewatch_time_ratio": (0.0, 1.0),
    "pause_freq_per_min": (0.0, 60.0),
    "attention_index": (0.0, 1.0),
    "engagement_velocity": (0.0, 5.0),
    "avg_playback_rate_when_playing": (0.5, 3.0),
}


def _predict_score(features: Dict[str, float]) -> float:
    booster = load_model()
    cols = load_feature_columns()

    x = np.array([[float(features.get(c, 0.0)) for c in cols]], dtype=float)
    dmat = xgb.DMatrix(x, feature_names=cols)
    return float(booster.predict(dmat)[0])


def _clamp(feature: str, value: float) -> float:
    if feature in FEATURE_BOUNDS:
        lo, hi = FEATURE_BOUNDS[feature]
        return float(max(lo, min(hi, value)))
    return float(max(0.0, value))


def _default_step(feature: str, current: float) -> float:
    """
    Heuristic step sizes:
      - ratios: +/- 0.05
      - attention_index: +/- 0.05
      - playback rate: +/- 0.10
      - *_sec: +/- 10% (min 5)
      - *_count: +/- 10% (min 1)
      - fallback: +/- 10% (min 1)
    """
    if feature.endswith("_ratio"):
        return 0.05
    if feature == "attention_index":
        return 0.05
    if feature == "avg_playback_rate_when_playing":
        return 0.10
    if feature.endswith("_count"):
        return max(1.0, round(current * 0.10))
    if feature.endswith("_sec"):
        return max(5.0, round(current * 0.10, 1))
    return max(1.0, round(current * 0.10))


def _suggest_action(feature: str) -> str:
    # NOTE: This is INTERNAL ONLY (do not show to users).
    if feature.startswith("seek_forward") or feature.startswith("num_seek_forward"):
        return "reduce skipping"
    if feature == "avg_playback_rate_when_playing":
        return "lower playback speed"
    if feature.startswith("seek_backward") or feature.startswith("num_seek_backward") or feature.startswith("rewatch"):
        return "increase rewatching"
    if feature.startswith("pause") or feature.startswith("num_pause"):
        return "increase pausing"
    if feature in ("completion_ratio", "watch_time_ratio", "watch_time_sec", "completed_flag"):
        return "increase coverage"
    if feature in ("attention_index", "engagement_velocity"):
        return "increase attention consistency"
    return "adjust behavior"


def _direction(feature: str) -> int:
    """
    Returns:
      -1 to decrease feature
      +1 to increase feature

    This is a heuristic. For real robustness you’d learn monotonic constraints
    or build per-feature “good direction” metadata.
    """
    # Typically harmful when high
    if feature.startswith("num_seek_forward") or feature.startswith("total_seek_forward") or feature.startswith("avg_seek_forward"):
        return -1
    if feature == "avg_playback_rate_when_playing":
        return -1  # assuming too fast harms engagement
    if feature.startswith("buffering") or feature.startswith("num_buffering"):
        return -1  # buffering is generally bad
    # Typically helpful when high
    if feature in ("watch_time_ratio", "completion_ratio", "watch_time_sec"):
        return +1
    if feature in ("attention_index",):
        return +1
    # fallback: decrease
    return -1


def generate_counterfactual(
    features: Dict[str, float],
    target_threshold: float = TARGET_THRESHOLD,
    max_iters: int = 40,
    top_k_levers: int = 2,
) -> Optional[Dict]:
    """
    INTERNAL tool.
    Returns None if already meets threshold or if no counterfactual found.
    Otherwise returns:
    {
      "target_threshold": 0.85,
      "suggestions": [
         {"feature": "...", "current": x, "suggested": y, "action": "..."},
         ...
      ],
      "best_score": float
    }
    """
    base_score = _predict_score(features)
    if base_score >= target_threshold:
        return None

    shap_rows = compute_local_shap(features)
    top_negative, _ = top_contributors(shap_rows, k=top_k_levers)
    levers = [r["feature"] for r in top_negative if "feature" in r]

    current = dict(features)
    suggestions: Dict[str, Dict] = {}
    best_score = base_score

    for _ in range(max_iters):
        improved_any = False

        for f in levers:
            cur_val = float(current.get(f, 0.0))
            step = _default_step(f, cur_val)
            dirn = _direction(f)

            new_val = _clamp(f, cur_val + dirn * step)
            if new_val == cur_val:
                continue

            trial = dict(current)
            trial[f] = new_val
            trial_score = _predict_score(trial)

            if trial_score > best_score:
                best_score = trial_score
                current = trial
                improved_any = True

                suggestions[f] = {
                    "feature": f,
                    "current": float(features.get(f, 0.0)),
                    "suggested": float(new_val),
                    "action": _suggest_action(f),
                }

                if best_score >= target_threshold:
                    return {
                        "target_threshold": target_threshold,
                        "suggestions": list(suggestions.values()),
                        "best_score": best_score,
                    }

        if not improved_any:
            break

    if suggestions:
        return {
            "target_threshold": target_threshold,
            "suggestions": list(suggestions.values()),
            "best_score": best_score,
        }

    return None
