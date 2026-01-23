from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from ml.inference.load import load_model, load_feature_columns
from ml.explain.shap_explain import compute_local_shap, top_contributors


TARGET_THRESHOLD = 0.85


# Reasonable bounds for common features (tune later if needed)
FEATURE_BOUNDS = {
    "avg_playback_rate": (0.75, 2.0),
    "completion_ratio": (0.0, 1.0),
    "watch_time_ratio": (0.0, 1.0),
    "rewatch_ratio": (0.0, 1.0),
    # counts and seconds: lower bound 0, upper bound left flexible
}


def _predict_score(features: Dict[str, float]) -> float:
    model = load_model()
    cols = load_feature_columns()
    x = np.array([[features[c] for c in cols]], dtype=float)
    return float(model.predict_proba(x)[0][1])


def _clamp(feature: str, value: float) -> float:
    if feature in FEATURE_BOUNDS:
        lo, hi = FEATURE_BOUNDS[feature]
        return float(max(lo, min(hi, value)))
    # Default clamp for non-negative features
    return float(max(0.0, value))


def _step_down(value: float, step: float) -> float:
    return value - step


def _step_up(value: float, step: float) -> float:
    return value + step


def _default_step(feature: str, current: float) -> float:
    """
    Heuristic step sizes:
    - playback rate: reduce in 0.1 increments
    - ratios: increase in 0.05 increments
    - counts: decrease by ~10% (at least 1)
    - seconds: increase/decrease by 10% (min 5)
    """
    if feature == "avg_playback_rate":
        return 0.10
    if feature.endswith("_ratio"):
        return 0.05
    if feature.endswith("_count"):
        return max(1.0, round(current * 0.10))
    if feature.endswith("_sec"):
        return max(5.0, round(current * 0.10, 1))
    # fallback
    return max(1.0, round(current * 0.10))


def _suggest_action(feature: str) -> str:
    if feature.startswith("seek_forward"):
        return "reduce skipping"
    if feature == "avg_playback_rate":
        return "lower playback speed"
    if feature.startswith("seek_backward") or feature == "rewatch_ratio":
        return "increase rewatching"
    if feature.startswith("pause"):
        return "increase reflective pausing"
    if feature in ("completion_ratio", "watch_time_ratio"):
        return "increase content coverage"
    return "adjust behavior"


def generate_counterfactual(
    features: Dict[str, float],
    target_threshold: float = TARGET_THRESHOLD,
    max_iters: int = 40,
    top_k_levers: int = 2,
) -> Optional[Dict]:
    """
    Returns None if already meets threshold or if no reasonable counterfactual found.
    Otherwise returns:
    {
      "target_threshold": 0.85,
      "suggestions": [
         {"feature": "...", "current": x, "suggested": y, "action": "..."},
         ...
      ]
    }
    """
    base_score = _predict_score(features)
    if base_score >= target_threshold:
        return None

    # Use SHAP to find most negative contributors (good levers)
    shap_rows = compute_local_shap(features)
    top_negative, _ = top_contributors(shap_rows, k=top_k_levers)

    levers = [r["feature"] for r in top_negative]

    # Work on a copy
    current = dict(features)
    suggestions = {}

    # Strategy:
    # For each lever, apply small changes iteratively and check score.
    # - For negative contributors like seek_forward_count or avg_playback_rate: step DOWN
    # - For positive-type levers (rare in negative list, but just in case): step UP
    for _ in range(max_iters):
        improved = False

        for f in levers:
            cur_val = float(current[f])
            step = _default_step(f, cur_val)

            # Decide direction:
            # If feature is usually "bad when high", reduce it.
            reduce_features = (
                f.startswith("seek_forward")
                or f == "avg_playback_rate"
                or f.endswith("_count")
                and f.startswith("seek_forward")
            )

            # Otherwise if it's a ratio/positive behavior, increase it.
            if reduce_features or f == "avg_playback_rate" or f.startswith("seek_forward"):
                new_val = _step_down(cur_val, step)
            else:
                new_val = _step_up(cur_val, step)

            new_val = _clamp(f, new_val)

            if new_val == cur_val:
                continue

            trial = dict(current)
            trial[f] = new_val
            trial_score = _predict_score(trial)

            # Accept change only if it improves score
            if trial_score > base_score:
                base_score = trial_score
                current = trial
                improved = True

                # store suggestion (keep latest best)
                suggestions[f] = {
                    "feature": f,
                    "current": float(features[f]),
                    "suggested": float(new_val),
                    "action": _suggest_action(f),
                }

                # stop if threshold met
                if base_score >= target_threshold:
                    return {
                        "target_threshold": target_threshold,
                        "suggestions": list(suggestions.values()),
                    }

        if not improved:
            break

    # If we couldn't reach threshold, still return best-effort suggestions if any
    if suggestions:
        return {
            "target_threshold": target_threshold,
            "suggestions": list(suggestions.values()),
        }

    return None
