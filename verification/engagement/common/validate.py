from typing import Dict, List


class FeatureValidationError(Exception):
    pass


def validate_features(
    features: Dict[str, float],
    expected_columns: List[str],
):
    if not isinstance(features, dict):
        raise FeatureValidationError("Features must be a dictionary")

    missing = [c for c in expected_columns if c not in features]
    extra = [k for k in features.keys() if k not in expected_columns]

    if missing:
        raise FeatureValidationError(f"Missing features: {missing}")

    if extra:
        raise FeatureValidationError(f"Unexpected features: {extra}")

    # Type safety (basic)
    for k, v in features.items():
        if not isinstance(v, (int, float)):
            raise FeatureValidationError(f"Feature '{k}' must be numeric")
