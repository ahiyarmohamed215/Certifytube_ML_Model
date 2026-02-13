"""
predict_ebm.py – EBM inference (scikit-learn interface, no DMatrix).
"""
from __future__ import annotations

import numpy as np

from ml.inference.load import load_ebm_model, load_ebm_feature_columns
from ml.inference.validate import validate_features

ENGAGEMENT_THRESHOLD = 0.85


def predict_engagement_ebm(features: dict) -> dict:
    """
    Predict engagement using the trained EBM model.

    Returns the same shape as predict_engagement() for XGBoost so the
    API layer can treat both interchangeably.
    """
    ebm = load_ebm_model()
    feature_columns = load_ebm_feature_columns()

    validate_features(features, feature_columns)

    x = np.array([[features[col] for col in feature_columns]], dtype=float)

    score = float(ebm.predict_proba(x)[0, 1])
    status = "ENGAGED" if score >= ENGAGEMENT_THRESHOLD else "NOT_ENGAGED"

    return {
        "engagement_score": score,
        "threshold": ENGAGEMENT_THRESHOLD,
        "status": status,
    }
