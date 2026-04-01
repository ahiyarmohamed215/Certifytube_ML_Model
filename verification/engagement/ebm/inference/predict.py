"""
predict_ebm.py – EBM inference (scikit-learn interface, no DMatrix).
"""
from __future__ import annotations

import numpy as np

from verification.engagement.common.preprocessing import prepare_feature_array
from verification.engagement.ebm.inference.load import load_ebm_model, load_ebm_feature_columns
from verification.engagement.ebm.inference.load import load_ebm_preprocessing
from verification.engagement.common.validate import validate_features


def predict_engagement_ebm(features: dict) -> dict:
    """
    Predict engagement using the trained EBM model (regression).

    Returns the same shape as predict_engagement() for XGBoost so the
    API layer can treat both interchangeably:
      {"engagement_score": float}
    """
    ebm = load_ebm_model()
    feature_columns = load_ebm_feature_columns()
    preprocessing = load_ebm_preprocessing()

    validate_features(features, feature_columns)

    x = prepare_feature_array(features, preprocessing)

    # The regressor emits the score directly through predict().
    raw_score = float(ebm.predict(x)[0])
    # Clip to [0, 1] since regression output may slightly exceed bounds.
    score = float(np.clip(raw_score, 0.0, 1.0))

    return {
        "engagement_score": score,
    }
