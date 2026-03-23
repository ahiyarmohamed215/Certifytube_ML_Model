"""
predict_ebm.py – EBM inference (scikit-learn interface, no DMatrix).
"""
from __future__ import annotations

from verification.engagement.common.preprocessing import prepare_feature_array
from verification.engagement.ebm.inference.load import load_ebm_model, load_ebm_feature_columns
from verification.engagement.ebm.inference.load import load_ebm_preprocessing
from verification.engagement.common.validate import validate_features


def predict_engagement_ebm(features: dict) -> dict:
    """
    Predict engagement using the trained EBM model.

    Returns the same shape as predict_engagement() for XGBoost so the
    API layer can treat both interchangeably:
      {"engagement_score": float}
    """
    ebm = load_ebm_model()
    feature_columns = load_ebm_feature_columns()
    preprocessing = load_ebm_preprocessing()

    validate_features(features, feature_columns)

    x = prepare_feature_array(features, preprocessing)

    score = float(ebm.predict_proba(x)[0, 1])

    return {
        "engagement_score": score,
    }
