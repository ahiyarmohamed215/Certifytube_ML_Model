import numpy as np
import xgboost as xgb

from ml.inference.load import load_model, load_feature_columns
from ml.inference.validate import validate_features

ENGAGEMENT_THRESHOLD = 0.85


def predict_engagement(features: dict):
    booster = load_model()  # xgboost.Booster
    feature_columns = load_feature_columns()

    validate_features(features, feature_columns)

    x = np.array([[features[col] for col in feature_columns]], dtype=float)
    dmat = xgb.DMatrix(x, feature_names=feature_columns)

    score = float(booster.predict(dmat)[0])
    status = "ENGAGED" if score >= ENGAGEMENT_THRESHOLD else "NOT_ENGAGED"

    return {
        "engagement_score": score,
        "threshold": ENGAGEMENT_THRESHOLD,
        "status": status,
    }


def predict_engagement_routed(features: dict, model_type: str = "xgboost"):
    """
    Route prediction to XGBoost or EBM based on model_type.

    Both return the same dict shape:
      {"engagement_score": float, "threshold": float, "status": str}
    """
    if model_type == "ebm":
        from ml.inference.predict_ebm import predict_engagement_ebm
        return predict_engagement_ebm(features)
    return predict_engagement(features)

