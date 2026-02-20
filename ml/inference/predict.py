import numpy as np
import xgboost as xgb

from ml.inference.load import load_model, load_feature_columns
from ml.inference.validate import validate_features


def predict_engagement(features: dict):
    booster = load_model()  # xgboost.Booster
    feature_columns = load_feature_columns()

    validate_features(features, feature_columns)

    x = np.array([[features[col] for col in feature_columns]], dtype=float)
    dmat = xgb.DMatrix(x, feature_names=feature_columns)

    score = float(booster.predict(dmat)[0])

    return {
        "engagement_score": score,
    }


def predict_engagement_routed(features: dict, model_type: str = "xgboost"):
    """
    Route prediction to XGBoost or EBM based on model_type.

    Returns: {"engagement_score": float}
    """
    if model_type == "ebm":
        from ml.inference.predict_ebm import predict_engagement_ebm
        return predict_engagement_ebm(features)
    return predict_engagement(features)
