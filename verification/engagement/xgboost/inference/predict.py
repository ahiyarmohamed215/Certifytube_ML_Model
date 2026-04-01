import xgboost as xgb

from verification.engagement.common.preprocessing import prepare_feature_array
from verification.engagement.xgboost.inference.load import (
    load_model,
    load_feature_columns,
    load_preprocessing,
)
from verification.engagement.common.validate import validate_features


def predict_engagement(features: dict):
    """Return the continuous engagement score from the trained XGBoost regressor."""
    booster = load_model()  # xgboost.Booster
    feature_columns = load_feature_columns()
    preprocessing = load_preprocessing()

    validate_features(features, feature_columns)

    x = prepare_feature_array(features, preprocessing)
    dmat = xgb.DMatrix(x, feature_names=feature_columns)

    score = float(booster.predict(dmat)[0])

    return {
        "engagement_score": score,
    }
