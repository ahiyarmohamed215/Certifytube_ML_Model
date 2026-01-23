import numpy as np

from ml.inference.load import load_model, load_feature_columns
from ml.inference.validate import validate_features


ENGAGEMENT_THRESHOLD = 0.85


def predict_engagement(features: dict):
    model = load_model()
    feature_columns = load_feature_columns()

    # Validate
    validate_features(features, feature_columns)

    # Arrange features in training order
    x = np.array([[features[col] for col in feature_columns]])

    # Predict probability (class 1 = engaged)
    score = float(model.predict_proba(x)[0][1])

    status = "ENGAGED" if score >= ENGAGEMENT_THRESHOLD else "NOT_ENGAGED"

    return {
        "engagement_score": score,
        "threshold": ENGAGEMENT_THRESHOLD,
        "status": status,
    }
