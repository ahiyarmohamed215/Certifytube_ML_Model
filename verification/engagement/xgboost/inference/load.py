import json
from pathlib import Path
import joblib
from app.core.settings import settings
from verification.engagement.common.preprocessing import load_preprocessing_artifact

ARTIFACTS_DIR = Path(settings.engagement_artifacts_xgboost_dir)

_model = None
_feature_columns = None
_metadata = None
_preprocessing = None


def load_model():
    global _model
    if _model is None:
        model_path = ARTIFACTS_DIR / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError("model.joblib not found. Train the model first.")
        _model = joblib.load(model_path)
    return _model


def load_feature_columns():
    global _feature_columns
    if _feature_columns is None:
        path = ARTIFACTS_DIR / "feature_columns.json"
        preprocessing_path = ARTIFACTS_DIR / "preprocessing.json"
        if preprocessing_path.exists():
            _feature_columns = load_preprocessing().feature_columns
        elif path.exists():
            with open(path, "r") as f:
                _feature_columns = json.load(f)
        else:
            raise FileNotFoundError("feature_columns.json not found.")
    return _feature_columns


def load_metadata():
    global _metadata
    if _metadata is None:
        path = ARTIFACTS_DIR / "metadata.json"
        if not path.exists():
            _metadata = {}
        else:
            with open(path, "r") as f:
                _metadata = json.load(f)
    return _metadata


def load_preprocessing():
    global _preprocessing
    if _preprocessing is None:
        path = ARTIFACTS_DIR / "preprocessing.json"
        if not path.exists():
            raise FileNotFoundError("preprocessing.json not found. Train the model first.")
        _preprocessing = load_preprocessing_artifact(path)
    return _preprocessing
