import json
from pathlib import Path
import joblib

ARTIFACTS_DIR = Path("ml/artifacts")

_model = None
_feature_columns = None
_metadata = None


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
        if not path.exists():
            raise FileNotFoundError("feature_columns.json not found.")
        with open(path, "r") as f:
            _feature_columns = json.load(f)
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


