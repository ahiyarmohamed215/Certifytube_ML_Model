import json
from pathlib import Path
import joblib
from app.core.settings import settings
from verification.engagement.common.preprocessing import load_preprocessing_artifact

ARTIFACTS_DIR = Path(settings.engagement_artifacts_ebm_dir)

_ebm_model = None
_ebm_feature_columns = None
_ebm_preprocessing = None


def load_ebm_model():
    global _ebm_model
    if _ebm_model is None:
        model_path = ARTIFACTS_DIR / "ebm_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError("ebm_model.joblib not found. Run train_ebm first.")
        _ebm_model = joblib.load(model_path)
    return _ebm_model


def load_ebm_feature_columns():
    global _ebm_feature_columns
    if _ebm_feature_columns is None:
        path = ARTIFACTS_DIR / "ebm_feature_columns.json"
        preprocessing_path = ARTIFACTS_DIR / "ebm_preprocessing.json"
        if preprocessing_path.exists():
            _ebm_feature_columns = load_ebm_preprocessing().feature_columns
        elif path.exists():
            with open(path, "r") as f:
                _ebm_feature_columns = json.load(f)
        else:
            raise FileNotFoundError("ebm_feature_columns.json not found.")
    return _ebm_feature_columns


def load_ebm_preprocessing():
    global _ebm_preprocessing
    if _ebm_preprocessing is None:
        path = ARTIFACTS_DIR / "ebm_preprocessing.json"
        if not path.exists():
            raise FileNotFoundError("ebm_preprocessing.json not found. Run train_ebm first.")
        _ebm_preprocessing = load_preprocessing_artifact(path)
    return _ebm_preprocessing
