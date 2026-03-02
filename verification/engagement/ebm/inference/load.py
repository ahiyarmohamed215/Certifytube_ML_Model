import json
from pathlib import Path
import joblib
from app.core.settings import settings

ARTIFACTS_DIR = Path(settings.engagement_artifacts_ebm_dir)

_ebm_model = None
_ebm_feature_columns = None


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
        if not path.exists():
            raise FileNotFoundError("ebm_feature_columns.json not found.")
        with open(path, "r") as f:
            _ebm_feature_columns = json.load(f)
    return _ebm_feature_columns
