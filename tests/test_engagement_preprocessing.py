import math

import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from verification.engagement.common.dataset import load_csv_with_header_fallback
from verification.engagement.common.preprocessing import (
    NumericPreprocessingArtifact,
    fit_numeric_preprocessing,
    prepare_feature_array,
    transform_numeric_frame,
)


def test_fit_numeric_preprocessing_uses_training_medians_only():
    train_df = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, np.nan],
            "feature_b": [10.0, np.nan, 30.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "feature_a": [100.0, np.nan],
            "feature_b": [200.0, np.nan],
        }
    )

    artifact = fit_numeric_preprocessing(train_df)
    transformed = transform_numeric_frame(test_df, artifact)

    assert math.isclose(artifact.median_imputation_values["feature_a"], 1.5)
    assert math.isclose(artifact.median_imputation_values["feature_b"], 20.0)
    assert math.isclose(transformed.loc[1, "feature_a"], 1.5)
    assert math.isclose(transformed.loc[1, "feature_b"], 20.0)


def test_transform_numeric_frame_reindexes_missing_columns():
    artifact = NumericPreprocessingArtifact(
        feature_columns=["feature_a", "feature_b"],
        median_imputation_values={"feature_a": 1.0, "feature_b": 2.0},
        fitted_rows=3,
    )
    frame = pd.DataFrame({"feature_a": [5.0]})

    transformed = transform_numeric_frame(frame, artifact)

    assert list(transformed.columns) == ["feature_a", "feature_b"]
    assert math.isclose(transformed.loc[0, "feature_a"], 5.0)
    assert math.isclose(transformed.loc[0, "feature_b"], 2.0)


def test_prepare_feature_array_imputes_non_finite_values():
    artifact = NumericPreprocessingArtifact(
        feature_columns=["feature_a", "feature_b"],
        median_imputation_values={"feature_a": 7.0, "feature_b": 8.0},
        fitted_rows=5,
    )

    array = prepare_feature_array(
        {"feature_a": float("nan"), "feature_b": float("inf")},
        artifact,
    )

    assert array.shape == (1, 2)
    assert math.isclose(array[0, 0], 7.0)
    assert math.isclose(array[0, 1], 8.0)


def test_load_csv_with_header_fallback_handles_stray_first_row():
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "sample.csv"
        csv_path.write_text(
            "bad,data,row\n"
            "session_id,user_id,engagement_label\n"
            "s1,u1,1\n",
            encoding="utf-8",
        )

        loaded = load_csv_with_header_fallback(
            csv_path,
            required_columns=["session_id", "user_id", "engagement_label"],
        )

        assert list(loaded.columns) == ["session_id", "user_id", "engagement_label"]
        assert loaded.loc[0, "session_id"] == "s1"
