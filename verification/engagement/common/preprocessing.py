from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NumericPreprocessingArtifact:
    feature_columns: List[str]
    median_imputation_values: Dict[str, float]
    fitted_rows: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "feature_columns": self.feature_columns,
            "median_imputation_values": self.median_imputation_values,
            "fitted_rows": self.fitted_rows,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "NumericPreprocessingArtifact":
        feature_columns = [str(col) for col in payload.get("feature_columns", [])]
        medians_raw = payload.get("median_imputation_values", {})
        median_imputation_values = {
            str(col): float(value)
            for col, value in dict(medians_raw).items()
        }
        fitted_rows = int(payload.get("fitted_rows", 0))
        return cls(
            feature_columns=feature_columns,
            median_imputation_values=median_imputation_values,
            fitted_rows=fitted_rows,
        )


def select_numeric_feature_frame(
    df: pd.DataFrame,
    drop_cols: Sequence[str] = (),
) -> pd.DataFrame:
    x = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")
    x = x.select_dtypes(include=[np.number]).copy()
    x = x.replace([np.inf, -np.inf], np.nan)

    if x.empty:
        raise ValueError("No numeric features found after preprocessing.")

    return x


def fit_numeric_preprocessing(
    df: pd.DataFrame,
    drop_cols: Sequence[str] = (),
) -> NumericPreprocessingArtifact:
    x = select_numeric_feature_frame(df, drop_cols=drop_cols)
    feature_columns = list(x.columns)
    medians = x.median(numeric_only=True)

    median_imputation_values: Dict[str, float] = {}
    for col in feature_columns:
        median = medians.get(col, np.nan)
        median_imputation_values[col] = float(0.0 if pd.isna(median) else median)

    return NumericPreprocessingArtifact(
        feature_columns=feature_columns,
        median_imputation_values=median_imputation_values,
        fitted_rows=len(x),
    )


def transform_numeric_frame(
    df: pd.DataFrame,
    artifact: NumericPreprocessingArtifact,
    drop_cols: Sequence[str] = (),
) -> pd.DataFrame:
    if all(col in df.columns for col in artifact.feature_columns):
        x = df.reindex(columns=artifact.feature_columns).copy()
    else:
        x = select_numeric_feature_frame(df, drop_cols=drop_cols)
        x = x.reindex(columns=artifact.feature_columns)

    x = x.replace([np.inf, -np.inf], np.nan)

    for col in artifact.feature_columns:
        fill_value = artifact.median_imputation_values.get(col, 0.0)
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(fill_value)

    return x.astype(float)


def prepare_feature_array(
    features: Dict[str, float],
    artifact: NumericPreprocessingArtifact,
) -> np.ndarray:
    row = {
        col: float(features[col]) if np.isfinite(float(features[col])) else np.nan
        for col in artifact.feature_columns
    }
    frame = pd.DataFrame([row], columns=artifact.feature_columns)
    return transform_numeric_frame(frame, artifact).values


def save_preprocessing_artifact(
    artifact: NumericPreprocessingArtifact,
    path: Path,
) -> None:
    path.write_text(json.dumps(artifact.to_dict(), indent=2), encoding="utf-8")


def load_preprocessing_artifact(path: Path) -> NumericPreprocessingArtifact:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return NumericPreprocessingArtifact.from_dict(payload)
