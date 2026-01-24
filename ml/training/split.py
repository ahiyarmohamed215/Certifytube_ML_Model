from __future__ import annotations

from typing import Optional, Tuple
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def split_train_test(
    df: pd.DataFrame,
    label_col: str,
    group_col: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified + Grouped split:
      - preserves label distribution
      - prevents leakage between groups (e.g., user_id)

    Uses StratifiedGroupKFold with n_splits ≈ 1/test_size.
    For 80/20 => n_splits=5 (exact).
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found.")
    if not group_col or group_col not in df.columns:
        raise ValueError("group_col must be provided and exist in df for StratifiedGroupKFold.")

    n_splits = int(round(1.0 / test_size))
    if abs((1.0 / n_splits) - test_size) > 1e-6:
        raise ValueError(f"test_size={test_size} not supported cleanly; use values like 0.2, 0.25, 0.1")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    X_dummy = df.drop(columns=[label_col])
    y = df[label_col].astype(int)
    groups = df[group_col].astype(str)

    # Use first fold as test for determinism
    train_idx, test_idx = next(sgkf.split(X_dummy, y, groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df
