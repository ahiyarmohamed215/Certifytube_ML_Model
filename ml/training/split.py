from __future__ import annotations

from typing import Optional, Tuple
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def split_train_test(
    df: pd.DataFrame,
    label_col: str,
    group_col: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, test_df)

    - If group_col is provided, split by group (e.g., user_id) to prevent leakage.
    - Otherwise use stratified random split.
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")

    if group_col and group_col in df.columns:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        groups = df[group_col].astype(str)
        train_idx, test_idx = next(splitter.split(df, df[label_col], groups))
        return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

    # fallback: stratified split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
