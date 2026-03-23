from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def load_csv_with_header_fallback(path: str | Path, required_columns: Sequence[str]) -> pd.DataFrame:
    csv_path = Path(path)
    last_columns = []

    for header_row in (0, 1):
        df = pd.read_csv(csv_path, header=header_row)
        last_columns = list(df.columns)
        if all(column in df.columns for column in required_columns):
            return df

    raise ValueError(
        f"Could not load required columns {list(required_columns)} from {csv_path}. "
        f"Last detected columns were: {last_columns}"
    )
