"""
SynDetector – auto-detects timestamp column, signal columns,
and sampling rate from a raw DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# column name fragments that suggest a timestamp
_TIMESTAMP_HINTS = (
    "time", "timestamp", "ts", "date", "datetime",
    "created", "recorded", "t",
)


class SynDetector:
    """
    Analyses a DataFrame to extract structural metadata.

    All methods are non-destructive – the DataFrame is never modified.
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df

    # – timestamp detection –––––––––––––––––––––––––––––––––––––––––––––––––––

    def find_timestamp_col(self) -> str | None:
        """
        Find the timestamp column by:
          1. Name heuristic (column name contains a timestamp hint)
          2. Dtype check (datetime64 dtype)
          3. Parseable string check (first non-null value parses as datetime)

        Returns the column name, or None if not found.
        """
        cols = list(self._df.columns)

        # 1. datetime dtype
        for col in cols:
            if pd.api.types.is_datetime64_any_dtype(self._df[col]):
                return col

        # 2. name heuristic (case-insensitive)
        col_lower = {col: col.lower() for col in cols}
        for col, lower in col_lower.items():
            for hint in _TIMESTAMP_HINTS:
                if hint == lower or lower.startswith(hint):
                    return col

        # 3. parseable string
        for col in cols:
            if self._df[col].dtype == object:
                sample = self._df[col].dropna().head(5)
                if len(sample) == 0:
                    continue
                try:
                    pd.to_datetime(sample)
                    return col
                except Exception:
                    continue

        return None

    # – signal column detection –––––––––––––––––––––––––––––––––––––––––––––––

    def find_signal_cols(
        self, exclude: list[str | None] | None = None
    ) -> list[str]:
        """
        Return names of numeric columns that are likely sensor signals.
        Excludes columns listed in `exclude` (e.g. the timestamp column).
        """
        excluded = {c for c in (exclude or []) if c is not None}
        return [
            col
            for col in self._df.columns
            if col not in excluded
            and pd.api.types.is_numeric_dtype(self._df[col])
        ]

    # – sampling rate inference –––––––––––––––––––––––––––––––––––––––––––––––

    def infer_sampling_rate(
        self, timestamp_col: str | None
    ) -> float | None:
        """
        Infer sampling rate in Hz from the timestamp column.

        Strategy:
          1. Parse the column as datetime
          2. Compute median time delta between consecutive rows
          3. Convert to Hz (1 / delta_seconds)

        Returns Hz as float, or None if inference fails.
        """
        if timestamp_col is None:
            return None

        col = self._df[timestamp_col].dropna()
        if len(col) < 2:
            return None

        # try to parse as datetime
        try:
            times = pd.to_datetime(col)
            deltas = times.diff().dropna()
            median_delta = deltas.median()
            seconds = median_delta.total_seconds()
            if seconds <= 0:
                return None
            return round(1.0 / seconds, 4)
        except Exception:
            pass

        # fallback: try as numeric (assume seconds)
        try:
            numeric = pd.to_numeric(col, errors="raise")
            deltas = numeric.diff().dropna()
            median_delta = float(deltas.median())
            if median_delta <= 0:
                return None
            return round(1.0 / median_delta, 4)
        except Exception:
            return None
