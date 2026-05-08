"""
SynValidator – checks a DataFrame for common data quality issues.
Returns warnings (non-fatal) rather than raising exceptions,
so the pipeline can proceed with degraded data and inform the user.
"""

from __future__ import annotations

import pandas as pd


# minimum rows to proceed without a warning
_MIN_ROWS = 50


class SynValidator:
    """
    Runs data quality checks and collects warnings.
    Does not raise – always returns a list of warning strings.
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def run(
        self,
        timestamp_col: str | None,
        signal_cols: list[str],
    ) -> list[str]:
        """
        Run all checks and return a list of warning strings.
        Empty list means no issues detected.
        """
        warnings: list[str] = []
        warnings.extend(self._check_min_rows())
        warnings.extend(self._check_all_nan_columns())
        warnings.extend(self._check_constant_columns(signal_cols))
        warnings.extend(self._check_high_missing(signal_cols))
        warnings.extend(self._check_duplicate_rows())
        return warnings

    # – individual checks –––––––––––––––––––––––––––––––––––––––––––––––––––––

    def _check_min_rows(self) -> list[str]:
        if len(self._df) < _MIN_ROWS:
            return [
                f"Dataset has only {len(self._df)} rows. "
                f"Generative models perform best with at least {_MIN_ROWS} rows. "
                "Results may be poor."
            ]
        return []

    def _check_all_nan_columns(self) -> list[str]:
        warnings = []
        for col in self._df.columns:
            if self._df[col].isna().all():
                warnings.append(
                    f"Column '{col}' is entirely NaN and will be dropped "
                    "by the imputer."
                )
        return warnings

    def _check_constant_columns(self, signal_cols: list[str]) -> list[str]:
        warnings = []
        for col in signal_cols:
            if self._df[col].dropna().nunique() == 1:
                warnings.append(
                    f"Column '{col}' is constant (single unique value). "
                    "It carries no signal information."
                )
        return warnings

    def _check_high_missing(self, signal_cols: list[str]) -> list[str]:
        warnings = []
        for col in signal_cols:
            pct = self._df[col].isna().mean() * 100
            if pct > 30:
                warnings.append(
                    f"Column '{col}' is {pct:.1f}% missing. "
                    "Imputation quality may be low above 30%."
                )
        return warnings

    def _check_duplicate_rows(self) -> list[str]:
        n_dupes = self._df.duplicated().sum()
        if n_dupes > 0:
            return [
                f"{n_dupes} duplicate rows detected. "
                "These will be included in training unless removed."
            ]
        return []
