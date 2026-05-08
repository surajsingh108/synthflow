"""
SynPatternDetector – classifies the missing data mechanism
(MCAR, MAR, MNAR) for each column independently.

Detection logic (in priority order per column):

  Step 0: trivial – < 1% missing – MCAR immediately, no test needed.

  Step 1: MNAR heuristic – check for monotone or tail-concentrated
          missingness. If > 60% of missing values are in the first
          or last 20% of rows, classify as MNAR.

  Step 2: MAR test – build a binary missingness indicator (1=missing,
          0=present) and compute Pearson correlation between it and
          every other numeric column. If any |correlation| > MAR_THRESHOLD,
          classify as MAR.

  Step 3: default – MCAR.

Note: MNAR cannot be statistically confirmed from observed data alone.
The monotone heuristic is a practical proxy, not a formal test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# – Tunable thresholds –––––––––––––––––––––––––––––––––––––––––––––––––––––––

# Below this % missing, skip detection and return MCAR
TRIVIAL_MISSING_PCT: float = 1.0

# Pearson |correlation| above this – classify as MAR
MAR_CORRELATION_THRESHOLD: float = 0.1

# Fraction of total rows considered "tail" for MNAR heuristic
MNAR_TAIL_FRACTION: float = 0.20

# Fraction of missing values that must be in tail to trigger MNAR
MNAR_TAIL_CONCENTRATION: float = 0.60


# – Result types –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

Pattern = str  # "MCAR" | "MAR" | "MNAR" | "none"


@dataclass
class ColumnResult:
    """
    Detection result for a single column.

    Attributes:
        column          : column name
        pattern         : detected pattern ("MCAR", "MAR", "MNAR", "none")
        missing_count   : number of missing values
        missing_pct     : percentage missing (0.0 – 100.0)
        mar_max_corr    : highest |correlation| found during MAR test,
                          or None if test was not run
        mnar_tail_frac  : fraction of missing values in tail,
                          or None if test was not run
        detection_path  : human-readable string describing which branch
                          triggered the classification
    """

    column: str
    pattern: Pattern
    missing_count: int
    missing_pct: float
    mar_max_corr: float | None = None
    mnar_tail_frac: float | None = None
    detection_path: str = ""


@dataclass
class DetectionResult:
    """
    Full detection result across all columns.

    Attributes:
        columns     : dict mapping column name → ColumnResult
        summary     : dict mapping pattern → list of column names
    """

    columns: dict[str, ColumnResult] = field(default_factory=dict)

    @property
    def summary(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {
            "none": [], "MCAR": [], "MAR": [], "MNAR": []
        }
        for col, result in self.columns.items():
            out[result.pattern].append(col)
        return out

    def pattern_for(self, column: str) -> Pattern:
        """Return the pattern for a single column."""
        return self.columns[column].pattern

    def as_dict(self) -> dict[str, Pattern]:
        """Return {column: pattern} mapping."""
        return {col: r.pattern for col, r in self.columns.items()}


# – Detector ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

class SynPatternDetector:
    """
    Detects the missing data mechanism per column.

    Usage:
        detector = SynPatternDetector()
        result = detector.detect(df)
        print(result.summary)
        # {'none': ['timestamp'], 'MCAR': ['accel_x'], 'MAR': ['rpm'], ...}

        print(result.pattern_for("accel_x"))
        # 'MCAR'
    """

    def __init__(
        self,
        mar_threshold: float = MAR_CORRELATION_THRESHOLD,
        mnar_tail_fraction: float = MNAR_TAIL_FRACTION,
        mnar_tail_concentration: float = MNAR_TAIL_CONCENTRATION,
        trivial_pct: float = TRIVIAL_MISSING_PCT,
    ):
        self.mar_threshold = mar_threshold
        self.mnar_tail_fraction = mnar_tail_fraction
        self.mnar_tail_concentration = mnar_tail_concentration
        self.trivial_pct = trivial_pct

    # – public API –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def detect(self, df: pd.DataFrame) -> DetectionResult:
        """
        Run detection on all columns of df.
        Returns a DetectionResult with per-column results.
        """
        result = DetectionResult()
        numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
        ]

        for col in df.columns:
            col_result = self._detect_column(df, col, numeric_cols)
            result.columns[col] = col_result

        return result

    def detect_column(
        self, df: pd.DataFrame, col: str
    ) -> ColumnResult:
        """
        Run detection on a single column.
        Convenience method for one-off checks.
        """
        numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
        ]
        return self._detect_column(df, col, numeric_cols)

    # – internal –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def _detect_column(
        self,
        df: pd.DataFrame,
        col: str,
        numeric_cols: list[str],
    ) -> ColumnResult:
        series = df[col]
        n = len(series)
        missing_count = int(series.isna().sum())
        missing_pct = round(missing_count / n * 100, 2) if n > 0 else 0.0

        # – no missing ––––––––––––––––––––––––––––––––––––––––––––––––––––
        if missing_count == 0:
            return ColumnResult(
                column=col,
                pattern="none",
                missing_count=0,
                missing_pct=0.0,
                detection_path="no missing values",
            )

        # – trivial – very few missing ––––––––––––––––––––––––––––––––––––
        if missing_pct < self.trivial_pct:
            return ColumnResult(
                column=col,
                pattern="MCAR",
                missing_count=missing_count,
                missing_pct=missing_pct,
                detection_path=f"trivial (<{self.trivial_pct}% missing) – MCAR",
            )

        # – MNAR heuristic ––––––––––––––––––––––––––––––––––––––––––––––––
        mnar_frac = self._tail_concentration(series)
        if mnar_frac >= self.mnar_tail_concentration:
            return ColumnResult(
                column=col,
                pattern="MNAR",
                missing_count=missing_count,
                missing_pct=missing_pct,
                mnar_tail_frac=mnar_frac,
                detection_path=(
                    f"tail concentration {mnar_frac:.2f} "
                    f">= {self.mnar_tail_concentration} – MNAR"
                ),
            )

        # – MAR test ––––––––––––––––––––––––––––––––––––––––––––––––––––––
        other_numeric = [
            c for c in numeric_cols
            if c != col and df[c].notna().sum() > 1
        ]
        max_corr = self._mar_correlation(series, df, other_numeric)

        if max_corr is not None and max_corr >= self.mar_threshold:
            return ColumnResult(
                column=col,
                pattern="MAR",
                missing_count=missing_count,
                missing_pct=missing_pct,
                mar_max_corr=max_corr,
                detection_path=(
                    f"max |correlation| {max_corr:.3f} "
                    f">= {self.mar_threshold} – MAR"
                ),
            )

        # – default –––––––––––––––––––––––––––––––––––––––––––––––––––––––
        return ColumnResult(
            column=col,
            pattern="MCAR",
            missing_count=missing_count,
            missing_pct=missing_pct,
            mar_max_corr=max_corr,
            mnar_tail_frac=mnar_frac,
            detection_path="no MAR/MNAR signal detected – MCAR",
        )

    def _tail_concentration(self, series: pd.Series) -> float:
        """
        Return the fraction of missing values that fall in the
        first or last (tail_fraction * n) rows.

        High concentration – monotone missingness – MNAR heuristic.
        """
        n = len(series)
        tail_n = max(1, int(n * self.mnar_tail_fraction))
        missing_mask = series.isna()
        total_missing = missing_mask.sum()
        if total_missing == 0:
            return 0.0
        head_missing = missing_mask.iloc[:tail_n].sum()
        tail_missing = missing_mask.iloc[-tail_n:].sum()
        return float((head_missing + tail_missing) / total_missing)

    def _mar_correlation(
        self,
        series: pd.Series,
        df: pd.DataFrame,
        other_cols: list[str],
    ) -> float | None:
        """
        Compute maximum absolute Pearson correlation between the
        binary missingness indicator of `series` and each column
        in `other_cols`.

        Returns the max |correlation|, or None if no other columns exist.
        """
        if not other_cols:
            return None

        indicator = series.isna().astype(float)
        max_corr = 0.0

        for other_col in other_cols:
            other = df[other_col]
            # align on non-null rows of the other column
            valid = other.notna()
            if valid.sum() < 10:
                continue
            ind_aligned = indicator[valid]
            other_aligned = other[valid]
            if ind_aligned.std() == 0 or other_aligned.std() == 0:
                continue
            try:
                corr = float(
                    np.corrcoef(ind_aligned.values, other_aligned.values)[0, 1]
                )
                max_corr = max(max_corr, abs(corr))
            except Exception:
                continue

        return round(max_corr, 4)
