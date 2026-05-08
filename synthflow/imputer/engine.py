"""
SynImputer – orchestrates pattern detection, strategy selection,
and imputation to produce a zero-NaN DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from synthflow.exceptions import SynImpError
from synthflow.imputer.missing_pattern import SynPatternDetector
from synthflow.imputer.strategies import (
    auto_strategy,
    apply_forward_fill,
    apply_spline,
    apply_ml_imputer,
)


# ── Result type ────────────────────────────────────────────────────────────

@dataclass
class ColumnImputation:
    """Per-column imputation details for the report."""
    column: str
    missing_before: int
    missing_pct_before: float
    pattern: str
    strategy: str


@dataclass
class ImputeResult:
    """
    Result returned by SynImputer.impute().

    Attributes:
        data                : imputed DataFrame (zero NaNs guaranteed)
        total_missing_before: total NaN count before imputation
        missing_pct_before  : overall % missing before
        missing_pct_after   : overall % missing after (should be 0.0)
        pattern_detected    : overall pattern label (most common non-none)
        strategy_used       : overall strategy label (most common used)
        columns             : per-column imputation details
    """
    data: pd.DataFrame
    total_missing_before: int
    missing_pct_before: float
    missing_pct_after: float
    pattern_detected: str
    strategy_used: str
    columns: list[ColumnImputation] = field(default_factory=list)

    def as_dict(self) -> dict:
        """Return report as plain dict for JSON serialisation."""
        return {
            "total_missing_before": self.total_missing_before,
            "missing_pct_before": self.missing_pct_before,
            "missing_pct_after": self.missing_pct_after,
            "pattern_detected": self.pattern_detected,
            "strategy_used": self.strategy_used,
            "columns_affected": {
                c.column: {
                    "missing": c.missing_before,
                    "pattern": c.pattern,
                    "strategy": c.strategy,
                }
                for c in self.columns
                if c.missing_before > 0
            },
        }


# ── Simple strategy names ──────────────────────────────────────────────────

_SIMPLE_STRATEGIES = {"forward_fill", "spline"}
_ML_STRATEGIES = {"knn", "mice", "missforest", "hyperimpute"}


class SynImputer:
    """
    Imputes missing values in a DataFrame using pattern-aware strategy
    selection.

    Usage:
        imputer = SynImputer()
        result = imputer.impute(df)
        print(result.missing_pct_after)   # 0.0
        print(result.data.isna().sum())   # all zeros
    """

    def impute(
        self,
        df: pd.DataFrame,
        missing_pattern: str = "auto",
        imputation_strategy: str = "auto",
        imputation_overrides: dict[str, str] | None = None,
        signal_cols: list[str] | None = None,
    ) -> ImputeResult:
        """
        Impute missing values in df.

        Args:
            df                    : input DataFrame (may contain NaNs)
            missing_pattern       : "auto" or explicit pattern for all cols
            imputation_strategy   : "auto" or explicit strategy for all cols
            imputation_overrides  : per-column strategy overrides
            signal_cols           : columns to impute (default: all numeric)

        Returns:
            ImputeResult with imputed data and full report.

        Raises:
            SynImpError if imputation fails or output still has NaNs.
        """
        overrides = imputation_overrides or {}
        df_out = df.copy()

        n_rows, n_cols = df.shape
        total_cells = n_rows * n_cols
        total_missing_before = int(df.isna().sum().sum())
        missing_pct_before = round(
            total_missing_before / total_cells * 100, 2
        ) if total_cells > 0 else 0.0

        # ── early exit: no missing values ──────────────────────────────────
        if total_missing_before == 0:
            return ImputeResult(
                data=df_out,
                total_missing_before=0,
                missing_pct_before=0.0,
                missing_pct_after=0.0,
                pattern_detected="none",
                strategy_used="none",
                columns=[],
            )

        # ── determine which columns to impute ───────────────────────────────
        cols_to_impute = signal_cols or [
            c for c in df.columns
            if df[c].isna().any()
        ]

        # ── detect patterns ────────────────────────────────────────────────
        if missing_pattern == "auto":
            detection = SynPatternDetector().detect(df)
            patterns = detection.as_dict()
        else:
            patterns = {col: missing_pattern for col in cols_to_impute}

        # ── determine strategy per column ──────────────────────────────────
        col_strategies: dict[str, str] = {}
        col_details: list[ColumnImputation] = []

        for col in cols_to_impute:
            missing_count = int(df[col].isna().sum())
            missing_pct = round(missing_count / n_rows * 100, 2)
            pattern = patterns.get(col, "MCAR")

            if col in overrides:
                strategy = overrides[col]
            elif imputation_strategy == "auto":
                strategy = auto_strategy(pattern, missing_pct)
            else:
                strategy = imputation_strategy

            if strategy == "none":
                continue

            col_strategies[col] = strategy
            col_details.append(ColumnImputation(
                column=col,
                missing_before=missing_count,
                missing_pct_before=missing_pct,
                pattern=pattern,
                strategy=strategy,
            ))

        # ── group by strategy type ─────────────────────────────────────────
        simple_cols: dict[str, str] = {
            col: strat
            for col, strat in col_strategies.items()
            if strat in _SIMPLE_STRATEGIES
        }
        ml_cols: dict[str, str] = {
            col: strat
            for col, strat in col_strategies.items()
            if strat in _ML_STRATEGIES
        }

        # ── apply simple strategies (per-column) ────────────────────────────
        for col, strategy in simple_cols.items():
            if strategy == "forward_fill":
                df_out[col] = apply_forward_fill(df_out[col])
            elif strategy == "spline":
                df_out[col] = apply_spline(df_out[col])

        # ── apply ML strategies (grouped by method) ────────────────────────
        if ml_cols:
            # group columns that share the same ML method
            method_groups: dict[str, list[str]] = {}
            for col, method in ml_cols.items():
                method_groups.setdefault(method, []).append(col)

            for method, method_cols in method_groups.items():
                # impute using all numeric cols as context,
                # but only update the target columns
                numeric_context = [
                    c for c in df_out.columns
                    if pd.api.types.is_numeric_dtype(df_out[c])
                ]
                if not numeric_context:
                    continue

                df_numeric = df_out[numeric_context].copy()
                df_imputed = apply_ml_imputer(df_numeric, method)

                # write back only the target columns
                for col in method_cols:
                    if col in df_imputed.columns:
                        df_out[col] = df_imputed[col].values

        # ── safety net: ffill/bfill any remaining NaNs ──────────────────────
        for col in cols_to_impute:
            if df_out[col].isna().any():
                df_out[col] = apply_forward_fill(df_out[col])

        # ── verify zero NaNs in imputed columns ────────────────────────────
        remaining = {
            col: int(df_out[col].isna().sum())
            for col in cols_to_impute
            if df_out[col].isna().any()
        }
        if remaining:
            raise SynImpError(
                "Imputation did not eliminate all NaNs.",
                detail=f"Remaining NaNs: {remaining}",
            )

        # ── build summary stats ────────────────────────────────────────────
        total_missing_after = int(df_out.isna().sum().sum())
        missing_pct_after = round(
            total_missing_after / total_cells * 100, 2
        ) if total_cells > 0 else 0.0

        # most common non-none pattern and strategy used
        non_none_patterns = [
            d.pattern for d in col_details if d.pattern != "none"
        ]
        strategies_used = [d.strategy for d in col_details]

        dominant_pattern = (
            max(set(non_none_patterns), key=non_none_patterns.count)
            if non_none_patterns else "none"
        )
        dominant_strategy = (
            max(set(strategies_used), key=strategies_used.count)
            if strategies_used else "none"
        )

        return ImputeResult(
            data=df_out,
            total_missing_before=total_missing_before,
            missing_pct_before=missing_pct_before,
            missing_pct_after=missing_pct_after,
            pattern_detected=dominant_pattern,
            strategy_used=dominant_strategy,
            columns=col_details,
        )
