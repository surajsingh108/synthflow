"""
Strategy routing – maps missing pattern + config to imputation method,
and dispatches the correct imputer for each method.

Simple methods  (forward_fill, spline) : pure pandas/scipy, per-column.
ML methods      (knn, mice, missforest,
                 hyperimpute)           : HyperImpute, applied to all
                                         numeric columns together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from synthflow.exceptions import SynImpError


# ── Pattern → strategy mapping (used when strategy == "auto") ──────────────

def auto_strategy(pattern: str, missing_pct: float) -> str:
    """
    Choose imputation strategy based on detected pattern and % missing.

    Rules:
      none            → skip (no imputation needed)
      MCAR < 5%       → forward_fill  (short random gaps, fast)
      MCAR 5-30%      → spline        (longer gaps, preserves signal shape)
      MCAR > 30%      → hyperimpute   (too much missing for interpolation)
      MAR             → mice          (missingness depends on other cols)
      MNAR            → missforest    (non-linear relationships)
      fallback        → hyperimpute
    """
    if pattern == "none":
        return "none"
    if pattern == "MCAR":
        if missing_pct < 5.0:
            return "forward_fill"
        elif missing_pct <= 30.0:
            return "spline"
        else:
            return "hyperimpute"
    if pattern == "MAR":
        return "mice"
    if pattern == "MNAR":
        return "missforest"
    return "hyperimpute"


# ── Simple imputers (per-column, no external deps beyond pandas/scipy) ────

def apply_forward_fill(series: pd.Series) -> pd.Series:
    """
    Forward fill then backward fill.
    Handles leading NaNs (bfill) and trailing NaNs (ffill already covered).
    """
    return series.ffill().bfill()


def apply_spline(series: pd.Series) -> pd.Series:
    """
    Cubic spline interpolation with forward/backward fill for edge NaNs.
    Falls back to forward_fill if spline fails (e.g. too few valid points).
    """
    try:
        interpolated = series.interpolate(
            method="spline",
            order=3,
            limit_direction="both",
        )
        # fill any remaining NaNs at boundaries
        return interpolated.ffill().bfill()
    except Exception:
        return apply_forward_fill(series)


# ── ML imputers (whole-DataFrame, via HyperImpute) ────────────────────────

_HYPERIMPUTE_METHOD_MAP = {
    "knn":         "ice",          # nearest-neighbours
    "mice":        "mice",         # iterative conditional expectations (MICE)
    "missforest":  "missforest",   # random forest imputation
    "hyperimpute": "hyperimpute",  # auto-select best method
}


def apply_ml_imputer(
    df_numeric: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    """
    Apply a HyperImpute ML method to a numeric-only DataFrame.

    Args:
        df_numeric : DataFrame with numeric columns only (may contain NaNs)
        method     : one of "knn", "mice", "missforest", "hyperimpute"

    Returns:
        DataFrame with same shape and columns, zero NaNs.

    Raises:
        SynImpError if HyperImpute fails or output still contains NaNs.
    """
    try:
        from hyperimpute.plugins.imputers import Imputers
    except ImportError as e:
        raise SynImpError(
            "HyperImpute is not installed.",
            detail="Run: pip install hyperimpute",
        ) from e

    hi_method = _HYPERIMPUTE_METHOD_MAP.get(method, "hyperimpute")

    try:
        plugin = Imputers().get(hi_method)
        result = plugin.fit_transform(df_numeric.copy())
    except Exception as exc:
        raise SynImpError(
            f"HyperImpute method '{hi_method}' failed.",
            detail=str(exc),
        ) from exc

    # result may be numpy array or DataFrame
    if isinstance(result, np.ndarray):
        result = pd.DataFrame(result, columns=df_numeric.columns,
                              index=df_numeric.index)
    elif not isinstance(result, pd.DataFrame):
        result = pd.DataFrame(result, columns=df_numeric.columns,
                              index=df_numeric.index)

    if result.isna().any().any():
        raise SynImpError(
            f"HyperImpute method '{hi_method}' left NaNs in output.",
            detail="Try a different imputation strategy.",
        )

    return result
