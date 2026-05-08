"""
Stage 5 tests – SynImputer

Tests that SynImputer:
  - returns zero NaNs after imputation (all strategies)
  - preserves DataFrame shape
  - produces correct ImputeResult fields
  - applies forward_fill, spline, and ML strategies correctly
  - respects per-column overrides
  - handles clean data (no imputation needed)
  - as_dict() returns correct report structure
  - safety net handles edge cases

DO NOT MODIFY THIS FILE.
Fix the implementation, not the tests.
"""

import numpy as np
import pandas as pd
import pytest


# ── fixtures ──────────────────────────────────────────────────────────────

def make_df_with_mcar(n: int = 300, pct: float = 0.05) -> pd.DataFrame:
    """Small MCAR dataset – triggers forward_fill (< 5%)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "accel_x": np.random.randn(n),
        "accel_y": np.random.randn(n),
        "temperature": np.random.uniform(20, 80, n),
    })
    n_missing = max(1, int(n * pct))
    idx = rng.choice(n, size=n_missing, replace=False)
    df.loc[idx, "accel_x"] = np.nan
    return df


def make_df_with_gaps(n: int = 300) -> pd.DataFrame:
    """MCAR dataset with 10% missing – triggers spline."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "signal": np.sin(np.linspace(0, 4 * np.pi, n)),
        "noise": np.random.randn(n),
    })
    n_missing = int(n * 0.10)
    idx = rng.choice(n, size=n_missing, replace=False)
    df.loc[idx, "signal"] = np.nan
    return df


def make_clean_df(n: int = 300) -> pd.DataFrame:
    """DataFrame with no missing values."""
    return pd.DataFrame({
        "accel_x": np.random.randn(n),
        "accel_y": np.random.randn(n),
        "temperature": np.random.uniform(20, 80, n),
    })


def make_df_with_mar(n: int = 400) -> pd.DataFrame:
    """MAR dataset – missingness correlated with temperature."""
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "accel_x": np.random.randn(n),
        "temperature": np.random.uniform(20, 80, n),
    })
    high_temp = df["temperature"] > 60
    noise = rng.random(n) > 0.15
    df.loc[high_temp & noise, "accel_x"] = np.nan
    return df


# ── 1. Import ──────────────────────────────────────────────────────────────

class TestImport:
    def test_syn_imputer_importable(self):
        from synthflow.imputer import SynImputer
        assert SynImputer is not None

    def test_impute_result_importable(self):
        from synthflow.imputer import ImputeResult
        assert ImputeResult is not None

    def test_importable_from_engine(self):
        from synthflow.imputer.engine import SynImputer, ImputeResult
        assert SynImputer is not None
        assert ImputeResult is not None


# ── 2. Zero NaN guarantee ──────────────────────────────────────────────────

class TestZeroNaNGuarantee:
    def test_forward_fill_produces_zero_nans(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_mcar(pct=0.03)
        result = SynImputer().impute(df)
        assert result.data.isna().sum().sum() == 0

    def test_spline_produces_zero_nans(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_gaps()
        result = SynImputer().impute(df)
        assert result.data.isna().sum().sum() == 0

    def test_explicit_forward_fill_produces_zero_nans(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_mcar(pct=0.10)
        result = SynImputer().impute(df, imputation_strategy="forward_fill")
        assert result.data.isna().sum().sum() == 0

    def test_explicit_spline_produces_zero_nans(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_gaps()
        result = SynImputer().impute(df, imputation_strategy="spline")
        assert result.data.isna().sum().sum() == 0

    def test_knn_produces_zero_nans(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_mcar(pct=0.10)
        result = SynImputer().impute(df, imputation_strategy="knn")
        assert result.data.isna().sum().sum() == 0

    def test_mice_produces_zero_nans(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_mar()
        result = SynImputer().impute(df, imputation_strategy="mice")
        assert result.data.isna().sum().sum() == 0

    def test_missforest_produces_zero_nans(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_mcar(pct=0.10)
        result = SynImputer().impute(df, imputation_strategy="missforest")
        assert result.data.isna().sum().sum() == 0

    def test_hyperimpute_produces_zero_nans(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_mcar(pct=0.10)
        result = SynImputer().impute(df, imputation_strategy="hyperimpute")
        assert result.data.isna().sum().sum() == 0

    def test_missing_pct_after_is_zero(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_mcar(pct=0.10)
        result = SynImputer().impute(df)
        assert result.missing_pct_after == 0.0


# ── 3. Shape preservation ──────────────────────────────────────────────────

class TestShapePreservation:
    def test_shape_preserved_forward_fill(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_mcar(n=300, pct=0.05)
        result = SynImputer().impute(df)
        assert result.data.shape == df.shape

    def test_shape_preserved_spline(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_gaps()
        result = SynImputer().impute(df)
        assert result.data.shape == df.shape

    def test_shape_preserved_knn(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_mcar(pct=0.10)
        result = SynImputer().impute(df, imputation_strategy="knn")
        assert result.data.shape == df.shape

    def test_columns_unchanged(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_mcar()
        result = SynImputer().impute(df)
        assert list(result.data.columns) == list(df.columns)

    def test_index_preserved(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_mcar()
        result = SynImputer().impute(df)
        assert list(result.data.index) == list(df.index)


# ── 4. ImputeResult fields ────────────────────────────────────────────────

class TestImputeResultFields:
    def test_total_missing_before_correct(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_mcar(n=300, pct=0.05)
        expected = int(df.isna().sum().sum())
        result = SynImputer().impute(df)
        assert result.total_missing_before == expected

    def test_missing_pct_before_positive(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_mcar(pct=0.10)
        result = SynImputer().impute(df)
        assert result.missing_pct_before > 0.0

    def test_pattern_detected_is_string(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_mcar()
        result = SynImputer().impute(df)
        assert isinstance(result.pattern_detected, str)

    def test_strategy_used_is_string(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_mcar()
        result = SynImputer().impute(df)
        assert isinstance(result.strategy_used, str)

    def test_columns_list_not_empty_for_dirty_data(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_mcar(pct=0.10)
        result = SynImputer().impute(df)
        assert len(result.columns) > 0

    def test_column_detail_has_correct_fields(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_mcar(pct=0.10)
        result = SynImputer().impute(df)
        affected = [c for c in result.columns if c.missing_before > 0]
        assert len(affected) > 0
        col = affected[0]
        assert isinstance(col.column, str)
        assert isinstance(col.missing_before, int)
        assert isinstance(col.missing_pct_before, float)
        assert isinstance(col.pattern, str)
        assert isinstance(col.strategy, str)

    def test_data_is_dataframe(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_mcar()
        result = SynImputer().impute(df)
        assert isinstance(result.data, pd.DataFrame)


# ── 5. Clean data ──────────────────────────────────────────────────────────

class TestCleanData:
    def test_clean_df_returns_zero_missing_before(self):
        from synthflow.imputer import SynImputer
        df = make_clean_df()
        result = SynImputer().impute(df)
        assert result.total_missing_before == 0

    def test_clean_df_pattern_is_none(self):
        from synthflow.imputer import SynImputer
        df = make_clean_df()
        result = SynImputer().impute(df)
        assert result.pattern_detected == "none"

    def test_clean_df_strategy_is_none(self):
        from synthflow.imputer import SynImputer
        df = make_clean_df()
        result = SynImputer().impute(df)
        assert result.strategy_used == "none"

    def test_clean_df_columns_list_empty(self):
        from synthflow.imputer import SynImputer
        df = make_clean_df()
        result = SynImputer().impute(df)
        assert result.columns == []

    def test_clean_df_data_unchanged(self):
        from synthflow.imputer import SynImputer
        df = make_clean_df()
        result = SynImputer().impute(df)
        pd.testing.assert_frame_equal(result.data, df)


# ── 6. Per-column overrides ────────────────────────────────────────────────

class TestPerColumnOverrides:
    def test_override_applies_specified_strategy(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_mcar(pct=0.10)
        result = SynImputer().impute(
            df,
            imputation_overrides={"accel_x": "forward_fill"}
        )
        # accel_x should use forward_fill
        accel_detail = next(
            (c for c in result.columns if c.column == "accel_x"), None
        )
        if accel_detail and accel_detail.missing_before > 0:
            assert accel_detail.strategy == "forward_fill"

    def test_override_does_not_affect_other_cols(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_mcar(pct=0.10)
        result = SynImputer().impute(
            df,
            imputation_overrides={"accel_x": "forward_fill"},
            imputation_strategy="spline",
        )
        other_details = [
            c for c in result.columns
            if c.column != "accel_x" and c.missing_before > 0
        ]
        for detail in other_details:
            assert detail.strategy == "spline"

    def test_result_still_zero_nans_with_overrides(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_mcar(pct=0.10)
        result = SynImputer().impute(
            df,
            imputation_overrides={"accel_x": "spline"}
        )
        assert result.data.isna().sum().sum() == 0


# ── 7. as_dict() report ────────────────────────────────────────────────────

class TestAsDict:
    def test_as_dict_returns_dict(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_mcar(pct=0.10)
        result = SynImputer().impute(df)
        assert isinstance(result.as_dict(), dict)

    def test_as_dict_has_required_keys(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_mcar(pct=0.10)
        d = SynImputer().impute(df).as_dict()
        assert "total_missing_before" in d
        assert "missing_pct_before" in d
        assert "missing_pct_after" in d
        assert "pattern_detected" in d
        assert "strategy_used" in d
        assert "columns_affected" in d

    def test_as_dict_missing_pct_after_zero(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_mcar(pct=0.10)
        d = SynImputer().impute(df).as_dict()
        assert d["missing_pct_after"] == 0.0

    def test_as_dict_columns_affected_not_empty(self):
        from synthflow.imputer import SynImputer
        df = make_df_with_mcar(pct=0.10)
        d = SynImputer().impute(df).as_dict()
        assert len(d["columns_affected"]) > 0

    def test_as_dict_clean_data_columns_affected_empty(self):
        from synthflow.imputer import SynImputer
        df = make_clean_df()
        d = SynImputer().impute(df).as_dict()
        assert d["columns_affected"] == {}


# ── 8. Signal cols filtering ──────────────────────────────────────────────

class TestSignalColsFiltering:
    def test_signal_cols_limits_imputation(self):
        from synthflow.imputer import SynImputer
        rng = np.random.default_rng(5)
        df = pd.DataFrame({
            "accel_x": np.random.randn(200),
            "accel_y": np.random.randn(200),
        })
        idx = rng.choice(200, size=20, replace=False)
        df.loc[idx, "accel_x"] = np.nan
        df.loc[idx[:10], "accel_y"] = np.nan

        # only impute accel_x
        result = SynImputer().impute(
            df,
            signal_cols=["accel_x"],
            imputation_strategy="forward_fill",
        )
        assert result.data["accel_x"].isna().sum() == 0
