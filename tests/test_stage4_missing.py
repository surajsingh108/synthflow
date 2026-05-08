"""
Stage 4 tests – Missing Pattern Detector

Tests that SynPatternDetector:
  - correctly identifies columns with no missing values
  - detects MCAR for random missingness
  - detects MAR when missingness correlates with another column
  - detects MNAR for tail-concentrated missingness
  - handles trivial missing (< 1%) as MCAR without running tests
  - returns correct metadata in ColumnResult
  - DetectionResult summary and as_dict work correctly
  - thresholds are configurable

DO NOT MODIFY THIS FILE.
Fix the implementation, not the tests.
"""

import numpy as np
import pandas as pd
import pytest


# – helpers –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

def make_clean_df(n: int = 500) -> pd.DataFrame:
    """DataFrame with no missing values."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="10ms"),
        "accel_x": np.random.randn(n),
        "accel_y": np.random.randn(n),
        "temperature": np.random.uniform(20, 80, n),
    })


def make_mcar_df(n: int = 500, missing_pct: float = 0.10) -> pd.DataFrame:
    """DataFrame with randomly placed missing values (MCAR)."""
    rng = np.random.default_rng(42)
    df = make_clean_df(n)
    n_missing = int(n * missing_pct)
    # random indices – no pattern
    idx = rng.choice(n, size=n_missing, replace=False)
    df.loc[idx, "accel_x"] = np.nan
    return df


def make_mar_df(n: int = 500) -> pd.DataFrame:
    """
    DataFrame where accel_x is missing when temperature is high.
    This creates a clear MAR dependency.
    """
    rng = np.random.default_rng(42)
    df = make_clean_df(n)
    # accel_x is missing when temperature > 60 (strong correlation)
    high_temp = df["temperature"] > 60
    # add some noise to avoid perfect correlation
    noise = rng.random(n) > 0.15
    df.loc[high_temp & noise, "accel_x"] = np.nan
    return df


def make_mnar_df(n: int = 500) -> pd.DataFrame:
    """
    DataFrame where accel_x values go missing at the tail
    (simulating sensor failure at end of recording).
    """
    df = make_clean_df(n)
    # last 40% of rows have accel_x missing (tail-concentrated)
    tail_start = int(n * 0.70)
    df.loc[tail_start:, "accel_x"] = np.nan
    return df


def make_trivial_missing_df(n: int = 500) -> pd.DataFrame:
    """DataFrame with < 1% missing – trivial MCAR."""
    df = make_clean_df(n)
    df.loc[0, "accel_x"] = np.nan  # only 1 missing out of 500 = 0.2%
    return df


# – 1. Import ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

class TestImport:
    def test_syn_pattern_detector_importable(self):
        from synthflow.imputer import SynPatternDetector
        assert SynPatternDetector is not None

    def test_importable_from_module(self):
        from synthflow.imputer.missing_pattern import SynPatternDetector
        assert SynPatternDetector is not None

    def test_column_result_importable(self):
        from synthflow.imputer.missing_pattern import ColumnResult
        assert ColumnResult is not None

    def test_detection_result_importable(self):
        from synthflow.imputer.missing_pattern import DetectionResult
        assert DetectionResult is not None


# – 2. No missing values ––––––––––––––––––––––––––––––––––––––––––––––––––––

class TestNoMissing:
    def test_clean_column_returns_none(self):
        from synthflow.imputer import SynPatternDetector
        df = make_clean_df()
        result = SynPatternDetector().detect(df)
        assert result.pattern_for("accel_x") == "none"

    def test_all_clean_columns_return_none(self):
        from synthflow.imputer import SynPatternDetector
        df = make_clean_df()
        result = SynPatternDetector().detect(df)
        for col in ["accel_x", "accel_y", "temperature"]:
            assert result.pattern_for(col) == "none"

    def test_none_pattern_has_zero_missing_count(self):
        from synthflow.imputer import SynPatternDetector
        df = make_clean_df()
        result = SynPatternDetector().detect(df)
        cr = result.columns["accel_x"]
        assert cr.missing_count == 0
        assert cr.missing_pct == 0.0


# – 3. MCAR detection –––––––––––––––––––––––––––––––––––––––––––––––––––––––

class TestMCAR:
    def test_random_missing_detected_as_mcar(self):
        from synthflow.imputer import SynPatternDetector
        df = make_mcar_df()
        result = SynPatternDetector().detect(df)
        assert result.pattern_for("accel_x") == "MCAR"

    def test_mcar_has_correct_missing_count(self):
        from synthflow.imputer import SynPatternDetector
        df = make_mcar_df(n=500, missing_pct=0.10)
        result = SynPatternDetector().detect(df)
        cr = result.columns["accel_x"]
        # allow ±5 for randomness
        assert 40 <= cr.missing_count <= 60

    def test_mcar_missing_pct_reasonable(self):
        from synthflow.imputer import SynPatternDetector
        df = make_mcar_df(n=500, missing_pct=0.10)
        result = SynPatternDetector().detect(df)
        cr = result.columns["accel_x"]
        assert 8.0 <= cr.missing_pct <= 12.0

    def test_trivial_missing_classified_as_mcar(self):
        from synthflow.imputer import SynPatternDetector
        df = make_trivial_missing_df()
        result = SynPatternDetector().detect(df)
        assert result.pattern_for("accel_x") == "MCAR"

    def test_trivial_detection_path_mentions_trivial(self):
        from synthflow.imputer import SynPatternDetector
        df = make_trivial_missing_df()
        result = SynPatternDetector().detect(df)
        cr = result.columns["accel_x"]
        assert "trivial" in cr.detection_path.lower()


# – 4. MAR detection ––––––––––––––––––––––––––––––––––––––––––––––––––––––––

class TestMAR:
    def test_correlated_missing_detected_as_mar(self):
        from synthflow.imputer import SynPatternDetector
        df = make_mar_df()
        result = SynPatternDetector().detect(df)
        assert result.pattern_for("accel_x") == "MAR"

    def test_mar_result_has_max_corr(self):
        from synthflow.imputer import SynPatternDetector
        df = make_mar_df()
        result = SynPatternDetector().detect(df)
        cr = result.columns["accel_x"]
        assert cr.mar_max_corr is not None
        assert cr.mar_max_corr > 0.0

    def test_mar_max_corr_above_threshold(self):
        from synthflow.imputer import SynPatternDetector
        df = make_mar_df()
        detector = SynPatternDetector()
        result = detector.detect(df)
        cr = result.columns["accel_x"]
        assert cr.mar_max_corr >= detector.mar_threshold

    def test_non_missing_columns_not_affected_by_mar_test(self):
        from synthflow.imputer import SynPatternDetector
        df = make_mar_df()
        result = SynPatternDetector().detect(df)
        # accel_y and temperature are clean
        assert result.pattern_for("accel_y") == "none"
        assert result.pattern_for("temperature") == "none"


# – 5. MNAR detection –––––––––––––––––––––––––––––––––––––––––––––––––––––––

class TestMNAR:
    def test_tail_missing_detected_as_mnar(self):
        from synthflow.imputer import SynPatternDetector
        df = make_mnar_df()
        result = SynPatternDetector().detect(df)
        assert result.pattern_for("accel_x") == "MNAR"

    def test_mnar_result_has_tail_frac(self):
        from synthflow.imputer import SynPatternDetector
        df = make_mnar_df()
        result = SynPatternDetector().detect(df)
        cr = result.columns["accel_x"]
        assert cr.mnar_tail_frac is not None

    def test_mnar_tail_frac_high(self):
        from synthflow.imputer import SynPatternDetector
        df = make_mnar_df()
        detector = SynPatternDetector()
        result = detector.detect(df)
        cr = result.columns["accel_x"]
        assert cr.mnar_tail_frac >= detector.mnar_tail_concentration

    def test_head_concentrated_missing_detected_as_mnar(self):
        """Missing values at the start (not end) also triggers MNAR."""
        from synthflow.imputer import SynPatternDetector
        df = make_clean_df(n=500)
        # first 40% missing
        df.loc[:149, "accel_x"] = np.nan
        result = SynPatternDetector().detect(df)
        assert result.pattern_for("accel_x") == "MNAR"


# – 6. DetectionResult ––––––––––––––––––––––––––––––––––––––––––––––––––––––

class TestDetectionResult:
    def test_summary_has_all_pattern_keys(self):
        from synthflow.imputer import SynPatternDetector
        df = make_clean_df()
        result = SynPatternDetector().detect(df)
        summary = result.summary
        assert "none" in summary
        assert "MCAR" in summary
        assert "MAR" in summary
        assert "MNAR" in summary

    def test_summary_groups_columns_correctly(self):
        from synthflow.imputer import SynPatternDetector
        df = make_mcar_df()
        result = SynPatternDetector().detect(df)
        summary = result.summary
        assert "accel_x" in summary["MCAR"]
        assert "accel_y" in summary["none"]

    def test_as_dict_returns_pattern_per_column(self):
        from synthflow.imputer import SynPatternDetector
        df = make_mcar_df()
        result = SynPatternDetector().detect(df)
        d = result.as_dict()
        assert isinstance(d, dict)
        assert "accel_x" in d
        assert d["accel_x"] == "MCAR"

    def test_as_dict_covers_all_columns(self):
        from synthflow.imputer import SynPatternDetector
        df = make_clean_df()
        result = SynPatternDetector().detect(df)
        d = result.as_dict()
        for col in df.columns:
            assert col in d

    def test_pattern_for_unknown_column_raises(self):
        from synthflow.imputer import SynPatternDetector
        df = make_clean_df()
        result = SynPatternDetector().detect(df)
        with pytest.raises(KeyError):
            result.pattern_for("nonexistent_column")


# – 7. Single column detection –––––––––––––––––––––––––––––––––––––––––––––

class TestSingleColumn:
    def test_detect_column_returns_column_result(self):
        from synthflow.imputer import SynPatternDetector
        from synthflow.imputer.missing_pattern import ColumnResult
        df = make_mcar_df()
        result = SynPatternDetector().detect_column(df, "accel_x")
        assert isinstance(result, ColumnResult)

    def test_detect_column_matches_detect_all(self):
        from synthflow.imputer import SynPatternDetector
        df = make_mcar_df()
        detector = SynPatternDetector()
        full = detector.detect(df)
        single = detector.detect_column(df, "accel_x")
        assert single.pattern == full.pattern_for("accel_x")


# – 8. Configurable thresholds –––––––––––––––––––––––––––––––––––––––––––––

class TestConfigurableThresholds:
    def test_higher_mar_threshold_changes_classification(self):
        """With a very high MAR threshold, MAR data gets classified as MCAR."""
        from synthflow.imputer import SynPatternDetector
        df = make_mar_df()
        # set threshold so high that MAR test never triggers
        detector = SynPatternDetector(mar_threshold=0.999)
        result = detector.detect(df)
        # should fall through to MCAR since threshold is unreachable
        assert result.pattern_for("accel_x") in ("MCAR", "MNAR")

    def test_lower_trivial_threshold_runs_full_detection(self):
        """With trivial_pct=0, even tiny missing runs full detection."""
        from synthflow.imputer import SynPatternDetector
        df = make_trivial_missing_df()
        detector = SynPatternDetector(trivial_pct=0.0)
        result = detector.detect(df)
        cr = result.columns["accel_x"]
        # should have run full detection path
        assert "trivial" not in cr.detection_path.lower()

    def test_lower_mnar_concentration_triggers_mnar_earlier(self):
        """With low MNAR concentration threshold, mild tails – MNAR."""
        from synthflow.imputer import SynPatternDetector
        df = make_clean_df(n=500)
        # only last 15% is missing (mild tail)
        df.loc[425:, "accel_x"] = np.nan
        # default threshold (0.60) might not catch this
        # aggressive threshold (0.30) should
        detector = SynPatternDetector(mnar_tail_concentration=0.30)
        result = detector.detect(df)
        assert result.pattern_for("accel_x") == "MNAR"


# – 9. Edge cases ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

class TestEdgeCases:
    def test_all_missing_column(self):
        from synthflow.imputer import SynPatternDetector
        df = pd.DataFrame({
            "accel_x": [np.nan] * 200,
            "accel_y": np.random.randn(200),
        })
        # should not raise – return some pattern
        result = SynPatternDetector().detect(df)
        assert result.pattern_for("accel_x") in ("MCAR", "MAR", "MNAR")

    def test_single_column_dataframe(self):
        from synthflow.imputer import SynPatternDetector
        df = pd.DataFrame({"value": [1.0, np.nan, 3.0] * 100})
        # no other columns for MAR test – should not raise
        result = SynPatternDetector().detect(df)
        assert result.pattern_for("value") in ("MCAR", "MNAR")

    def test_non_numeric_columns_detected_as_none_or_mcar(self):
        from synthflow.imputer import SynPatternDetector
        df = pd.DataFrame({
            "label": ["a", "b", np.nan] * 50,
            "value": np.random.randn(150),
        })
        # should not raise on non-numeric columns
        result = SynPatternDetector().detect(df)
        assert "label" in result.columns

    def test_detect_all_columns_present_in_result(self):
        from synthflow.imputer import SynPatternDetector
        df = make_mcar_df()
        result = SynPatternDetector().detect(df)
        for col in df.columns:
            assert col in result.columns
