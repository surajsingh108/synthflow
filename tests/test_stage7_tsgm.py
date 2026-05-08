"""
Stage 7 tests – TSGM Backend

Tests that:
  - SynBackend abstract interface is correctly defined
  - Data preparation utilities (window/dewindow/build_output) work correctly
  - TsgmBackend.run() produces correct output format
  - GaussianProcess model: fits and generates (fast, CPU)
  - AR model: fits and generates (fast, CPU)
  - Output has synthetic_ prefix on signal columns
  - Timestamp column is preserved unchanged
  - Output shape matches n_samples
  - Reproducibility: same seed → same output (GP, AR)
  - SynBackendError raised for bad inputs

DO NOT MODIFY THIS FILE.
Fix the implementation, not the tests.
"""

import numpy as np
import pandas as pd
import pytest


# ──── fixtures ────────────────────────────────────────────────────────────────

def make_sensor_df(n: int = 200, freq: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="10ms"),
        "accel_x":   np.sin(np.linspace(0, 4 * np.pi, n)) + 0.1 * np.random.randn(n),
        "accel_y":   np.cos(np.linspace(0, 4 * np.pi, n)) + 0.1 * np.random.randn(n),
        "temperature": np.random.uniform(20, 80, n),
    })


# ──── 1. Import ────────────────────────────────────────────────────────────────

class TestImport:
    def test_syn_backend_importable(self):
        from synthflow.backends import SynBackend
        assert SynBackend is not None

    def test_tsgm_backend_importable(self):
        from synthflow.backends import TsgmBackend
        assert TsgmBackend is not None

    def test_base_importable_from_module(self):
        from synthflow.backends.base import SynBackend
        assert SynBackend is not None

    def test_utilities_importable(self):
        from synthflow.backends.tsgm_backend import (
            window_dataframe, dewindow_array, build_output_df, infer_seq_len
        )
        assert window_dataframe is not None


# ──── 2. Abstract interface ────────────────────────────────────────────────────

class TestAbstractInterface:
    def test_cannot_instantiate_syn_backend(self):
        from synthflow.backends.base import SynBackend
        with pytest.raises(TypeError):
            SynBackend()

    def test_concrete_must_implement_fit(self):
        from synthflow.backends.base import SynBackend
        import inspect
        assert "fit" in [m for m in dir(SynBackend)
                         if not m.startswith("_")]

    def test_concrete_must_implement_generate(self):
        from synthflow.backends.base import SynBackend
        assert "generate" in [m for m in dir(SynBackend)
                               if not m.startswith("_")]


# ──── 3. window_dataframe ──────────────────────────────────────────────────────

class TestWindowDataframe:
    def test_output_shape_correct(self):
        from synthflow.backends.tsgm_backend import window_dataframe
        df = make_sensor_df(n=200)
        X = window_dataframe(df, ["accel_x", "accel_y"], seq_len=24, step=12)
        assert X.ndim == 3
        assert X.shape[1] == 24   # seq_len
        assert X.shape[2] == 2   # n_features

    def test_n_windows_reasonable(self):
        from synthflow.backends.tsgm_backend import window_dataframe
        df = make_sensor_df(n=200)
        X = window_dataframe(df, ["accel_x"], seq_len=24, step=12)
        # (200 - 24) / 12 + 1 → 15 windows
        assert X.shape[0] >= 10

    def test_default_step_is_half_seq_len(self):
        from synthflow.backends.tsgm_backend import window_dataframe
        df = make_sensor_df(n=200)
        X_half = window_dataframe(df, ["accel_x"], seq_len=24)
        X_explicit = window_dataframe(df, ["accel_x"], seq_len=24, step=12)
        assert X_half.shape == X_explicit.shape

    def test_dtype_is_float32(self):
        from synthflow.backends.tsgm_backend import window_dataframe
        df = make_sensor_df(n=200)
        X = window_dataframe(df, ["accel_x"], seq_len=24)
        assert X.dtype == np.float32

    def test_too_short_raises_syn_backend_error(self):
        from synthflow.backends.tsgm_backend import window_dataframe
        from synthflow.exceptions import SynBackendError
        df = make_sensor_df(n=10)
        with pytest.raises(SynBackendError):
            window_dataframe(df, ["accel_x"], seq_len=50)


# ──── 4. dewindow_array ────────────────────────────────────────────────────────

class TestDewindowArray:
    def test_output_shape_correct(self):
        from synthflow.backends.tsgm_backend import dewindow_array
        arr = np.random.randn(10, 24, 3).astype(np.float32)
        flat = dewindow_array(arr)
        assert flat.ndim == 2
        assert flat.shape == (240, 3)  # 10 * 24, 3

    def test_n_target_rows_trims(self):
        from synthflow.backends.tsgm_backend import dewindow_array
        arr = np.random.randn(10, 24, 3).astype(np.float32)
        flat = dewindow_array(arr, n_target_rows=100)
        assert flat.shape == (100, 3)

    def test_n_target_rows_pads(self):
        from synthflow.backends.tsgm_backend import dewindow_array
        arr = np.random.randn(2, 24, 3).astype(np.float32)
        flat = dewindow_array(arr, n_target_rows=200)
        assert flat.shape == (200, 3)


# ──── 5. build_output_df ───────────────────────────────────────────────────────

class TestBuildOutputDf:
    def test_signal_cols_have_synthetic_prefix(self):
        from synthflow.backends.tsgm_backend import build_output_df
        df = make_sensor_df(n=200)
        arr = np.random.randn(10, 24, 2).astype(np.float32)
        out = build_output_df(
            arr=arr,
            signal_cols=["accel_x", "accel_y"],
            timestamp_col="timestamp",
            source_df=df,
            n_rows=100,
        )
        assert "synthetic_accel_x" in out.columns
        assert "synthetic_accel_y" in out.columns

    def test_original_col_names_not_present(self):
        from synthflow.backends.tsgm_backend import build_output_df
        df = make_sensor_df(n=200)
        arr = np.random.randn(10, 24, 2).astype(np.float32)
        out = build_output_df(
            arr=arr,
            signal_cols=["accel_x", "accel_y"],
            timestamp_col="timestamp",
            source_df=df,
            n_rows=100,
        )
        assert "accel_x" not in out.columns
        assert "accel_y" not in out.columns

    def test_timestamp_col_preserved_without_prefix(self):
        from synthflow.backends.tsgm_backend import build_output_df
        df = make_sensor_df(n=200)
        arr = np.random.randn(5, 24, 2).astype(np.float32)
        out = build_output_df(
            arr=arr,
            signal_cols=["accel_x", "accel_y"],
            timestamp_col="timestamp",
            source_df=df,
            n_rows=50,
        )
        assert "timestamp" in out.columns
        assert "synthetic_timestamp" not in out.columns

    def test_output_row_count_matches_n_rows(self):
        from synthflow.backends.tsgm_backend import build_output_df
        df = make_sensor_df(n=200)
        arr = np.random.randn(10, 24, 2).astype(np.float32)
        out = build_output_df(
            arr=arr,
            signal_cols=["accel_x", "accel_y"],
            timestamp_col=None,
            source_df=df,
            n_rows=150,
        )
        assert len(out) == 150

    def test_no_timestamp_col_not_in_output(self):
        from synthflow.backends.tsgm_backend import build_output_df
        df = make_sensor_df(n=200)
        arr = np.random.randn(5, 24, 2).astype(np.float32)
        out = build_output_df(
            arr=arr,
            signal_cols=["accel_x", "accel_y"],
            timestamp_col=None,
            source_df=df,
            n_rows=50,
        )
        assert "timestamp" not in out.columns


# ──── 6. infer_seq_len ────────────────────────────────────────────────────────

class TestInferSeqLen:
    def test_seq_len_at_least_8(self):
        from synthflow.backends.tsgm_backend import infer_seq_len
        assert infer_seq_len(50, 1.0) >= 8

    def test_seq_len_at_most_128(self):
        from synthflow.backends.tsgm_backend import infer_seq_len
        assert infer_seq_len(10_000, 44_100) <= 128

    def test_seq_len_reasonable_for_100hz(self):
        from synthflow.backends.tsgm_backend import infer_seq_len
        seq = infer_seq_len(1000, 100.0)
        assert 8 <= seq <= 128

    def test_ensures_enough_windows(self):
        from synthflow.backends.tsgm_backend import infer_seq_len
        n_rows = 100
        seq = infer_seq_len(n_rows, 100.0)
        n_windows = n_rows // (seq // 2)
        assert n_windows >= 10 or seq == 8  # 8 is minimum floor


# ──── 7. GaussianProcess model ────────────────────────────────────────────────

class TestGaussianProcess:
    def test_gp_fits_without_error(self):
        from synthflow.backends.tsgm_backend import _GaussianProcessWrapper, window_dataframe
        df = make_sensor_df(n=200)
        X = window_dataframe(df, ["accel_x", "accel_y"], seq_len=24)
        model = _GaussianProcessWrapper(seq_len=24, n_features=2)
        model.fit(X)  # should not raise

    def test_gp_is_fitted_after_fit(self):
        from synthflow.backends.tsgm_backend import _GaussianProcessWrapper, window_dataframe
        df = make_sensor_df(n=200)
        X = window_dataframe(df, ["accel_x", "accel_y"], seq_len=24)
        model = _GaussianProcessWrapper(seq_len=24, n_features=2)
        assert not model.is_fitted
        model.fit(X)
        assert model.is_fitted

    def test_gp_generate_correct_shape(self):
        from synthflow.backends.tsgm_backend import _GaussianProcessWrapper, window_dataframe
        df = make_sensor_df(n=200)
        X = window_dataframe(df, ["accel_x", "accel_y"], seq_len=24)
        model = _GaussianProcessWrapper(seq_len=24, n_features=2)
        model.fit(X)
        out = model.generate(n=10)
        assert out.shape == (10, 24, 2)

    def test_gp_generate_before_fit_raises(self):
        from synthflow.backends.tsgm_backend import _GaussianProcessWrapper
        from synthflow.exceptions import SynBackendError
        model = _GaussianProcessWrapper(seq_len=24, n_features=2)
        with pytest.raises(SynBackendError):
            model.generate(n=5)

    def test_gp_reproducible_with_same_seed(self):
        from synthflow.backends.tsgm_backend import _GaussianProcessWrapper, window_dataframe
        df = make_sensor_df(n=200)
        X = window_dataframe(df, ["accel_x", "accel_y"], seq_len=24)
        m1 = _GaussianProcessWrapper(seq_len=24, n_features=2, random_seed=42)
        m2 = _GaussianProcessWrapper(seq_len=24, n_features=2, random_seed=42)
        m1.fit(X)
        m2.fit(X)
        np.testing.assert_array_equal(m1.generate(5), m2.generate(5))

    def test_gp_name(self):
        from synthflow.backends.tsgm_backend import _GaussianProcessWrapper
        assert _GaussianProcessWrapper(24, 2).name == "GaussianProcess"


# ──── 8. AR model ──────────────────────────────────────────────────────────────

class TestARModel:
    def test_ar_fits_without_error(self):
        from synthflow.backends.tsgm_backend import _ARWrapper, window_dataframe
        df = make_sensor_df(n=200)
        X = window_dataframe(df, ["accel_x", "accel_y"], seq_len=24)
        model = _ARWrapper(seq_len=24, n_features=2)
        model.fit(X)

    def test_ar_generate_correct_shape(self):
        from synthflow.backends.tsgm_backend import _ARWrapper, window_dataframe
        df = make_sensor_df(n=200)
        X = window_dataframe(df, ["accel_x", "accel_y"], seq_len=24)
        model = _ARWrapper(seq_len=24, n_features=2)
        model.fit(X)
        out = model.generate(n=10)
        assert out.shape == (10, 24, 2)

    def test_ar_reproducible_with_same_seed(self):
        from synthflow.backends.tsgm_backend import _ARWrapper, window_dataframe
        df = make_sensor_df(n=200)
        X = window_dataframe(df, ["accel_x", "accel_y"], seq_len=24)
        m1 = _ARWrapper(seq_len=24, n_features=2, random_seed=0)
        m2 = _ARWrapper(seq_len=24, n_features=2, random_seed=0)
        m1.fit(X); m2.fit(X)
        np.testing.assert_array_equal(m1.generate(5), m2.generate(5))

    def test_ar_name(self):
        from synthflow.backends.tsgm_backend import _ARWrapper
        assert _ARWrapper(24, 2).name == "AR"


# ──── 9. TsgmBackend.run() – fast models only ──────────────────────────────────

class TestTsgmBackendRun:
    def test_run_gaussian_process_output_format(self):
        from synthflow.backends import TsgmBackend
        df = make_sensor_df(n=200)
        out = TsgmBackend().run(
            df=df,
            signal_cols=["accel_x", "accel_y"],
            timestamp_col="timestamp",
            model_name="GaussianProcess",
            n_samples=100,
            epochs=1,
        )
        assert "synthetic_accel_x" in out.columns
        assert "synthetic_accel_y" in out.columns
        assert "timestamp" in out.columns
        assert len(out) == 100

    def test_run_ar_output_format(self):
        from synthflow.backends import TsgmBackend
        df = make_sensor_df(n=200)
        out = TsgmBackend().run(
            df=df,
            signal_cols=["accel_x", "temperature"],
            timestamp_col="timestamp",
            model_name="AR",
            n_samples=50,
            epochs=1,
        )
        assert "synthetic_accel_x" in out.columns
        assert "synthetic_temperature" in out.columns
        assert len(out) == 50

    def test_run_no_timestamp(self):
        from synthflow.backends import TsgmBackend
        df = make_sensor_df(n=200)
        out = TsgmBackend().run(
            df=df,
            signal_cols=["accel_x"],
            timestamp_col=None,
            model_name="GaussianProcess",
            n_samples=80,
        )
        assert "timestamp" not in out.columns
        assert "synthetic_accel_x" in out.columns
        assert len(out) == 80

    def test_run_no_nans_in_output(self):
        from synthflow.backends import TsgmBackend
        df = make_sensor_df(n=200)
        out = TsgmBackend().run(
            df=df,
            signal_cols=["accel_x", "accel_y"],
            timestamp_col="timestamp",
            model_name="GaussianProcess",
            n_samples=100,
        )
        assert out.isna().sum().sum() == 0

    def test_run_unknown_model_raises(self):
        from synthflow.backends import TsgmBackend
        from synthflow.exceptions import SynBackendError
        df = make_sensor_df(n=200)
        with pytest.raises(SynBackendError):
            TsgmBackend().run(
                df=df,
                signal_cols=["accel_x"],
                model_name="GPT5",
            )

    def test_run_with_nans_raises(self):
        from synthflow.backends import TsgmBackend
        from synthflow.exceptions import SynBackendError
        df = make_sensor_df(n=200)
        df.loc[0, "accel_x"] = np.nan
        with pytest.raises(SynBackendError):
            TsgmBackend().run(
                df=df,
                signal_cols=["accel_x"],
                model_name="GaussianProcess",
            )

    def test_run_reproducible_gp(self):
        from synthflow.backends import TsgmBackend
        df = make_sensor_df(n=200)
        kwargs = dict(
            df=df, signal_cols=["accel_x"],
            timestamp_col=None, model_name="GaussianProcess",
            n_samples=50, random_seed=7,
        )
        out1 = TsgmBackend().run(**kwargs)
        out2 = TsgmBackend().run(**kwargs)
        pd.testing.assert_frame_equal(out1, out2)
