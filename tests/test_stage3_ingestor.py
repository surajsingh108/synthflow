"""
Stage 3 tests – Ingestor

Tests that SynIngestor:
  - loads CSV, Excel, and JSON correctly
  - detects timestamp column and signal columns
  - infers sampling rate from timestamps
  - validates data and produces warnings
  - raises SynIngestError on bad inputs
  - accepts a DataFrame directly

DO NOT MODIFY THIS FILE.
Fix the implementation, not the tests.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# – fixtures –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

def make_sensor_df(
    n_rows: int = 200,
    freq_hz: float = 100.0,
    with_missing: bool = False,
) -> pd.DataFrame:
    """Create a realistic sensor DataFrame for testing."""
    t = pd.date_range(
        start="2024-01-01",
        periods=n_rows,
        freq=f"{int(1000 / freq_hz)}ms",
    )
    df = pd.DataFrame({
        "timestamp": t,
        "accel_x": np.random.randn(n_rows),
        "accel_y": np.random.randn(n_rows),
        "accel_z": np.random.randn(n_rows),
        "temperature": np.random.uniform(20, 80, n_rows),
    })
    if with_missing:
        idx = np.random.choice(n_rows, size=int(n_rows * 0.05), replace=False)
        df.loc[idx, "accel_x"] = np.nan
    return df


@pytest.fixture
def sensor_df():
    return make_sensor_df()


@pytest.fixture
def csv_file(tmp_path, sensor_df):
    path = tmp_path / "sensor.csv"
    sensor_df.to_csv(path, index=False)
    return path


@pytest.fixture
def excel_file(tmp_path, sensor_df):
    path = tmp_path / "sensor.xlsx"
    sensor_df.to_excel(path, index=False)
    return path


@pytest.fixture
def json_file(tmp_path, sensor_df):
    path = tmp_path / "sensor.json"
    records = sensor_df.assign(
        timestamp=sensor_df["timestamp"].astype(str)
    ).to_dict(orient="records")
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


@pytest.fixture
def json_tsdb_file(tmp_path, sensor_df):
    """InfluxDB-style JSON export with 'data' key."""
    path = tmp_path / "sensor_tsdb.json"
    records = sensor_df.assign(
        timestamp=sensor_df["timestamp"].astype(str)
    ).to_dict(orient="records")
    payload = {
        "columns": list(sensor_df.columns),
        "data": records,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# – 1. Import –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

class TestImport:
    def test_syningestor_importable(self):
        from synthflow.ingestor import SynIngestor
        assert SynIngestor is not None

    def test_loadresult_importable(self):
        from synthflow.ingestor.ingestor import LoadResult
        assert LoadResult is not None


# – 2. CSV loading –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

class TestCSVLoading:
    def test_loads_csv_file(self, csv_file):
        from synthflow.ingestor import SynIngestor
        result = SynIngestor().load(csv_file)
        assert result.data is not None

    def test_csv_correct_row_count(self, csv_file):
        from synthflow.ingestor import SynIngestor
        result = SynIngestor().load(csv_file)
        assert result.n_rows == 200

    def test_csv_correct_col_count(self, csv_file):
        from synthflow.ingestor import SynIngestor
        result = SynIngestor().load(csv_file)
        assert result.n_cols == 5  # timestamp + 4 signals

    def test_csv_source_label(self, csv_file):
        from synthflow.ingestor import SynIngestor
        result = SynIngestor().load(csv_file)
        assert "sensor.csv" in result.source

    def test_csv_as_string_path(self, csv_file):
        from synthflow.ingestor import SynIngestor
        result = SynIngestor().load(str(csv_file))
        assert result.n_rows == 200


# – 3. Excel loading –––––––––––––––––––––––––––––––––––––––––––––––––––––––––

class TestExcelLoading:
    def test_loads_excel_file(self, excel_file):
        from synthflow.ingestor import SynIngestor
        result = SynIngestor().load(excel_file)
        assert result.data is not None

    def test_excel_correct_row_count(self, excel_file):
        from synthflow.ingestor import SynIngestor
        result = SynIngestor().load(excel_file)
        assert result.n_rows == 200

    def test_excel_source_label(self, excel_file):
        from synthflow.ingestor import SynIngestor
        result = SynIngestor().load(excel_file)
        assert "sensor.xlsx" in result.source


# – 4. JSON loading ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

class TestJSONLoading:
    def test_loads_json_array(self, json_file):
        from synthflow.ingestor import SynIngestor
        result = SynIngestor().load(json_file)
        assert result.data is not None

    def test_json_correct_row_count(self, json_file):
        from synthflow.ingestor import SynIngestor
        result = SynIngestor().load(json_file)
        assert result.n_rows == 200

    def test_loads_json_tsdb_format(self, json_tsdb_file):
        from synthflow.ingestor import SynIngestor
        result = SynIngestor().load(json_tsdb_file)
        assert result.n_rows == 200

    def test_json_tsdb_has_correct_columns(self, json_tsdb_file):
        from synthflow.ingestor import SynIngestor
        result = SynIngestor().load(json_tsdb_file)
        assert "accel_x" in result.data.columns


# – 5. DataFrame input ––––––––––––––––––––––––––––––––––––––––––––––––––––––

class TestDataFrameInput:
    def test_accepts_dataframe_directly(self, sensor_df):
        from synthflow.ingestor import SynIngestor
        result = SynIngestor().load(sensor_df)
        assert result.n_rows == 200

    def test_dataframe_source_label(self, sensor_df):
        from synthflow.ingestor import SynIngestor
        result = SynIngestor().load(sensor_df)
        assert result.source == "dataframe"

    def test_does_not_mutate_input_dataframe(self, sensor_df):
        from synthflow.ingestor import SynIngestor
        original_shape = sensor_df.shape
        SynIngestor().load(sensor_df)
        assert sensor_df.shape == original_shape


# – 6. Timestamp detection ––––––––––––––––––––––––––––––––––––––––––––––––––

class TestTimestampDetection:
    def test_detects_timestamp_column(self, sensor_df):
        from synthflow.ingestor import SynIngestor
        result = SynIngestor().load(sensor_df)
        assert result.timestamp_col == "timestamp"

    def test_detects_ts_column_by_name(self):
        from synthflow.ingestor import SynIngestor
        df = pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=100, freq="10ms"),
            "value": np.random.randn(100),
        })
        result = SynIngestor().load(df)
        assert result.timestamp_col == "ts"

    def test_detects_datetime_dtype_column(self):
        from synthflow.ingestor import SynIngestor
        df = pd.DataFrame({
            "sensor_time": pd.date_range("2024-01-01", periods=100, freq="10ms"),
            "reading": np.random.randn(100),
        })
        result = SynIngestor().load(df)
        assert result.timestamp_col == "sensor_time"

    def test_no_timestamp_returns_none(self):
        from synthflow.ingestor import SynIngestor
        df = pd.DataFrame({
            "accel_x": np.random.randn(100),
            "accel_y": np.random.randn(100),
        })
        result = SynIngestor().load(df)
        assert result.timestamp_col is None

    def test_user_hint_overrides_detection(self, sensor_df):
        from synthflow.ingestor import SynIngestor
        # sensor_df has "timestamp" but we force "accel_x" as ts col
        result = SynIngestor().load(sensor_df, timestamp_col="accel_x")
        assert result.timestamp_col == "accel_x"


# – 7. Signal column detection ––––––––––––––––––––––––––––––––––––––––––––––

class TestSignalColDetection:
    def test_detects_signal_cols(self, sensor_df):
        from synthflow.ingestor import SynIngestor
        result = SynIngestor().load(sensor_df)
        assert "accel_x" in result.signal_cols
        assert "accel_y" in result.signal_cols
        assert "temperature" in result.signal_cols

    def test_timestamp_excluded_from_signal_cols(self, sensor_df):
        from synthflow.ingestor import SynIngestor
        result = SynIngestor().load(sensor_df)
        assert "timestamp" not in result.signal_cols

    def test_user_hint_overrides_signal_cols(self, sensor_df):
        from synthflow.ingestor import SynIngestor
        result = SynIngestor().load(
            sensor_df,
            signal_cols=["accel_x"]
        )
        assert result.signal_cols == ["accel_x"]


# – 8. Sampling rate inference ––––––––––––––––––––––––––––––––––––––––––––––

class TestSamplingRateInference:
    def test_infers_100hz(self):
        from synthflow.ingestor import SynIngestor
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=200, freq="10ms"),
            "value": np.random.randn(200),
        })
        result = SynIngestor().load(df)
        assert result.sampling_rate_hz == pytest.approx(100.0, rel=0.01)

    def test_infers_500hz(self):
        from synthflow.ingestor import SynIngestor
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=200, freq="2ms"),
            "value": np.random.randn(200),
        })
        result = SynIngestor().load(df)
        assert result.sampling_rate_hz == pytest.approx(500.0, rel=0.01)

    def test_no_timestamp_sampling_rate_none(self):
        from synthflow.ingestor import SynIngestor
        df = pd.DataFrame({
            "accel_x": np.random.randn(100),
            "accel_y": np.random.randn(100),
        })
        result = SynIngestor().load(df)
        assert result.sampling_rate_hz is None


# – 9. Missing data stats –––––––––––––––––––––––––––––––––––––––––––––––––––

class TestMissingStats:
    def test_missing_pct_zero_for_clean_data(self, sensor_df):
        from synthflow.ingestor import SynIngestor
        result = SynIngestor().load(sensor_df)
        assert result.missing_pct == 0.0

    def test_missing_pct_nonzero_for_dirty_data(self):
        from synthflow.ingestor import SynIngestor
        df = make_sensor_df(with_missing=True)
        result = SynIngestor().load(df)
        assert result.missing_pct > 0.0


# – 10. Error handling ––––––––––––––––––––––––––––––––––––––––––––––––––––––

class TestErrorHandling:
    def test_missing_file_raises_syn_ingest_error(self, tmp_path):
        from synthflow.ingestor import SynIngestor
        from synthflow.exceptions import SynIngestError
        with pytest.raises(SynIngestError):
            SynIngestor().load(tmp_path / "nonexistent.csv")

    def test_unsupported_format_raises_syn_ingest_error(self, tmp_path):
        from synthflow.ingestor import SynIngestor
        from synthflow.exceptions import SynIngestError
        p = tmp_path / "data.parquet"
        p.write_text("fake")
        with pytest.raises(SynIngestError):
            SynIngestor().load(p)

    def test_empty_csv_raises_syn_ingest_error(self, tmp_path):
        from synthflow.ingestor import SynIngestor
        from synthflow.exceptions import SynIngestError
        p = tmp_path / "empty.csv"
        p.write_text("")
        with pytest.raises(SynIngestError):
            SynIngestor().load(p)

    def test_empty_dataframe_raises_syn_ingest_error(self):
        from synthflow.ingestor import SynIngestor
        from synthflow.exceptions import SynIngestError
        with pytest.raises(SynIngestError):
            SynIngestor().load(pd.DataFrame())


# – 11. Validator warnings ––––––––––––––––––––––––––––––––––––––––––––––––––

class TestValidatorWarnings:
    def test_small_dataset_produces_warning(self):
        from synthflow.ingestor import SynIngestor
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="10ms"),
            "value": np.random.randn(10),
        })
        result = SynIngestor().load(df)
        assert any("rows" in w.lower() for w in result.warnings)

    def test_clean_large_dataset_no_warnings(self, sensor_df):
        from synthflow.ingestor import SynIngestor
        result = SynIngestor().load(sensor_df)
        assert result.warnings == []

    def test_constant_column_produces_warning(self):
        from synthflow.ingestor import SynIngestor
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="10ms"),
            "flat": np.ones(100),
            "signal": np.random.randn(100),
        })
        result = SynIngestor().load(df)
        assert any("constant" in w.lower() for w in result.warnings)

    def test_high_missing_produces_warning(self):
        from synthflow.ingestor import SynIngestor
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="10ms"),
            "bad_col": [np.nan] * 70 + list(np.random.randn(30)),
        })
        result = SynIngestor().load(df)
        assert any("missing" in w.lower() for w in result.warnings)

    def test_duplicate_rows_produces_warning(self):
        from synthflow.ingestor import SynIngestor
        base = make_sensor_df(n_rows=100)
        df = pd.concat([base, base.head(10)], ignore_index=True)
        result = SynIngestor().load(df)
        assert any("duplicate" in w.lower() for w in result.warnings)
